from __future__ import annotations

import json
import os
import subprocess
import threading
import time
import uuid

from .versions import by_key

IMAGE_PREFIX = "distfuzz-repro/torch"
DOCKERFILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Dockerfile.torch")
CACHE = os.path.join(".cache", "repro")
LIMITS = ["--memory", "3g", "--cpus", "4"]
SLOTS = threading.BoundedSemaphore(int(os.environ.get("DISTFUZZ_REPRO_SLOTS", "2")))  # max concurrent containers
# no network: make the container hostname resolve to loopback so Gloo/torchrun do not stall on DNS
NET = [
    "--network",
    "none",
    "--hostname",
    "repro",
    "--add-host",
    "repro:127.0.0.1",
    "-e",
    "GLOO_SOCKET_IFNAME=lo",
    "--shm-size",
    "512m",
]


def image_name(key):
    return f"{IMAGE_PREFIX}:{key}"


def _docker(args, timeout=None, retries=3, **kw):
    """Run a docker CLI command; retry when the daemon is restarting (user may resize Docker once)."""
    last = None
    for attempt in range(retries):
        p = subprocess.run(["docker"] + args, capture_output=True, text=True, timeout=timeout, **kw)
        daemon_down = p.returncode != 0 and (
            "Cannot connect to the Docker daemon" in p.stderr
            or "error during connect" in p.stderr
            or "context canceled" in p.stderr
        )
        if not daemon_down:
            return p
        last = p
        time.sleep(20 * (attempt + 1))
    return last


def image_exists(key):
    return _docker(["image", "inspect", image_name(key)]).returncode == 0


def build_image(key, force=False, log=print):
    """Build (or reuse cached) image for a matrix entry. Returns dict with pinned versions."""
    m = by_key(key)
    os.makedirs(CACHE, exist_ok=True)
    meta_path = os.path.join(CACHE, f"image-{key}.json")
    if image_exists(key) and not force and os.path.exists(meta_path):
        return json.load(open(meta_path))
    log(f"[build] {image_name(key)} ({m['torch_spec']})")
    args = [
        "build",
        "-f",
        DOCKERFILE,
        "-t",
        image_name(key),
        "--build-arg",
        f"TORCH_SPEC={m['torch_spec']}",
        "--build-arg",
        f"NUMPY_SPEC={m['numpy']}",
        "--build-arg",
        f"INDEX_URL={m['index']}",
        "--build-arg",
        f"PIP_EXTRA={m['pip_extra']}",
        os.path.dirname(DOCKERFILE),
    ]
    p = _docker(args, timeout=3600)
    open(os.path.join(CACHE, f"build-{key}.log"), "w").write(p.stdout + p.stderr)
    if p.returncode != 0:
        raise RuntimeError(f"image build failed for {key}: see {CACHE}/build-{key}.log")
    freeze = _docker(["run", "--rm", "--network", "none", image_name(key), "cat", "/pinned-requirements.txt"]).stdout
    ver = _docker(["run", "--rm", "--network", "none", image_name(key), "cat", "/torch-version.txt"]).stdout.strip()
    digest = _docker(["image", "inspect", image_name(key), "--format", "{{.Id}}"]).stdout.strip()
    meta = dict(
        key=key,
        image=image_name(key),
        image_id=digest,
        torch_version=ver,
        requirements=freeze.strip().splitlines(),
        spec=m,
    )
    json.dump(meta, open(meta_path, "w"), indent=1)
    return meta


def run_fresh(key, workdir, argv, timeout=150, name=None):
    """Run `argv` once in a FRESH container of image `key`, with `workdir` mounted read-only at /repro.
    Returns dict(rc, out, secs, timed_out)."""
    workdir = os.path.abspath(workdir)
    name = name or f"distfuzz-repro-{uuid.uuid4().hex[:10]}"
    cmd = [
        "run",
        "--rm",
        "--name",
        name,
        *NET,
        *LIMITS,
        "-v",
        f"{workdir}:/repro:ro",
        "-w",
        "/repro",
        image_name(key),
        "timeout",
        "-s",
        "KILL",
        str(int(timeout)),
        *argv,
    ]
    with SLOTS:
        t0 = time.time()
        try:
            p = _docker(cmd, timeout=timeout + 90)
            rc, out = p.returncode, (p.stdout or "") + (p.stderr or "")
        except subprocess.TimeoutExpired:
            _docker(["rm", "-f", name])
            rc, out = 137, ""
        secs = time.time() - t0
    # 137 = `timeout -s KILL` (hang) or the container OOM-killed (infra); 125 = docker error
    timed_out = rc == 137 and secs >= timeout - 1
    return dict(
        rc=rc, out=out, secs=round(secs, 2), timed_out=timed_out, infra=rc == 125 or (rc == 137 and not timed_out)
    )


class Box:
    """A long-lived container (minimization / bisection batches); every exec is a fresh process tree."""

    def __init__(self, key, workdir):
        self.key, self.workdir = key, os.path.abspath(workdir)
        self.name = f"distfuzz-repro-box-{key.replace('.', '_')}-{uuid.uuid4().hex[:6]}"

    def __enter__(self):
        SLOTS.acquire()
        p = _docker(
            [
                "run",
                "-d",
                "--name",
                self.name,
                *NET,
                *LIMITS,
                "-v",
                f"{self.workdir}:/repro:ro",
                "-w",
                "/repro",
                image_name(self.key),
                "sleep",
                "infinity",
            ]
        )
        if p.returncode != 0:
            SLOTS.release()
            raise RuntimeError(p.stderr)
        return self

    def run_cmd(self, argv, timeout=150):
        t0 = time.time()
        cmd = ["exec", self.name, "timeout", "-s", "KILL", str(int(timeout)), *argv]
        try:
            p = _docker(cmd, timeout=timeout + 90)
            rc, out = p.returncode, (p.stdout or "") + (p.stderr or "")
        except subprocess.TimeoutExpired:
            rc, out = 137, ""
        # orphaned rank processes (the launcher was killed) must not leak into the next run
        _docker(["exec", self.name, "sh", "-c", "pkill -9 -f 'python|torchrun' || true"])
        secs = time.time() - t0
        timed_out = rc == 137 and secs >= timeout - 1
        return dict(
            rc=rc,
            out=out,
            secs=round(secs, 2),
            timed_out=timed_out,
            infra=rc in (125, 126) or "No such container" in out or (rc == 137 and not timed_out),
        )

    def __exit__(self, *a):
        _docker(["rm", "-f", self.name])
        SLOTS.release()
