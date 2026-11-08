from __future__ import annotations

import faulthandler
import math
import multiprocessing as mp
import os
import re
import signal
import socket
import sys
import tempfile
import time
import traceback
from contextlib import suppress
from multiprocessing.connection import wait as conn_wait

from .desc import BACKENDS

# written to each rank's log before every program, so a crash is read from its own program's output only
MARK = "@@ distfuzz program"
# per-collective. Measured cross-rank arrival skew is <= 0.12 s (FINDINGS.md); a call that raises on some ranks
# leaves the others waiting this long (Gloo has no way to wake them), which dominated run time at 2.0
TIMEOUT = 0.5
# a new finding that involved a timeout must reproduce at this one before it is reported (Fuzzer.confirm)
CONFIRM_TIMEOUT = 4.0
# NCCL sets up a communicator per new group (and per p2p pair on first use) and a cold kernel launch is slow; 0.5 s
# would time out healthy collectives. Not measured on T4 yet: the notebook's calibration cell checks it.
NCCL_TIMEOUT = 3.0
NCCL_CONFIRM_TIMEOUT = 15.0
# set before torch is imported in a rank (desc.NCCL_SEMANTICS, "async errors")
NCCL_ENV = {
    "TORCH_NCCL_BLOCKING_WAIT": "1",
    "TORCH_NCCL_ASYNC_ERROR_HANDLING": "1",
    "TORCH_NCCL_DUMP_ON_TIMEOUT": "0",
    "TORCH_NCCL_ENABLE_MONITORING": "1",
    "TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC": "60",
    # Blocking wait only bounds Work.wait(). The first collective on a communicator (and the first p2p to a peer)
    # connects inside ncclGroupEnd, and new_group splits the world communicator; both block the host until every
    # peer arrives, so a desynchronised program sat there until the session deadline (Kaggle run 1: 97 of 748
    # programs, ~85% of the hour). Non-blocking communicators make PyTorch poll those calls and give up after
    # TORCH_NCCL_NONBLOCKING_TIMEOUT (set per session from the collective timeout). Opt out with
    # DISTFUZZ_NCCL_NONBLOCKING=0 (pytorch#153960: a non-blocking hang in NCCL 2.26).
    "TORCH_NCCL_USE_COMM_NONBLOCKING": "1",
}


def nccl_env(timeout):
    env = dict(NCCL_ENV, TORCH_NCCL_NONBLOCKING_TIMEOUT=str(max(2, math.ceil(2 * timeout))))
    if os.environ.get("DISTFUZZ_NCCL_NONBLOCKING") == "0":
        env.pop("TORCH_NCCL_USE_COMM_NONBLOCKING")
        env.pop("TORCH_NCCL_NONBLOCKING_TIMEOUT")
    return env


# an NCCL error or timeout leaves the communicators (and possibly in-flight kernels) in an undefined state; PyTorch's
# own message says later GPU work "might run on corrupted/incomplete data". Ranks are restarted after such a program.
NCCL_DIRTY = re.compile(
    r"DistBackendError|NCCL error|ncclInternalError|ncclSystemError|communicator was aborted|timed out|timing out"
    r"|Timed out|Watchdog"
)


def _free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def _rank_main(rank, world, port, conn, cfg):
    fd = os.open(os.path.join(cfg["logdir"], f"rank{rank}.log"), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    os.dup2(fd, 2)
    os.dup2(fd, 1)
    # Python stacks on a fatal signal, and on SIGUSR1 (sent by Session.run before it kills a stuck rank): without
    # them the run's SIGSEGV and hang findings had empty logs
    faulthandler.enable(all_threads=True)
    if hasattr(signal, "SIGUSR1"):
        faulthandler.register(signal.SIGUSR1, all_threads=True)
    os.environ.update(
        MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port), GLOO_SOCKET_IFNAME="lo0" if sys.platform == "darwin" else "lo"
    )
    if cfg.get("detail"):
        os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"
    backend, device = cfg.get("backend", "gloo"), cfg.get("device", "cpu")
    if backend == "nccl":
        for k, v in nccl_env(cfg["timeout"]).items():
            os.environ.setdefault(k, v)
    import warnings

    warnings.filterwarnings("ignore")
    from datetime import timedelta

    import torch
    import torch.distributed as dist
    import torch.distributed.distributed_c10d as c10d

    torch.set_num_threads(1)
    from . import interp

    timeout = cfg["timeout"]
    dev = None
    if device == "cuda":
        dev = torch.device("cuda", rank % torch.cuda.device_count())
        torch.cuda.set_device(dev)
    # device_id makes NCCL init eagerly, so new_group can split the world communicator instead of creating one
    dist.init_process_group(
        backend, rank=rank, world_size=world, timeout=timedelta(seconds=max(timeout, 10)), device_id=dev
    )
    covs = []
    if "py" in cfg["signals"] and hasattr(sys, "monitoring"):  # Python < 3.12 (some notebook images): no coverage
        covs.append(interp.Coverage())
    if "cxx" in cfg["signals"]:
        from .cxxcov import CxxCoverage

        covs.append(CxxCoverage())
    conn.send(("ready", rank))
    nprog = 0
    while True:
        msg = conn.recv()
        if msg[0] == "exit":
            break
        _, prog, timeout = msg
        nprog += 1
        os.write(2, f"{MARK} {nprog}\n".encode())
        res = {}
        try:
            # resync the group-name counter so a desynchronised new_group() in an earlier program can't leak
            c10d._world.group_count = 1_000_000 + nprog * 1000
            pg = dist.new_group(list(range(world)), timeout=timedelta(seconds=timeout))
            status, env = interp.run_program(prog, rank, pg, timeout, device)
            res.update(status)
            for g in env.created_groups + [pg]:
                with suppress(Exception):
                    dist.destroy_process_group(g)
        except Exception:  # noqa: BLE001
            res["executor_error"] = traceback.format_exc()[-2000:]
        if covs:
            res["cov"] = set().union(*(c.take() for c in covs))
        conn.send(("result", rank, res))
    with suppress(Exception):
        dist.destroy_process_group()


SIGNALS = ("py", "cxx", "py+cxx")


def signals(coverage):
    """Coverage kinds to collect: False for none, True for Python lines, or one of SIGNALS."""
    if coverage is False:
        return set()
    if coverage is True:
        return {"py"}
    if coverage not in SIGNALS:
        raise ValueError(f"coverage must be a bool or one of {SIGNALS}, got {coverage!r}")
    return set(coverage.split("+"))


def check_backend(backend, device):
    if BACKENDS.get(backend) != device:
        raise ValueError(f"supported backend/device pairs are {BACKENDS}, got {backend}/{device}")


# a CUDA error poisons the rank's context: every later program would fail the same way
CUDA_STICKY = ("CUDA error", "device-side assert", "illegal memory access")


def _errors(r):
    r = r or {}
    return f"{r.get('exc')} {r.get('wait_exc')} {r.get('executor_error')}"


class Session:
    def __init__(
        self,
        world=4,
        timeout=TIMEOUT,
        detail=False,
        coverage=True,
        logdir=None,
        hang_deadline=None,
        backend="gloo",
        device="cpu",
    ):
        check_backend(backend, device)
        self.world = world
        self.backend, self.device = backend, device
        self.cfg = dict(
            backend=backend,
            device=device,
            timeout=timeout,
            detail=detail,
            signals=signals(coverage),
            logdir=logdir or tempfile.mkdtemp(prefix="distfuzz-"),
        )
        self.hang_deadline = hang_deadline or (6 * timeout + 10)
        self.procs, self.conns = [], []
        self.started = False
        self.restarts = 0

    def start(self, attempts=8):
        # Fault mode restarts the session constantly and a fresh rank sometimes dies during init
        # (port still in TIME_WAIT after a crash). Retry on a fresh port instead of ending the run.
        for attempt in range(attempts):
            try:
                self._spawn()
                self.started = True
                return
            except (EOFError, OSError, RuntimeError) as e:
                last = e
                self._kill()
                time.sleep(0.5 + 0.3 * attempt)
        raise RuntimeError(f"session failed to start after {attempts} attempts: {last}")

    def _spawn(self):
        ctx = mp.get_context("spawn")
        port = _free_port()
        self.procs, self.conns = [], []
        for r in range(self.world):
            a, b = ctx.Pipe()
            p = ctx.Process(target=_rank_main, args=(r, self.world, port, b, self.cfg), daemon=True)
            p.start()
            self.procs.append(p)
            self.conns.append(a)
        deadline = time.time() + 120
        for c in self.conns:
            if not c.poll(max(0.1, deadline - time.time())):
                raise RuntimeError("rank did not become ready")
            c.recv()

    def _kill(self):
        for p in self.procs:
            if p.is_alive():
                p.kill()
            p.join(timeout=3)
        for c in self.conns:
            c.close()

    def stop(self, hard=False):
        if not self.started:
            return
        if not hard:
            for c in self.conns:
                with suppress(Exception):
                    c.send(("exit",))
            for p in self.procs:
                p.join(timeout=3)
        for p in self.procs:
            if p.is_alive():
                p.kill()
                p.join(timeout=3)
        for c in self.conns:
            c.close()
        self.started = False

    def _log_tails(self, ranks, keep=8):
        d = self.cfg["logdir"]
        time.sleep(0.2)
        out = []
        for r in ranks:
            try:
                with open(os.path.join(d, f"rank{r}.log"), "rb") as f:
                    f.seek(0, 2)
                    f.seek(max(0, f.tell() - 65536))
                    log = f.read().decode(errors="replace")
                # the log spans every program since the rank started; earlier programs' errors are not this crash
                if MARK in log:
                    log = log.rsplit(MARK, 1)[1].split("\n", 1)[-1]
                tail = log.splitlines()
                lines = [ln for ln in tail if "Warning" not in ln and "warn(" not in ln][-keep:]
                out.append(f"[rank {r}] " + " | ".join(lines))
            except OSError:
                pass
        return "\n".join(out)

    def restart(self):
        self.stop(hard=True)
        self.restarts += 1
        self.start()

    def run(self, prog, timeout=None):
        """Run one program on every rank; `timeout` overrides the per-collective timeout for this program only."""
        if not self.started:
            self.start()
        to = timeout or self.cfg["timeout"]
        t0 = time.time()
        sent = True
        try:
            for c in self.conns:
                c.send(("run", prog, to))
        except OSError:
            sent = False
        dead = [r for r, p in enumerate(self.procs) if not p.is_alive()]
        if dead or not sent:
            # a rank died after reporting the previous program, e.g. a late abort from in-flight work
            out = {
                "kind": "crash",
                "late": True,
                "results": [{"died": True}] * self.world,
                "wall": 0.0,
                "stuck_ranks": [],
                "exitcodes": [p.exitcode for p in self.procs],
                "crash_log": self._log_tails(dead),
            }
            self.restart()
            return out
        results = [None] * self.world
        pending = {self.conns[r]: r for r in range(self.world)}
        dl = t0 + (self.hang_deadline if timeout is None else 6 * to + 10)
        kind = "ok"
        while pending:
            left = dl - time.time()
            if left <= 0:
                if kind != "crash":
                    kind = "hang"
                break
            ready = conn_wait(list(pending), timeout=min(left, 0.5))
            for c in ready:
                r = pending.pop(c)
                try:
                    results[r] = c.recv()[2]
                except (EOFError, OSError):
                    results[r] = {"died": True}
                    kind = "crash"
            for c, r in list(pending.items()):
                if not self.procs[r].is_alive():
                    results[r] = {"died": True, "exitcode": self.procs[r].exitcode}
                    pending.pop(c)
                    kind = "crash"
            if kind == "crash":
                dl = min(dl, time.time() + 2 * to + 2)
        out = {
            "kind": kind,
            "results": results,
            "wall": time.time() - t0,
            "stuck_ranks": sorted(pending.values()) if kind != "ok" else [],
        }
        if kind == "crash":
            out["crash_log"] = self._log_tails([r for r in range(self.world) if (results[r] or {}).get("died")], 30)
            out["exitcodes"] = [p.exitcode for p in self.procs]
        if kind == "hang":
            out["hang_log"] = self._dump_stacks(out["stuck_ranks"])
        if kind == "ok" and self.device == "cuda" and any(s in _errors(r) for r in results for s in CUDA_STICKY):
            out["cuda_sticky"] = True
        if kind == "ok" and self.backend == "nccl" and any(NCCL_DIRTY.search(_errors(r)) for r in results):
            out["nccl_dirty"] = True
        if kind != "ok" or out.get("cuda_sticky") or out.get("nccl_dirty"):
            self.restart()
        return out

    def _dump_stacks(self, ranks):
        """Ask the stuck ranks for their Python stacks (faulthandler on SIGUSR1) and return the log tails."""
        if not hasattr(signal, "SIGUSR1"):
            return ""
        for r in ranks:
            with suppress(Exception):
                os.kill(self.procs[r].pid, signal.SIGUSR1)
        time.sleep(1.0)
        return self._log_tails(ranks, 40)


def run_once(prog, world=4, timeout=TIMEOUT, detail=False, coverage=False, logdir=None, backend="gloo", device="cpu"):
    s = Session(world, timeout, detail, coverage, logdir, backend=backend, device=device)
    s.start()
    try:
        return s.run(prog)
    finally:
        s.stop(hard=False)
