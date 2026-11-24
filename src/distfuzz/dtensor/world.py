import json
import os
import selectors
import socket
import subprocess
import sys
import time

# see collectives.executor.NCCL_ENV: wait() raises after the group timeout instead of blocking the rank forever
NCCL_ENV = {"TORCH_NCCL_BLOCKING_WAIT": "1", "TORCH_NCCL_ASYNC_ERROR_HANDLING": "1", "TORCH_NCCL_DUMP_ON_TIMEOUT": "0"}
# Under non-blocking communicators every equal-split all_to_all_single raises "NCCL Error 7: NCCL operation in
# progress" (GPU run 2, N1), which DTensor's Shard(i) -> Shard(j) and EP token exchanges reach; the error was not
# sticky, so later programs ran on corrupted state. Workers never inherit the setting from the parent.
NCCL_NONBLOCKING_VARS = ("TORCH_NCCL_USE_COMM_NONBLOCKING", "TORCH_NCCL_NONBLOCKING_TIMEOUT")


def worker_env(device):
    """Environment for a DTensor or training worker rank."""
    env = dict(os.environ, OMP_NUM_THREADS="1", PYTHONUNBUFFERED="1")
    if device == "cuda":
        env.update(NCCL_ENV)
        for k in NCCL_NONBLOCKING_VARS:
            env.pop(k, None)
    return env


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


class World:
    # loop_secs: a torch.distributed frame spinning this long is a hang (view_groups calls take microseconds);
    # coll_secs: a collective waiting this long is a hang, and must stay below the workers' 20 s Gloo timeout
    def __init__(self, n, logdir, cov=True, startup_timeout=90, loop_secs=5.0, coll_secs=12.0, grace=3.0, device="cpu"):
        if device not in ("cpu", "cuda"):
            raise ValueError(f"device must be cpu (gloo) or cuda (nccl), got {device}")
        self.n, self.logdir, self.cov, self.device = n, logdir, cov, device
        self.startup_timeout = startup_timeout
        self.loop_secs, self.coll_secs, self.grace = loop_secs, coll_secs, grace
        self.procs = []
        self.restarts = 0
        os.makedirs(logdir, exist_ok=True)
        self.start()

    def start(self):
        port = free_port()
        env = worker_env(self.device)
        self.procs = []
        for f in getattr(self, "_logs", []):
            f.close()
        self._logs = []
        for r in range(self.n):
            log = open(os.path.join(self.logdir, f"rank{r}.log"), "a")
            self._logs.append(log)
            p = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "distfuzz.dtensor.worker",
                    "--rank",
                    str(r),
                    "--world",
                    str(self.n),
                    "--port",
                    str(port),
                    "--cov",
                    str(int(self.cov)),
                    "--device",
                    self.device,
                ],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=log,
                text=True,
                bufsize=1,
                env=env,
            )
            self.procs.append(p)
        res, _ = self._collect(self.startup_timeout)
        if res is None or not all(isinstance(r, dict) and r.get("ready") for r in res):
            self.kill()
            raise RuntimeError(f"world failed to start: {res}")

    def _collect(self, timeout):
        sel = selectors.DefaultSelector()
        for i, p in enumerate(self.procs):
            sel.register(p.stdout, selectors.EVENT_READ, i)
        out = [None] * self.n
        deadline = time.time() + timeout
        hung = False
        while any(o is None for o in out):
            rem = deadline - time.time()
            if rem <= 0:
                sel.close()
                return (out if hung else None), True
            for key, _ in sel.select(rem):
                i = key.data
                line = self.procs[i].stdout.readline()
                if not line:
                    sel.close()
                    out[i] = {"dead": True, "rc": self.procs[i].poll()}
                    return out, False
                out[i] = json.loads(line)
                sel.unregister(self.procs[i].stdout)
                if "hang" in out[i] and not hung:
                    # the other ranks' watchdogs fire within a moment if they are stuck too; wait that long, no more
                    hung = True
                    deadline = min(deadline, time.time() + self.grace)
        sel.close()
        return out, False

    def run(self, prog, timeout):
        wd = dict(deadline=max(timeout - 2.0, timeout * 0.8), loop=self.loop_secs, coll=self.coll_secs)
        line = json.dumps(dict(prog, wd=wd)) + "\n"
        try:
            for p in self.procs:
                p.stdin.write(line)
                p.stdin.flush()
        except (BrokenPipeError, OSError):
            self.restart()
            return "crash", None
        res, _ = self._collect(timeout)
        if res is None:
            self.restart()
            return "timeout", None
        if any(r is not None and "hang" in r for r in res):
            self.restart()
            return "hang", [r["hang"] if r is not None and "hang" in r else None for r in res]
        if any(r is None or r.get("dead") for r in res):
            rcs = [p.poll() for p in self.procs]
            self.restart()
            return "crash", [{"rcs": rcs}]
        if any(r.get("aborted") for r in res):
            self.restart()
            return "aborted", res
        return "ok", res

    def kill(self):
        for p in self.procs:
            try:
                p.kill()
            except Exception:
                pass
        for p in self.procs:
            try:
                p.wait(timeout=5)
            except Exception:
                pass

    def restart(self):
        self.kill()
        self.restarts += 1
        for _ in range(3):
            try:
                self.start()
                return
            except Exception:
                self.kill()
                time.sleep(2)
        raise RuntimeError("could not restart world")

    def close(self):
        try:
            for p in self.procs:
                p.stdin.write(json.dumps({"cmd": "exit"}) + "\n")
                p.stdin.flush()
        except Exception:
            pass
        time.sleep(0.5)
        self.kill()
