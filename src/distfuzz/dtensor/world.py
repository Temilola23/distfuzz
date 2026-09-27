import json
import os
import selectors
import socket
import subprocess
import sys
import time


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


class World:
    def __init__(self, n, logdir, cov=True, startup_timeout=90):
        self.n, self.logdir, self.cov = n, logdir, cov
        self.startup_timeout = startup_timeout
        self.procs = []
        self.restarts = 0
        os.makedirs(logdir, exist_ok=True)
        self.start()

    def start(self):
        port = free_port()
        env = dict(os.environ, OMP_NUM_THREADS="1", PYTHONUNBUFFERED="1")
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
                ],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=log,
                text=True,
                bufsize=1,
                env=env,
            )
            self.procs.append(p)
        res = self._collect(self.startup_timeout)
        if res is None or not all(isinstance(r, dict) and r.get("ready") for r in res):
            self.kill()
            raise RuntimeError(f"world failed to start: {res}")

    def _collect(self, timeout):
        sel = selectors.DefaultSelector()
        for i, p in enumerate(self.procs):
            sel.register(p.stdout, selectors.EVENT_READ, i)
        out = [None] * self.n
        deadline = time.time() + timeout
        while any(o is None for o in out):
            rem = deadline - time.time()
            if rem <= 0:
                sel.close()
                return None
            for key, _ in sel.select(rem):
                i = key.data
                line = self.procs[i].stdout.readline()
                if not line:
                    sel.close()
                    out[i] = {"dead": True, "rc": self.procs[i].poll()}
                    return out
                out[i] = json.loads(line)
                sel.unregister(self.procs[i].stdout)
        sel.close()
        return out

    def run(self, prog, timeout):
        line = json.dumps(prog) + "\n"
        try:
            for p in self.procs:
                p.stdin.write(line)
                p.stdin.flush()
        except (BrokenPipeError, OSError):
            self.restart()
            return "crash", None
        res = self._collect(timeout)
        if res is None:
            self.restart()
            return "timeout", None
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
