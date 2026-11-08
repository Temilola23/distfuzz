import os
import subprocess
import sys

from distfuzz.dtensor.world import World as DTWorld
from distfuzz.dtensor.world import free_port


class World(DTWorld):
    """The DTensor world, running ``distfuzz.compile.worker``. Each rank writes the leg it is in to
    ``<logdir>/rank<r>.leg``; ``legs()`` reads them back after a hang."""

    def __init__(self, n, logdir, startup_timeout=120, pg_timeout=180, shared_cache=False):
        self.pg_timeout = pg_timeout
        self.shared_cache = shared_cache
        super().__init__(n, logdir, cov=False, startup_timeout=startup_timeout)

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
            renv = dict(env)
            if not self.shared_cache:
                # ranks on one device get the same AOTAutograd cache key for DTensor inputs and load each other's
                # graphs (VERIFIED.md section 53); a cache dir per rank keeps that out of every other finding
                renv["TORCHINDUCTOR_CACHE_DIR"] = os.path.join(self.logdir, f"inductor_cache_rank{r}")
            cmd = [sys.executable, "-m", "distfuzz.compile.worker", "--rank", str(r), "--world", str(self.n)]
            cmd += ["--port", str(port), "--legdir", self.logdir, "--pg-timeout", str(self.pg_timeout)]
            p = subprocess.Popen(
                cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log, text=True, bufsize=1, env=renv
            )
            self.procs.append(p)
        res = self._collect(self.startup_timeout)
        if res is None or not all(isinstance(r, dict) and r.get("ready") for r in res):
            self.kill()
            raise RuntimeError(f"world failed to start: {res}")

    def legs(self):
        out = []
        for r in range(self.n):
            try:
                with open(os.path.join(self.logdir, f"rank{r}.leg")) as f:
                    out.append(f.read().strip())
            except OSError:
                out.append("?")
        return out

    def run(self, prog, timeout):
        for r in range(self.n):
            try:
                os.remove(os.path.join(self.logdir, f"rank{r}.leg"))
            except OSError:
                pass
        status, res = super().run(prog, timeout)
        return status, res
