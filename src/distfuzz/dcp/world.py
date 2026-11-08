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


def stack_location(txt):
    """API-level location of the main thread in the last faulthandler dump: the first torch frame called from
    distfuzz code, and the innermost torch.distributed frame, e.g. 'state_dict.py:set_state_dict >
    distributed_c10d.py:broadcast'."""
    i = txt.rfind("Current thread")
    if i < 0:
        i = txt.rfind("most recent call first")
    if i < 0:
        return None
    frames = []
    for ln in txt[i:].splitlines()[1:]:
        ln = ln.strip()
        if not ln.startswith("File "):
            if frames:
                break  # end of the first (current) thread's stack
            continue
        try:
            path = ln.split('"')[1]
            func = ln.rsplit(" in ", 1)[1]
        except IndexError:
            continue
        frames.append((path, func))
    frames.reverse()  # outermost first
    api, inner = None, None
    for k, (path, func) in enumerate(frames):
        if "distfuzz" in path:
            continue
        if api is None and k and "distfuzz" in frames[k - 1][0] and "torch" in path:
            api = os.path.basename(path) + ":" + func
        if os.sep + "distributed" + os.sep in path:
            inner = os.path.basename(path) + ":" + func
    if api is None and inner is None and frames:
        api = os.path.basename(frames[-1][0]) + ":" + frames[-1][1]
    return " > ".join(x for x in (api, inner) if x)


class World:
    def __init__(self, n, logdir, cov=True, startup_timeout=90):
        self.n, self.logdir, self.cov = n, logdir, cov
        self.startup_timeout = startup_timeout
        self.procs = []
        self.restarts = 0
        os.makedirs(logdir, exist_ok=True)
        for k in range(3):  # free_port can race with another world's TCPStore (EADDRINUSE)
            try:
                self.start()
                return
            except RuntimeError:
                if k == 2:
                    raise
                time.sleep(2)

    def start(self):
        port = free_port()
        env = dict(os.environ, OMP_NUM_THREADS="1", PYTHONUNBUFFERED="1")
        self.procs = []
        for r in range(self.n):
            log = open(os.path.join(self.logdir, f"rank{r}.log"), "a")
            p = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "distfuzz.dcp.worker",
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
        if res is None or not all(r.get("ready") for r in res):
            self.kill()
            raise RuntimeError("world failed to start")

    def _err_seen(self, offs):
        for r in range(self.n):
            try:
                with open(os.path.join(self.logdir, f"rank{r}.log"), errors="replace") as f:
                    f.seek(offs[r])
                    if "DISTFUZZ_ERR" in f.read():
                        return True
            except OSError:
                pass
        return False

    def _collect(self, timeout, offs=None, err_grace=None):
        """Wait for one line per rank. With err_grace, give up err_grace seconds after any rank logged an error
        (DISTFUZZ_ERR) while others have not returned: those ranks are stuck in a collective."""
        sel = selectors.DefaultSelector()
        for i, p in enumerate(self.procs):
            sel.register(p.stdout, selectors.EVENT_READ, i)
        out = [None] * self.n
        deadline = time.time() + timeout
        err_at = None
        while any(o is None for o in out):
            rem = deadline - time.time()
            if err_grace is not None and offs is not None:
                if err_at is None and self._err_seen(offs):
                    err_at = time.time()
                if err_at is not None:
                    rem = min(rem, err_at + err_grace - time.time())
            if rem <= 0:
                sel.close()
                self.last_partial = out
                return None
            for key, _ in sel.select(min(rem, 1.0)):
                i = key.data
                line = self.procs[i].stdout.readline()
                if not line:  # EOF: worker died
                    sel.close()
                    out[i] = {"dead": True, "rc": self.procs[i].poll()}
                    return out
                out[i] = json.loads(line)
                sel.unregister(self.procs[i].stdout)
        sel.close()
        return out

    def log_sizes(self):
        out = []
        for r in range(self.n):
            try:
                out.append(os.path.getsize(os.path.join(self.logdir, f"rank{r}.log")))
            except OSError:
                out.append(0)
        return out

    def hang_report(self, offs, partial):
        """Per rank: did it return, the error it printed (DISTFUZZ_ERR), and its faulthandler stack."""
        rep = []
        for r in range(self.n):
            try:
                with open(os.path.join(self.logdir, f"rank{r}.log"), errors="replace") as f:
                    f.seek(offs[r])
                    txt = f.read()
            except OSError:
                txt = ""
            marks = [ln.split(" phase=", 1)[-1].strip() for ln in txt.splitlines() if ln.startswith("DISTFUZZ_PHASE")]
            errs = [ln.split(" msg=", 1)[-1] for ln in txt.splitlines() if ln.startswith("DISTFUZZ_ERR")]
            phases = [
                ln.split(" phase=", 1)[-1].split(" ")[0] for ln in txt.splitlines() if ln.startswith("DISTFUZZ_ERR")
            ]
            rep.append(
                dict(
                    rank=r,
                    returned=bool(partial and partial[r] is not None),
                    err=errs[-1] if errs else None,
                    err_phase=phases[-1] if phases else None,
                    phase=marks[-1] if marks else None,
                    loc=stack_location(txt),
                )
            )
        return rep

    def run(self, prog, timeout, err_grace=None):
        line = json.dumps(prog) + "\n"
        offs = self.log_sizes()
        self.last_partial = None
        try:
            for p in self.procs:
                p.stdin.write(line)
                p.stdin.flush()
        except (BrokenPipeError, OSError):
            self.restart()
            return "crash", None
        res = self._collect(timeout, offs, err_grace)
        if res is None:
            import signal

            for p in self.procs:
                try:
                    p.send_signal(signal.SIGUSR1)  # faulthandler stack dump into rank log
                except Exception:
                    pass
            time.sleep(2)
            rep = self.hang_report(offs, self.last_partial)
            self.restart()
            return "timeout", rep
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
            except RuntimeError:
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
