from __future__ import annotations

import multiprocessing as mp
import os
import socket
import sys
import tempfile
import time
import traceback
from contextlib import suppress
from multiprocessing.connection import wait as conn_wait

TIMEOUT = 2.0  # per-collective; measured cross-rank arrival skew is <= 0.12 s (FINDINGS.md)


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
    os.environ.update(
        MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port), GLOO_SOCKET_IFNAME="lo0" if sys.platform == "darwin" else "lo"
    )
    if cfg.get("detail"):
        os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"
    import warnings

    warnings.filterwarnings("ignore")
    from datetime import timedelta

    import torch
    import torch.distributed as dist
    import torch.distributed.distributed_c10d as c10d

    torch.set_num_threads(1)
    from . import interp

    timeout = cfg["timeout"]
    dist.init_process_group("gloo", rank=rank, world_size=world, timeout=timedelta(seconds=max(timeout, 10)))
    cov = interp.Coverage() if cfg["coverage"] else None
    conn.send(("ready", rank))
    nprog = 0
    while True:
        msg = conn.recv()
        if msg[0] == "exit":
            break
        _, prog = msg
        nprog += 1
        res = {}
        try:
            # resync the group-name counter so a desynchronised new_group() in an earlier program can't leak
            c10d._world.group_count = 1_000_000 + nprog * 1000
            pg = dist.new_group(list(range(world)), timeout=timedelta(seconds=timeout))
            status, env = interp.run_program(prog, rank, pg, timeout)
            res.update(status)
            for g in env.created_groups + [pg]:
                with suppress(Exception):
                    dist.destroy_process_group(g)
        except Exception:  # noqa: BLE001
            res["executor_error"] = traceback.format_exc()[-2000:]
        if cov is not None:
            res["cov"] = cov.take()
        conn.send(("result", rank, res))
    with suppress(Exception):
        dist.destroy_process_group()


class Session:
    def __init__(self, world=4, timeout=TIMEOUT, detail=False, coverage=True, logdir=None, hang_deadline=None):
        self.world = world
        self.cfg = dict(
            timeout=timeout, detail=detail, coverage=coverage, logdir=logdir or tempfile.mkdtemp(prefix="distfuzz-")
        )
        self.hang_deadline = hang_deadline or (6 * timeout + 10)
        self.procs, self.conns = [], []
        self.started = False
        self.restarts = 0

    def start(self):
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
        self.started = True

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

    def _log_tails(self, ranks):
        d = self.cfg["logdir"]
        time.sleep(0.2)
        out = []
        for r in ranks:
            try:
                with open(os.path.join(d, f"rank{r}.log"), "rb") as f:
                    f.seek(0, 2)
                    f.seek(max(0, f.tell() - 3000))
                    tail = f.read().decode(errors="replace").splitlines()
                keep = [ln for ln in tail if "Warning" not in ln and "warn(" not in ln][-8:]
                out.append(f"[rank {r}] " + " | ".join(keep))
            except OSError:
                pass
        return "\n".join(out)

    def restart(self):
        self.stop(hard=True)
        self.restarts += 1
        self.start()

    def run(self, prog):
        if not self.started:
            self.start()
        t0 = time.time()
        for c in self.conns:
            c.send(("run", prog))
        results = [None] * self.world
        pending = {self.conns[r]: r for r in range(self.world)}
        dl = t0 + self.hang_deadline
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
                dl = min(dl, time.time() + 2 * self.cfg["timeout"] + 2)
        out = {
            "kind": kind,
            "results": results,
            "wall": time.time() - t0,
            "stuck_ranks": sorted(pending.values()) if kind != "ok" else [],
        }
        if kind == "crash":
            out["crash_log"] = self._log_tails([r for r in range(self.world) if (results[r] or {}).get("died")])
            out["exitcodes"] = [p.exitcode for p in self.procs]
        if kind != "ok":
            self.restart()
        return out


def run_once(prog, world=4, timeout=TIMEOUT, detail=False, coverage=False, logdir=None):
    s = Session(world, timeout, detail, coverage, logdir)
    s.start()
    try:
        return s.run(prog)
    finally:
        s.stop(hard=False)
