from __future__ import annotations

import multiprocessing as mp
import os
import socket
from datetime import timedelta


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def _rank_main(rank, world, port, conn, cases, timeout):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port), GLOO_SOCKET_IFNAME="lo")
    import warnings

    warnings.filterwarnings("ignore")
    import torch
    import torch.distributed as dist

    torch.set_num_threads(1)
    dist.init_process_group("gloo", rank=rank, world_size=world, timeout=timedelta(seconds=60))
    dist.barrier()
    out = {}
    for name, fn in cases:
        try:
            out[name] = ("ok", fn(rank, world))
        except Exception as e:  # noqa: BLE001
            out[name] = ("exc", f"{type(e).__name__}: {str(e).splitlines()[0][:200] if str(e) else ''}")
        try:
            dist.barrier()
        except Exception:  # noqa: BLE001
            pass
    conn.send(out)
    try:
        dist.destroy_process_group()
    except Exception:  # noqa: BLE001
        pass


def run_cases(cases, world=4, timeout=5.0, deadline=600):
    ctx = mp.get_context("spawn")
    port = free_port()
    procs, conns = [], []
    for r in range(world):
        a, b = ctx.Pipe()
        p = ctx.Process(target=_rank_main, args=(r, world, port, b, cases, timeout))
        p.start()
        procs.append(p)
        conns.append(a)
    outs = []
    for c in conns:
        if not c.poll(deadline):
            for q in procs:
                q.kill()
            raise RuntimeError("rank did not finish")
        outs.append(c.recv())
    for p in procs:
        p.join(10)
    res = {}
    for name, _ in cases:
        res[name] = [o.get(name, ("missing", None)) for o in outs]
    return res
