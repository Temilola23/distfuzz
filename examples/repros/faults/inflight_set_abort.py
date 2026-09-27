# python inflight_set_abort.py [world=4] [iters=200] [set_|grow|shrink]  (Docker; REPRODUCED = SIGABRT/SIGSEGV)
import multiprocessing as mp
import os
import sys
from datetime import timedelta

WORLD = int(sys.argv[1]) if len(sys.argv) > 1 else 4
ITERS = int(sys.argv[2]) if len(sys.argv) > 2 else 200
HOW = sys.argv[3] if len(sys.argv) > 3 else "set_"


def worker(rank, port):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port), GLOO_SOCKET_IFNAME="lo")
    import torch
    import torch.distributed as dist

    torch.set_num_threads(1)
    dist.init_process_group("gloo", rank=rank, world_size=WORLD, timeout=timedelta(seconds=5))
    sub = dist.new_group([0, 2] if WORLD > 2 else [0, 1], timeout=timedelta(seconds=2))
    members = [0, 2] if WORLD > 2 else [0, 1]
    errs = 0
    for _ in range(ITERS):
        if rank not in members:
            continue
        t = torch.ones(3, 6, dtype=torch.int32).t()  # non-contiguous [6,3] view
        try:
            w = dist.all_reduce(t, async_op=True, group=sub)
            if HOW == "set_":
                t.set_(torch.empty(t.numel() + 4, dtype=t.dtype))  # swap storage in flight
            elif HOW == "grow":
                t.resize_(t.numel() * 4 + 8)
            else:
                t.resize_(0)
            w.wait()
        except Exception:  # noqa: BLE001 -- a Python error is the acceptable outcome
            errs += 1
    print(f"rank{rank} finished, python errors={errs}", flush=True)
    os._exit(0)


def main():
    import random

    port = 29000 + random.randrange(2000)
    ctx = mp.get_context("spawn")
    ps = [ctx.Process(target=worker, args=(r, port)) for r in range(WORLD)]
    for p in ps:
        p.start()
    for p in ps:
        p.join(timeout=120)
    hung = [i for i, p in enumerate(ps) if p.is_alive()]
    for p in ps:
        if p.is_alive():
            p.kill()
    codes = [p.exitcode for p in ps]
    bad = [c for c in codes if c in (-6, -11, -7, -4)]
    print("EXITCODES", codes, "HUNG", hung, "REPRODUCED" if bad else "clean", flush=True)


if __name__ == "__main__":
    main()
