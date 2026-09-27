# python inflight_free.py [world=4] [iters=300]  (Docker; rarely crashes alone, the fuzzer needed prior state)
import multiprocessing as mp
import os
import random
import sys
from datetime import timedelta

WORLD = int(sys.argv[1]) if len(sys.argv) > 1 else 4
ITERS = int(sys.argv[2]) if len(sys.argv) > 2 else 300


def worker(rank, port):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port), GLOO_SOCKET_IFNAME="lo")
    import torch
    import torch.distributed as dist

    torch.set_num_threads(1)
    dist.init_process_group("gloo", rank=rank, world_size=WORLD, timeout=timedelta(seconds=5))
    for _ in range(ITERS):
        t = torch.ones(2, 3)
        try:
            w = dist.all_reduce(t, async_op=True)
            t.set_(torch.empty(0))  # drop the storage the op is reducing
            del t
            w.wait()
        except Exception:
            pass
    print(f"rank{rank} done", flush=True)
    os._exit(0)


def main():
    port = 29000 + random.randrange(2000)
    ctx = mp.get_context("spawn")
    ps = [ctx.Process(target=worker, args=(r, port)) for r in range(WORLD)]
    for p in ps:
        p.start()
    for p in ps:
        p.join(timeout=120)
    for p in ps:
        if p.is_alive():
            p.kill()
    codes = [p.exitcode for p in ps]
    bad = [c for c in codes if c in (-6, -11, -7, -4)]
    print("EXITCODES", codes, "REPRODUCED" if bad else "clean", flush=True)


if __name__ == "__main__":
    main()
