# python inflight_resize_layouts.py [contig|transposed|strided] [grow|set_|shrink] [iters=200]  (Docker only; world 2)
import multiprocessing as mp
import os
import random
import sys
from datetime import timedelta

LAYOUT = sys.argv[1] if len(sys.argv) > 1 else "contig"  # contig | transposed | strided
HOW = sys.argv[2] if len(sys.argv) > 2 else "grow"  # grow | set_ | shrink
ITERS = int(sys.argv[3]) if len(sys.argv) > 3 else 200


def make(torch):
    if LAYOUT == "contig":
        return torch.ones(18, dtype=torch.int32)
    if LAYOUT == "transposed":
        return torch.ones(3, 6, dtype=torch.int32).t()
    return torch.ones(36, dtype=torch.int32)[::2]


def worker(rank, port):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port), GLOO_SOCKET_IFNAME="lo")
    import torch
    import torch.distributed as dist

    torch.set_num_threads(1)
    dist.init_process_group("gloo", rank=rank, world_size=2, timeout=timedelta(seconds=5))
    errs = set()
    for _ in range(ITERS):
        t = make(torch)
        try:
            w = dist.all_reduce(t, async_op=True)
            if HOW == "grow":
                t.resize_(t.numel() * 4 + 8)
            elif HOW == "set_":
                t.set_(torch.empty(t.numel() + 4, dtype=t.dtype))
            else:
                t.resize_(0)
            w.wait()
        except Exception as e:
            errs.add(type(e).__name__ + ": " + str(e).splitlines()[0][:80])
    print(f"rank{rank} errs={sorted(errs)[:2]}", flush=True)
    os._exit(0)


if __name__ == "__main__":
    port = 29000 + random.randrange(2000)
    ctx = mp.get_context("spawn")
    ps = [ctx.Process(target=worker, args=(r, port)) for r in range(2)]
    for p in ps:
        p.start()
    for p in ps:
        p.join(timeout=120)
    for p in ps:
        if p.is_alive():
            p.kill()
    codes = [p.exitcode for p in ps]
    print("RESULT", LAYOUT, HOW, codes, "CRASH" if any(c in (-6, -11, -7, -4) for c in codes) else "clean", flush=True)
