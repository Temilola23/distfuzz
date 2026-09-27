# python size_mismatch.py  (Docker; control: a fresh 2-rank size mismatch raises cleanly)
import multiprocessing as mp
import os
from datetime import timedelta


def worker(rank):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT="29888", GLOO_SOCKET_IFNAME="lo")
    import torch
    import torch.distributed as dist

    dist.init_process_group("gloo", rank=rank, world_size=2, timeout=timedelta(seconds=10))
    # rank 0 contributes 1 element, rank 1 contributes 4: a data-size mismatch
    t = torch.ones(1 if rank == 0 else 4)
    try:
        dist.all_reduce(t)  # user wraps the collective defensively
        print(f"rank{rank}: all_reduce returned (unexpected)", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"rank{rank}: caught {type(e).__name__} (good, recoverable)", flush=True)
    dist.destroy_process_group()


def main():
    ctx = mp.get_context("spawn")
    ps = [ctx.Process(target=worker, args=(r,)) for r in range(2)]
    for p in ps:
        p.start()
    for p in ps:
        p.join(timeout=30)
    codes = [p.exitcode for p in ps]
    for p in ps:
        if p.is_alive():
            p.kill()
    aborted = [c for c in codes if c in (-6, -11, -7, -4)]
    print("EXITCODES", codes, "ABORTED" if aborted else "clean", flush=True)
    return 1 if aborted else 0


if __name__ == "__main__":
    raise SystemExit(main())
