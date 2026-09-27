# A5: SGD(fused=True) on FSDP2 params has no DTensor sharding strategy; fused Adam/AdamW work
# usage: python fused_sgd_dtensor.py
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.fsdp import fully_shard


def run(rank, W):
    dist.init_process_group("gloo", init_method="tcp://127.0.0.1:29511", rank=rank, world_size=W)
    for name, mk in (
        ("sgd", lambda p: torch.optim.SGD(p, lr=0.1, momentum=0.9, fused=True)),
        ("adamw", lambda p: torch.optim.AdamW(p, lr=0.1, fused=True)),
    ):
        m = torch.nn.Linear(4, 4)
        fully_shard(m)
        opt = mk(m.parameters())
        m(torch.randn(2, 4)).sum().backward()
        try:
            opt.step()
            r = "ok"
        except Exception as e:
            r = f"{type(e).__name__}: {str(e)[:100]}"
        if rank == 0:
            print(f"fused {name}: {r}", flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    mp.spawn(run, args=(2,), nprocs=2)
