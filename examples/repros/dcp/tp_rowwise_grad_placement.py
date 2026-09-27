# A6: RowwiseParallel Linear with in_features < TP degree: weight Shard(1), grad Replicate, optimizer step fails
# usage: python tp_rowwise_grad_placement.py WORLD IN_FEATURES   (e.g. 4 3)
import sys

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import Replicate
from torch.distributed.tensor.parallel import RowwiseParallel, parallelize_module


def run(rank, W, din):
    dist.init_process_group("gloo", init_method="tcp://127.0.0.1:29519", rank=rank, world_size=W)
    torch.manual_seed(0)
    m = torch.nn.Linear(din, 13, bias=False)
    parallelize_module(m, init_device_mesh("cpu", (W,)), RowwiseParallel(input_layouts=Replicate()))
    opt = torch.optim.SGD(m.parameters(), lr=0.1, foreach=False)
    m(torch.randn(6, 2, din)).sum().backward()
    msg = f"weight {m.weight.placements} grad {m.weight.grad.placements}"
    try:
        opt.step()
        msg += " step ok"
    except Exception as e:
        msg += f" step {type(e).__name__}: {str(e)[:80]}"
    if rank == 0:
        print(f"W={W} in_features={din}: {msg}", flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    W, din = int(sys.argv[1]), int(sys.argv[2])
    mp.spawn(run, args=(W, din), nprocs=W)
