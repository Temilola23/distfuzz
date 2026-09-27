# B (known, #150336): Colwise->Rowwise MLP whose hidden dim is not divisible by the TP degree fails
# usage: python tp_uneven_mlp.py WORLD HIDDEN   (e.g. 3 5)
import sys

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor.parallel import ColwiseParallel, RowwiseParallel, parallelize_module


def run(rank, W, H):
    dist.init_process_group("gloo", init_method="tcp://127.0.0.1:29513", rank=rank, world_size=W)
    torch.manual_seed(0)
    ref = torch.nn.Sequential(torch.nn.Linear(8, H), torch.nn.ReLU(), torch.nn.Linear(H, 8)).double()
    m = torch.nn.Sequential(torch.nn.Linear(8, H), torch.nn.ReLU(), torch.nn.Linear(H, 8)).double()
    m.load_state_dict(ref.state_dict())
    parallelize_module(m, init_device_mesh("cpu", (W,)), {"0": ColwiseParallel(), "2": RowwiseParallel()})
    x = torch.randn(6, 2, 8, dtype=torch.double, generator=torch.Generator().manual_seed(1))
    try:
        out = m(x)
        out.sum().backward()
        r = ref(x)
        r.sum().backward()
        g = m[0].weight.grad.full_tensor()
        gdiff = (g - ref[0].weight.grad).abs().max()
        print(f"rank{rank} H={H}: out maxdiff={(out - r).abs().max():.2e} grad maxdiff={gdiff:.2e}", flush=True)
    except Exception as e:
        print(f"rank{rank} H={H}: {type(e).__name__}: {str(e)[:90]}", flush=True)
        raise


if __name__ == "__main__":
    mp.spawn(run, args=(int(sys.argv[1]), int(sys.argv[2])), nprocs=int(sys.argv[1]))
