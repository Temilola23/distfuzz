# A2: set_state_dict(full_state_dict=True, flatten_optimizer_state_dict=True) rejects get_state_dict's own output
# usage: python flat_osd_full_load.py WORLD FSDP   (e.g. 2 1)
import sys

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.checkpoint.state_dict import StateDictOptions, get_state_dict, set_state_dict
from torch.distributed.fsdp import fully_shard


def run(rank, W, wrap):
    dist.init_process_group("gloo", init_method="tcp://127.0.0.1:29518", rank=rank, world_size=W)
    torch.manual_seed(0)
    m = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.Linear(4, 2))
    if wrap:
        fully_shard(m)
    opt = torch.optim.Adam(m.parameters(), lr=0.1)
    m(torch.randn(2, 4)).sum().backward()
    opt.step()
    for full in (False, True):
        o = StateDictOptions(full_state_dict=full, flatten_optimizer_state_dict=True)
        msd, osd = get_state_dict(m, opt, options=o)
        try:
            set_state_dict(m, opt, model_state_dict=msd, optim_state_dict=osd, options=o)
            r = "ok"
        except Exception as e:
            r = f"{type(e).__name__}: {e}"
        if rank == 0:
            print(f"W={W} fsdp={wrap} full_state_dict={full}: {r}", flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    W, wrap = int(sys.argv[1]), int(sys.argv[2])
    mp.spawn(run, args=(W, wrap), nprocs=W)
