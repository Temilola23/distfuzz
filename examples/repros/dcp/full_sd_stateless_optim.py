# A3: set_state_dict(full_state_dict=True) into an optimizer with no tensor state (SGD, momentum=0) raises
# usage: python full_sd_stateless_optim.py
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.checkpoint.state_dict import StateDictOptions, get_state_dict, set_state_dict
from torch.distributed.fsdp import fully_shard


def run(rank, W):
    dist.init_process_group("gloo", init_method="tcp://127.0.0.1:29515", rank=rank, world_size=W)
    m = torch.nn.Linear(4, 4)
    fully_shard(m)
    opt = torch.optim.SGD(m.parameters(), lr=0.1)  # momentum=0: no per-param state
    m(torch.randn(2, 4)).sum().backward()
    opt.step()
    opts = StateDictOptions(full_state_dict=True)
    msd, osd = get_state_dict(m, opt, options=opts)
    try:
        set_state_dict(m, opt, model_state_dict=msd, optim_state_dict=osd, options=opts)
        r = "ok"
    except Exception as e:
        r = f"{type(e).__name__}: {e}"
    print(f"rank{rank}: {r}", flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    mp.spawn(run, args=(2,), nprocs=2)
