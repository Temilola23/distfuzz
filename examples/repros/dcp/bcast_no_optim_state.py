# A4: set_state_dict(full_state_dict=True, broadcast_from_rank0=True) with empty non-zero-rank dicts desyncs ranks
# usage: TORCH_DISTRIBUTED_DEBUG=DETAIL python bcast_no_optim_state.py WORLD MOMENTUM FSDP  (SPLIT=1: control)
import os
import sys

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_state_dict,
    set_model_state_dict,
    set_optimizer_state_dict,
    set_state_dict,
)


def run(rank, W, momentum, fsdp):
    dist.init_process_group("gloo", init_method="tcp://127.0.0.1:29514", rank=rank, world_size=W)
    m = torch.nn.Linear(4, 4)
    if fsdp:
        from torch.distributed.fsdp import fully_shard

        fully_shard(m)
    opt = torch.optim.SGD(m.parameters(), lr=0.1, momentum=momentum)
    msd, osd = get_state_dict(m, opt, options=StateDictOptions(full_state_dict=True, cpu_offload=True))
    try:
        o = StateDictOptions(full_state_dict=True, broadcast_from_rank0=True)
        if os.environ.get("SPLIT"):  # same data through the two single-purpose APIs
            set_model_state_dict(m, msd, options=o)
            set_optimizer_state_dict(m, opt, osd, options=o)
        else:
            set_state_dict(m, opt, model_state_dict=msd, optim_state_dict=osd, options=o)
        r = "ok"
    except Exception as e:
        r = f"{type(e).__name__}: {e}"
    print(f"rank{rank} fsdp={fsdp} momentum={momentum}: {r}", flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    mp.spawn(run, args=(int(sys.argv[1]), float(sys.argv[2]), int(sys.argv[3])), nprocs=int(sys.argv[1]))
