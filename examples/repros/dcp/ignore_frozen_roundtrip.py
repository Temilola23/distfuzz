# C (usage): get_state_dict(ignore_frozen_params=True) -> set_state_dict needs strict=False
# usage: python ignore_frozen_roundtrip.py
import torch
import torch.distributed as dist
from torch.distributed.checkpoint.state_dict import StateDictOptions, get_state_dict, set_state_dict

dist.init_process_group("gloo", init_method="tcp://127.0.0.1:29516", rank=0, world_size=1)
m = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.Linear(4, 4))
m[0].requires_grad_(False)
opt = torch.optim.Adam([p for p in m.parameters() if p.requires_grad])
opts = StateDictOptions(ignore_frozen_params=True)
msd, osd = get_state_dict(m, opt, options=opts)
print(sorted(msd))
try:
    set_state_dict(m, opt, model_state_dict=msd, optim_state_dict=osd, options=opts)
    print("ok")
except Exception as e:
    print(type(e).__name__, str(e).splitlines()[:2])
