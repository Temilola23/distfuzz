# B1 (known, #164929): get_state_dict before the first optimizer.step() runs a phantom Adam step
# usage: python get_state_dict_phantom_step.py
import torch
import torch.distributed as dist
from torch.distributed.checkpoint.state_dict import get_state_dict

dist.init_process_group("gloo", init_method="tcp://127.0.0.1:29517", rank=0, world_size=1)


def train(snapshot_first):
    torch.manual_seed(0)
    m = torch.nn.Linear(4, 1).double()
    opt = torch.optim.Adam(m.parameters(), lr=0.1)
    if snapshot_first:
        get_state_dict(m, opt)  # e.g. saving an "initial" checkpoint at step 0
    x = torch.randn(8, 4, dtype=torch.double, generator=torch.Generator().manual_seed(1))
    for _ in range(3):
        m(x).pow(2).mean().backward()
        opt.step()
        opt.zero_grad()
    return m.weight.detach().clone(), opt.state[m.weight]["step"].item()


(w0, s0), (w1, s1) = train(False), train(True)
print("adam step counter:", s0, "vs", s1)
print("weight maxdiff after 3 steps:", (w0 - w1).abs().max().item())
