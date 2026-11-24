# §30: clip_grad_norm_(norm_type=inf) on FSDP2 fails on the rank whose dim-0 shard is empty; the other ranks wait
# usage: python clip_inf_empty_shard.py   (4 Gloo ranks; run inside the Docker image)
import os
from datetime import timedelta

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.fsdp import fully_shard


def run(rank, world, out_features, norm_type, foreach, port):
    # 10 s timeout so the hang shows up as an error instead of the 30 min default
    dist.init_process_group(
        "gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=world, timeout=timedelta(seconds=10)
    )
    torch.manual_seed(0)
    model = torch.nn.Sequential(torch.nn.Linear(8, out_features), torch.nn.Linear(out_features, 8))
    ref = torch.nn.Sequential(torch.nn.Linear(8, out_features), torch.nn.Linear(out_features, 8))
    ref.load_state_dict(model.state_dict())
    for layer in model:
        fully_shard(layer)
    fully_shard(model)
    x = torch.randn(4, 8)
    model(x).sum().backward()
    ref(x).sum().backward()
    want = torch.nn.utils.clip_grad_norm_(ref.parameters(), 1.0, norm_type=norm_type).item()
    rows = [p.grad.to_local().shape[0] for p in model.parameters()]
    try:
        got = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, norm_type=norm_type, foreach=foreach)
        res = f"norm {got.full_tensor().item():.4f}, single process {want:.4f}"
    except Exception as e:
        res = f"{type(e).__name__}: {str(e).splitlines()[0].split('] ', 1)[-1][:75]}"
    # one write per line so the ranks' lines do not interleave
    os.write(
        1,
        f"out_features {out_features}, norm_type {norm_type}, foreach {foreach}, rank {rank} "
        f"(local dim-0 rows {rows}): {res}\n".encode(),
    )


if __name__ == "__main__":
    cases = [(8, float("inf"), True), (5, 2.0, True), (5, float("inf"), True), (5, float("inf"), False)]
    for i, (n, p, fe) in enumerate(cases):
        mp.spawn(run, args=(4, n, p, fe, 29650 + i), nprocs=4, join=False).join()
