# usage: python cases.py <case> [world=4]  (case = any function below except show/partial_from/main)
import datetime
import socket
import sys

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor, Partial, Replicate, Shard, distribute_tensor


def show(rank, name, ref, dt):
    full = dt.full_tensor() if isinstance(dt, DTensor) else dt
    if rank == 0:
        same = (
            ref.shape == full.shape
            and ref.dtype == full.dtype
            and torch.allclose(ref.double(), full.double(), equal_nan=True, atol=1e-5)
        )
        print(f"{name}: {'OK' if same else 'MISMATCH'}")
        print(f"   ref  {tuple(ref.shape)} {ref.dtype}: {ref.flatten()[:10].tolist()}")
        print(
            f"   dt   {tuple(full.shape)} {full.dtype}: {full.flatten()[:10].tolist()}"
            f"  placements={getattr(dt, 'placements', None)}"
        )


def partial_from(rank, mesh, full, op="sum"):
    n = mesh.size()
    if op == "sum":
        local = full / n if full.is_floating_point() else (full if rank == 0 else torch.zeros_like(full))
    else:
        local = full.clone()
    return DTensor.from_local(local, mesh, [Partial(op)], run_check=False)


def mse_loss_uneven(rank, mesh):
    """A3: mean-reduced losses on a tensor sharded unevenly (3 rows over 2 ranks; use world 2)."""
    torch.manual_seed(0)
    a, b = torch.randn(3, 4), torch.randn(3, 4)
    da = distribute_tensor(a, mesh, [Shard(0)])
    db = distribute_tensor(b, mesh, [Shard(0)])
    show(rank, "mse_loss", F.mse_loss(a, b), F.mse_loss(da, db))
    show(rank, "l1_loss", F.l1_loss(a, b), F.l1_loss(da, db))
    show(rank, "smooth_l1_loss", F.smooth_l1_loss(a, b), F.smooth_l1_loss(da, db))
    show(rank, "huber_loss", F.huber_loss(a, b), F.huber_loss(da, db))
    show(rank, "mean (control)", a.mean(), da.mean())


def cast_partial(rank, mesh):
    """Fixed in 2.14 (#172684): casting a Partial(sum) DTensor to int/bool kept the Partial."""
    full = torch.tensor([0.0, 2.0, -3.0, 1.0])
    p = partial_from(rank, mesh, full, "sum")
    show(rank, "partial_sum.bool()", full.bool(), p.bool())
    show(rank, "partial_sum.to(int64)", full.to(torch.int64), p.to(torch.int64))
    full2 = torch.tensor([0.5, 1.5, -2.5, 3.0])
    p2 = partial_from(rank, mesh, full2, "sum")
    show(rank, "partial_sum(0.5,..).long()", full2.long(), p2.long())


def fill_partial(rank, mesh):
    """A2: fill_ on a Partial(sum) DTensor keeps the Partial (add_ correctly refuses)."""
    full = torch.arange(4.0)
    p = partial_from(rank, mesh, full, "sum")
    p.fill_(3.0)
    show(rank, "partial_sum.fill_(3)", torch.full((4,), 3.0), p)
    q = partial_from(rank, mesh, full, "sum")
    q.add_(1.0)
    show(rank, "partial_sum.add_(1)", full + 1, q)


def setitem_shard(rank, mesh):
    """A1: x[i] = v on a DTensor sharded along dim 0 is silently dropped."""
    full = torch.arange(9.0)
    x = distribute_tensor(full.clone(), mesh, [Shard(0)])
    x[6] = 100.0
    ref = full.clone()
    ref[6] = 100.0
    show(rank, "sharded[6] = 100", ref, x)


def max_empty_shard(rank, mesh):
    """A6: full max() when some ranks hold an empty shard (2 elements over 4 ranks): ranks diverge."""
    full = torch.tensor([1.0, 5.0])
    x = distribute_tensor(full, mesh, [Shard(0)])
    try:
        out = x.max()
    except Exception as e:
        print(f"[rank {rank}] raised {type(e).__name__}: {str(e)[:120]}", flush=True)
        raise
    show(rank, "max()", full.max(), out)


def any_normpartial(rank, mesh):
    """Fixed in 2.14: any() on the _NormPartial output of norm() returned float32."""
    full = torch.randn(8)
    x = distribute_tensor(full, mesh, [Shard(0)])
    n = x.norm()
    r = n.any(dim=-1)
    if rank == 0:
        print("norm placements", n.placements, "any() dtype:", r.dtype, "expected", full.norm().any(dim=-1).dtype)
    show(rank, "norm().any(dim=-1)", full.norm().any(dim=-1), r)


def embedding_full_tensor_twice(rank, mesh):
    """A7: an embedding output (_MaskPartial) can be reduced only once."""
    w = torch.randn(8, 3)
    idx = torch.tensor([1, 6, 3])
    dw = distribute_tensor(w, mesh, [Shard(0)])
    didx = distribute_tensor(idx, mesh, [Replicate()])
    out = F.embedding(didx, dw)
    if rank == 0:
        print("placements", out.placements)
    show(rank, "embedding 1st full_tensor", F.embedding(idx, w), out)
    show(rank, "embedding 2nd full_tensor", F.embedding(idx, w), out)


def getitem_partial_max(rank, mesh):
    """Fixed in 2.14: advanced indexing a Partial(max) tensor with a sharded index."""
    full = torch.arange(6.0).reshape(1, 6)
    x = DTensor.from_local(full - (rank % 2), mesh, [Partial("max")], run_check=False)
    idx = torch.tensor([5, 0, 3, 1, 2, 4])
    didx = distribute_tensor(idx, mesh, [Shard(0)])
    show(rank, "partial_max[:, idx]", full[:, idx], x[:, didx])


def any_then_cast(rank, mesh):
    """Fixed in 2.14: any() yields a bool Partial(sum); casting it to float kept Partial(sum)."""
    full = torch.tensor([0.0, 1.0, 0.0, 2.0, 0.0, 3.0, 0.0, 4.0])
    x = distribute_tensor(full, mesh, [Shard(0)])
    a = x.any()
    if rank == 0:
        print("any() placements:", a.placements)
    show(rank, "x.any().float()", full.any().float(), a.float())
    show(rank, "x.any().long()", full.any().long(), a.long())


def cumsum_uneven(rank, mesh):
    """Fixed in 2.14: cumsum along a sharded dim with uneven/empty shards."""
    for dt in (torch.float32, torch.bfloat16):
        full = torch.tensor([-4.0, 3.0, 3.0], dtype=dt)
        x = distribute_tensor(full, mesh, [Shard(0)])
        show(rank, f"cumsum {dt}", full.cumsum(0), x.cumsum(0))


def flatten_zero_size(rank, mesh):
    """A4: flatten() of a zero-size DTensor loops forever in view_groups (20 s watchdog)."""
    import faulthandler

    faulthandler.dump_traceback_later(20, exit=True)
    full = torch.zeros(0, 7, 4, 4)
    x = distribute_tensor(full, mesh, [Replicate()])
    show(rank, "flatten zero-size", full.flatten(), x.flatten())


def loss_uneven_mixed(rank, mesh):
    """A3: mean-reduced losses, one operand sharded unevenly (9 over 4 ranks -> 3,3,3,0)."""
    torch.manual_seed(0)
    a, b = torch.randn(1, 9, 9), torch.randn(1, 9, 9)
    for pa, pb in (([Replicate()], [Shard(2)]), ([Shard(2)], [Shard(2)]), ([Shard(1)], [Replicate()])):
        da, db = distribute_tensor(a, mesh, pa), distribute_tensor(b, mesh, pb)
        for fn in (F.mse_loss, F.l1_loss, F.smooth_l1_loss, F.huber_loss):
            show(rank, f"{fn.__name__} {pa}x{pb}", fn(a, b), fn(da, db))
        show(rank, f"control ((a-b)**2).mean() {pa}x{pb}", ((a - b) ** 2).mean(), ((da - db) ** 2).mean())


def inf_norm_partial_max(rank, mesh):
    """A5: inf-norm of a Partial(max) tensor; |.| does not commute with max."""
    full = torch.tensor([3.0, -1.0, 2.0])
    local = full if rank == 0 else full - 5.0  # max over ranks == full
    x = DTensor.from_local(local, mesh, [Partial("max")], run_check=False)
    show(
        rank,
        "vector_norm(inf)",
        torch.linalg.vector_norm(full, ord=float("inf")),
        torch.linalg.vector_norm(x, ord=float("inf")),
    )
    show(rank, "abs()", full.abs(), x.abs())


def split_backward(rank, mesh):
    """A8: backward through split()[k] mixes torch.Tensor and DTensor in aten.cat."""
    full = torch.arange(6.0)
    x = distribute_tensor(full, mesh, [Replicate()]).requires_grad_()
    try:
        x.split(2, 0)[2].sum().backward()
        show(rank, "grad of split(2)[2].sum()", torch.tensor([0.0, 0, 0, 0, 1, 1]), x.grad)
    except Exception as e:
        if rank == 0:
            print(f"split backward raised {type(e).__name__}: {str(e)[:120]}")


def squeeze_zero_size(rank, mesh):
    """A9: squeeze() of a zero-size DTensor with (Replicate, Shard(3)) on a 2x2 mesh moves the shard dim."""
    mesh2 = init_device_mesh("cpu", (2, 2))
    full = torch.zeros(0, 8, 9, 3)
    x = distribute_tensor(full, mesh2, [Replicate(), Shard(3)])
    out = x.squeeze()
    if rank == 0:
        print(f"squeeze(): global {tuple(out.shape)}, expected {tuple(full.squeeze().shape)}")
    show(rank, "squeeze zero-size", full.squeeze(), out)


def main(rank, world, port, name):
    dist.init_process_group(
        "gloo",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=world,
        timeout=datetime.timedelta(seconds=15),
    )
    mesh = init_device_mesh("cpu", (world,))
    globals()[name](rank, mesh)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    if len(sys.argv) < 2 or not callable(globals().get(sys.argv[1])) or sys.argv[1] in ("show", "partial_from", "main"):
        sys.exit("usage: python cases.py <case> [world=4]")
    name = sys.argv[1]
    world = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    mp.spawn(main, args=(world, port, name), nprocs=world, join=True)
