# torchrun --nproc-per-node 4 gloo_strided_allreduce.py
import torch
import torch.distributed as dist

dist.init_process_group("gloo")
rank, world = dist.get_rank(), dist.get_world_size()
expected = sum(torch.arange(6, dtype=torch.int64).reshape(3, 2) * (r + 1) for r in range(world))


def make(r, noncontig):
    t = torch.arange(6, dtype=torch.int64).reshape(3, 2) * (r + 1)
    if noncontig:
        buf = torch.zeros(3, 4, dtype=torch.int64)
        v = buf[:, ::2]  # [3,2] view, strides (4,2)
        v.copy_(t)
        return v
    return t.clone()


def case(name, noncontig_ranks, async_op, other_before_wait):
    t = make(rank, rank in noncontig_ranks)
    w = dist.all_reduce(t, async_op=async_op)
    if other_before_wait:
        dist.all_reduce(torch.ones(1))
    if async_op:
        w.wait()
    ok = torch.equal(t, expected)
    oks = [None] * world
    dist.all_gather_object(oks, ok)
    if rank == 0:
        print(f"{name:55s} {'OK' if all(oks) else 'WRONG on ranks ' + str([i for i, o in enumerate(oks) if not o])}")


for nc in ([], [0], [0, 1, 2, 3]):
    for async_op in (False, True):
        for other in (False, True) if async_op else (False,):
            case(f"noncontig={nc} async={async_op} other_before_wait={other}", nc, async_op, other)
dist.destroy_process_group()
