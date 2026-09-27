# torchrun --nproc-per-node 4 gloo_strided_sweep.py
from datetime import timedelta

import torch
import torch.distributed as dist

dist.init_process_group("gloo", timeout=timedelta(seconds=15))
R, W = dist.get_rank(), dist.get_world_size()
N = 4


def strided(vals):
    buf = torch.full((2 * len(vals),), 7777.0)
    v = buf[::2]
    v.copy_(torch.as_tensor(vals, dtype=torch.float32))
    return v, buf


def base(r):
    return [float(10 * r + i) for i in range(N)]


def report(name, got, exp, buf):
    ok = torch.equal(got, torch.as_tensor(exp, dtype=torch.float32))
    gaps_ok = bool((buf[1::2] == 7777.0).all()) if buf is not None else True
    res = [None] * W
    dist.all_gather_object(res, (ok, gaps_ok, str(err[0]) if err[0] else None))
    if R == 0:
        bad = [i for i, (o, g, e) in enumerate(res) if not o]
        clob = [i for i, (o, g, e) in enumerate(res) if not g]
        errs = {e for _, _, e in res if e}
        verdict = (
            "RAISES " + str(errs)[:90]
            if errs
            else ("OK" if not bad and not clob else f"WRONG ranks {bad}; out-of-view writes ranks {clob}")
        )
        print(f"{name:22s} {verdict}", flush=True)


err = [None]


def run(fn):
    err[0] = None
    try:
        return fn()
    except Exception as e:  # noqa: BLE001
        err[0] = type(e).__name__ + ": " + str(e).splitlines()[0][:70]


v, b = strided(base(R))
run(lambda: dist.all_reduce(v))
report("all_reduce", v, [sum(base(r)[i] for r in range(W)) for i in range(N)], b)
v, b = strided(base(R))
run(lambda: dist.reduce(v, dst=0))
report("reduce", v, [sum(base(r)[i] for r in range(W)) for i in range(N)] if R == 0 else base(R), b)
v, b = strided(base(R))
run(lambda: dist.broadcast(v, src=0))
report("broadcast", v, base(0), b)
outs = [strided([0.0] * N) for _ in range(W)]
t = torch.tensor(base(R))
run(lambda: dist.all_gather([o for o, _ in outs], t))
ok_all = all(torch.equal(o, torch.tensor(base(r))) for r, (o, _) in enumerate(outs))
report("all_gather(out)", torch.tensor([1.0]) if ok_all else torch.tensor([0.0]), [1.0], outs[0][1])
v, b = strided(base(R))
outs2 = [torch.zeros(N) for _ in range(W)]
run(lambda: dist.all_gather(outs2, v))
ok_all = all(torch.equal(o, torch.tensor(base(r))) for r, o in enumerate(outs2))
report("all_gather(in)", torch.tensor([1.0]) if ok_all else torch.tensor([0.0]), [1.0], b)
v, b = strided([0.0] * N)
chunks = [torch.tensor(base(r)) for r in range(W)] if R == 0 else None
run(lambda: dist.scatter(v, chunks, src=0))
report("scatter(out)", v, base(R), b)
v, b = strided([0.0] * N)
ins = [torch.tensor(base(r)) * (R + 1) for r in range(W)]
run(lambda: dist.reduce_scatter(v, ins))
report("reduce_scatter(out)", v, [x * sum(k + 1 for k in range(W)) for x in base(R)], b)
v, b = strided([float(100 * R + j) for j in range(W)])
o = torch.zeros(W)
run(lambda: dist.all_to_all_single(o, v))
report("all_to_all_single(in)", o, [float(100 * j + R) for j in range(W)], b)
if R == 0:
    print("torch", torch.__version__)
dist.destroy_process_group()
