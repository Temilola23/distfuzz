import math

import torch
import torch.nn.functional as F
from torch.distributed.tensor import DTensor, Partial, Replicate, Shard, distribute_tensor

DTYPES = {
    "f32": torch.float32,
    "f64": torch.float64,
    "bf16": torch.bfloat16,
    "f16": torch.float16,
    "i64": torch.int64,
    "i32": torch.int32,
    "bool": torch.bool,
}


def parse_pl(s):
    if s == "R":
        return Replicate()
    if s.startswith("S"):
        return Shard(int(s[1:]))
    if s.startswith("P:"):
        return Partial(s[2:])
    raise ValueError(s)


def gen_full(spec):
    # integer-valued so most float ops are exact regardless of reduction order
    g = torch.Generator().manual_seed(spec["seed"])
    shape = tuple(spec["shape"])
    dt = DTYPES[spec["dtype"]]
    if spec.get("kind") == "index":
        hi = max(1, spec["high"])
        return torch.randint(0, hi, shape, generator=g, dtype=torch.int64)
    if dt == torch.bool:
        return torch.randint(0, 2, shape, generator=g).bool()
    return torch.randint(-4, 5, shape, generator=g).to(dt)


def partial_pieces(T, n, op, seed):
    g = torch.Generator().manual_seed(seed)
    if n == 1:
        return [T.clone()]
    if T.dtype == torch.bool:
        owner = torch.randint(0, n, T.shape, generator=g)
        if op in ("sum", "max"):
            return [torch.where(owner == i, T, torch.zeros_like(T)) for i in range(n)]
        if op == "min":
            return [torch.where(owner == i, T, torch.ones_like(T)) for i in range(n)]
        raise ValueError(op)
    if op in ("sum", "avg"):
        tgt = T * n if op == "avg" else T
        pieces = [torch.randint(-3, 4, T.shape, generator=g).to(T.dtype) for _ in range(n - 1)]
        last = tgt.clone()
        for p in pieces:
            last = last - p
        return pieces + [last]
    if op in ("max", "min"):
        owner = torch.randint(0, n, T.shape, generator=g)
        pieces = []
        for i in range(n):
            d = torch.randint(0, 3, T.shape, generator=g).to(T.dtype)
            p = T - d if op == "max" else T + d
            pieces.append(torch.where(owner == i, T, p))
        return pieces
    raise ValueError(op)


class Ctx:
    def __init__(self, mode, mesh=None, inputs=None):
        self.mode = mode
        self.mesh = mesh
        self.inputs = inputs or []
        self.made = []  # (input_idx, tensor) for grad checks
        self._compiled = {}


def make_input(ctx, i):
    spec = ctx.inputs[i]
    T = gen_full(spec)
    rg = bool(spec.get("rg"))
    if ctx.mode == "ref":
        t = T.clone()
        if rg:
            t.requires_grad_()
        ctx.made.append((i, t))
        return t
    mesh = ctx.mesh
    pls = [parse_pl(p) for p in spec["pl"]]
    pdims = [d for d, p in enumerate(pls) if p.is_partial()]
    local_full = T
    if pdims:
        coord = mesh.get_coordinate()
        n, idx = 1, 0
        for d in pdims:
            idx = idx * mesh.size(d) + coord[d]
            n *= mesh.size(d)
        local_full = partial_pieces(T, n, pls[pdims[0]].reduce_op, spec["seed"] + 7919)[idx]
    repl = [Replicate() if p.is_partial() else p for p in pls]
    tmp = distribute_tensor(local_full, mesh, repl, src_data_rank=None)
    dt = DTensor.from_local(tmp.to_local(), mesh, pls, run_check=False, shape=T.shape, stride=T.stride())
    if rg:
        dt = dt.detach().requires_grad_()
    ctx.made.append((i, dt))
    return dt


def make_env(ctx):
    def mk(i):
        return make_input(ctx, i)

    def R(x, pls):
        # always a fresh tensor, so in-place ops after a no-op redistribute do not alias
        if ctx.mode == "ref":
            return x.clone()
        y = x.redistribute(x.device_mesh, [parse_pl(p) for p in pls])
        if y is x or (
            y.to_local().numel()
            and y.to_local().untyped_storage().data_ptr() == x.to_local().untyped_storage().data_ptr()
        ):
            y = y.clone()
        return y

    def BW(x):
        x.sum().backward()
        return None

    def SETITEM(x, idx, y):
        x[idx] = y
        return x

    def CMP(src, *args):
        fn = eval(src, env)
        if ctx.mode == "ref":
            return fn(*args)
        if src not in ctx._compiled:
            ctx._compiled[src] = torch.compile(fn, backend="aot_eager", dynamic=False)
        return ctx._compiled[src](*args)

    env = {"torch": torch, "F": F, "math": math, "mk": mk, "R": R, "BW": BW, "SETITEM": SETITEM, "CMP": CMP}
    return env


def tol_for(dtype, scale=1.0):
    # atol scales with the largest reference magnitude: sharded reductions round each rank's
    # partial result, so an output that cancels to a small value inherits the terms' error
    if dtype in (torch.bfloat16, torch.float16):
        return 2e-2, 2e-2 * max(1.0, scale)
    if dtype.is_floating_point:
        return 1e-4, 1e-4 * max(1.0, scale)
    return 0.0, 0.0


def compare(ref, got):
    if tuple(ref.shape) != tuple(got.shape):
        return f"shape ref={tuple(ref.shape)} got={tuple(got.shape)}"
    if ref.dtype != got.dtype:
        return f"dtype ref={ref.dtype} got={got.dtype}"
    r, g = ref.detach(), got.detach()
    if r.numel() == 0:
        return None
    scale = float(r.double().nan_to_num(0, 0, 0).abs().max())
    rtol, atol = tol_for(r.dtype, scale)
    if r.dtype.is_floating_point or r.dtype.is_complex:
        ok = torch.isclose(g.double(), r.double(), rtol=rtol, atol=atol, equal_nan=True)
        if bool(ok.all()):
            return None
        bad = (~ok).nonzero()[:3].tolist()
        diff = (g.double() - r.double()).abs()
        if bool(diff.isnan().any()):
            return f"values differ at {bad} (nan/inf vs finite)"
        return f"values differ at {bad} maxdiff={diff.max().item():.4g}"
    if torch.equal(r, g):
        return None
    bad = (r != g).nonzero()[:3].tolist()
    return f"values differ at {bad}"
