import copy
import math
import re
import warnings

import torch

from distfuzz.dtensor.runtime import Ctx, make_env

warnings.filterwarnings("ignore")

MAX_NUMEL = 4096
MESH_SHAPES = {4: {"1d": (4,), "2d": (2, 2)}, 3: {"1d": (3,)}, 2: {"1d": (2,)}, 6: {"1d": (6,), "2d": (2, 3)}}
NAME_RE = re.compile(r"\bv\d+\b")
MUT_RE = re.compile(r"_\(|SETITEM|BW\(")
MESH_NDIM = {"1d": 1, "2d": 2}

LAMBDAS = [
    "lambda a: torch.relu(a) + 1",
    "lambda a: a.sum(dim=0)",
    "lambda a: (a * a).mean()",
    "lambda a: a.transpose(0, -1).contiguous()",
    "lambda a: torch.softmax(a, -1)",
    "lambda a: a.reshape(-1)",
    "lambda a: a[1:] * 2",
    "lambda a: a.amax(-1, keepdim=True) - a",
    "lambda a: torch.cumsum(a, 0)",
    "lambda a: (a @ a.transpose(-1, -2))",
    "lambda a: a.view(-1, a.shape[-1]).sum(0)",
    "lambda a: torch.cat([a, a], dim=0)",
]


def rand_placements(rng, mesh, ndim, dtype):
    opts = ["R"] * 3 + [f"S{d}" for d in range(ndim) for _ in range(2)]
    pops = ["sum", "max", "min"] + (["avg"] if dtype in ("f32", "f64", "bf16", "f16") else [])
    out, pop = [], None
    for _ in range(MESH_NDIM[mesh]):
        if dtype != "bool" and rng.random() < 0.15:
            pop = pop or rng.choice(pops)
            out.append("P:" + pop)
        else:
            out.append(rng.choice(opts))
    return out


def parse_R(expr):
    m = re.match(r"^R\((v\d+), \[(.*)\]\)$", expr)
    return m and (m.group(1), [x.strip().strip("'") for x in m.group(2).split(",")])


def rand_shape(rng, ndim=None):
    if ndim is None:
        ndim = rng.choices([0, 1, 2, 3, 4], [1, 3, 5, 4, 1])[0]
    sizes = [0] * 1 + [1] * 3 + [2, 3, 4, 5, 6, 7, 8, 9, 12] * 2
    while True:
        shp = [rng.choice(sizes) for _ in range(ndim)]
        if math.prod(shp) <= 1024:
            return shp


def rand_dtype(rng):
    return rng.choices(["f32", "f64", "bf16", "f16", "i64", "i32", "bool"], [10, 3, 2, 1, 3, 1, 2])[0]


class Gen:
    def __init__(self, world, rng, allow_compile=True):
        self.world, self.rng = world, rng
        self.meshes = list(MESH_SHAPES[world].keys())
        self.allow_compile = allow_compile

    def replay(self, prog, upto=None):
        ctx = Ctx("ref", None, prog["inputs"])
        env = make_env(ctx)
        oks = []
        torch.manual_seed(0)
        for st in prog["steps"][:upto]:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    v = eval(st["expr"], env)
                env[st["out"]] = v
                oks.append(True)
            except Exception:
                oks.append(False)
        return env, ctx, oks

    def repair(self, prog):
        for _ in range(50):
            oks = self.replay(prog)[2]
            if all(oks):
                break
            del prog["steps"][oks.index(False)]
            if not prog["steps"]:
                break
        return prog

    def new_input(self, prog, shape, dtype, kind=None, high=None, rg=None, pl=None):
        spec = dict(shape=list(shape), dtype=dtype, seed=self.rng.randrange(1 << 30))
        if kind:
            spec["kind"], spec["high"] = kind, high
        if rg is None:
            rg = prog.get("grad", False) and dtype in ("f32", "f64") and self.rng.random() < 0.7
        spec["rg"] = bool(rg)
        spec["pl"] = pl or rand_placements(self.rng, prog["mesh"], len(shape), dtype if kind != "index" else "i64")
        prog["inputs"].append(spec)
        return len(prog["inputs"]) - 1

    def fresh_name(self, prog):
        prog["nv"] = prog.get("nv", 0) + 1
        return f"v{prog['nv'] - 1}"

    def generate(self):
        rng = self.rng
        prog = dict(mesh=rng.choice(self.meshes), inputs=[], steps=[], nv=0, grad=rng.random() < 0.35)
        for _ in range(rng.choice([1, 1, 2, 2, 3])):
            dt = rand_dtype(rng)
            if prog["grad"] and rng.random() < 0.7:
                dt = "f32"
            j = self.new_input(prog, rand_shape(rng), dt)
            prog["steps"].append(dict(out=self.fresh_name(prog), expr=f"mk({j})"))
        self.extend(prog, rng.randint(3, 18))
        if prog["grad"]:
            self.add_backward(prog)
        return prog

    def extend(self, prog, n_new):
        env = self.replay(prog)[0]
        added = 0
        for _ in range(n_new * 12):
            if added >= n_new:
                break
            expr = self.propose(prog, env)
            if expr is None:
                continue
            if self.try_step(prog, env, expr):
                added += 1
            elif MUT_RE.search(expr):
                env = self.replay(prog)[0]
        return prog

    def try_step(self, prog, env, expr):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                v = eval(expr, env)
        except Exception:
            return False
        if v is not None and not isinstance(v, torch.Tensor):
            return False
        if isinstance(v, torch.Tensor) and v.numel() > MAX_NUMEL:
            return False
        name = self.fresh_name(prog)
        env[name] = v
        prog["steps"].append(dict(out=name, expr=expr))
        return True

    def tensors(self, env, prog, pred=lambda t: True):
        names = [s["out"] for s in prog["steps"] if isinstance(env.get(s["out"]), torch.Tensor)]
        return [n for n in names if pred(env[n])]

    def add_backward(self, prog):
        env = self.replay(prog)[0]
        cands = self.tensors(env, prog, lambda t: t.requires_grad and t.dtype in (torch.float32, torch.float64))
        if cands:
            x = self.rng.choice(cands[-4:])
            self.try_step(prog, env, f"BW({x})")

    def propose(self, prog, env):
        rng = self.rng
        names = self.tensors(env, prog)
        if not names:
            return None
        x = rng.choice(names[-5:] if rng.random() < 0.6 else names)
        t = env[x]
        cat = rng.choices(
            ["unary", "binary", "reduce", "shape", "matmul", "index", "inplace", "redist", "loss", "compile"],
            [12, 10, 12, 16, 6, 6, 7, 14, 3, 2 if self.allow_compile else 0],
        )[0]
        return getattr(self, "p_" + cat)(prog, env, names, x, t)

    def rdim(self, t):
        nd = t.dim()
        if nd == 0:
            return self.rng.choice([0, -1])
        return self.rng.randrange(-nd, nd)

    def p_unary(self, prog, env, names, x, t):
        r = self.rng
        if t.dtype == torch.bool:
            ops = [
                "{x}.logical_not()",
                "{x}.int()",
                "{x}.float()",
                "{x} & {x}.logical_not()",
                "~{x}",
                "{x}.long().cumsum(0)" if t.dim() else "{x}.long()",
            ]
        elif t.is_floating_point():
            ops = [
                "-{x}",
                "{x}.abs()",
                "torch.relu({x})",
                "{x}.sign()",
                "torch.exp({x})",
                "torch.tanh({x})",
                "torch.sigmoid({x})",
                "{x} * 2",
                "{x} + 1",
                "{x}.clamp(min=-1, max=2)",
                "{x}.pow(2)",
                "F.gelu({x})",
                "F.silu({x})",
                "{x}.floor()",
                "torch.sqrt({x}.abs())",
                "{x}.reciprocal()",
                "{x}.masked_fill({x} > 0, 1.5)",
                "torch.where({x} > 0, {x}, {x} * 3)",
                "{x}.to(torch.float64)",
                "{x}.to(torch.bfloat16)",
                "{x}.to(torch.int64)",
                "{x}.bool()",
                "{x}.float()",
                "torch.nan_to_num({x}.log())",
                "{x}.detach()",
                "{x}.clone()",
                "{x}.contiguous()",
                "torch.zeros_like({x})",
                "torch.ones_like({x})",
                "torch.full_like({x}, 3)",
                "F.dropout({x}, p=0.0)",
                "{x}.half()",
                "torch.erf({x})",
                "F.softplus({x})",
                "torch.nn.functional.hardtanh({x})",
                "{x}.round()",
                "{x}.new_zeros(({n},))".replace("{n}", str(r.randint(0, 5))),
                "{x}.sum() + {x}",
            ]
        else:
            ops = [
                "-{x}",
                "{x}.abs()",
                "{x} * 3",
                "{x} + 1",
                "{x} % 3",
                "{x} // 2",
                "{x}.float()",
                "{x}.bool()",
                "{x}.clamp(-1, 2)",
                "{x} & 3",
                "{x}.to(torch.int32)",
                "{x}.to(torch.int64)",
                "torch.bitwise_left_shift({x}, 1)",
                "{x}.clone()",
                "{x}.sign()",
            ]
        return r.choice(ops).format(x=x)

    def p_binary(self, prog, env, names, x, t):
        r = self.rng
        mode = r.random()
        if mode < 0.45:
            same = [n for n in names if tuple(env[n].shape) == tuple(t.shape) and n != x]
            y = r.choice(same) if same else x
        elif mode < 0.7:
            y = r.choice(names)
        else:
            shp = list(t.shape)
            if shp and r.random() < 0.5:
                k = r.randrange(len(shp) + 1)
                shp = shp[k:]
                shp = [1 if r.random() < 0.3 else s for s in shp]
            dt = r.choice(["f32", "f32", "i64", "bool", "f64"]) if r.random() < 0.3 else self.dtcode(t)
            y = self.partner(prog, env, shp, dt)
        ops = [
            "{a} + {b}",
            "{a} - {b}",
            "{a} * {b}",
            "torch.maximum({a}, {b})",
            "torch.minimum({a}, {b})",
            "{a} == {b}",
            "{a} < {b}",
            "torch.where({a} > 0, {a}, {b})",
            "torch.add({a}, {b}, alpha=2)",
            "{a} / {b}",
            "torch.atan2({a}.float(), {b}.float())",
            "{a} + 2.5 * {b}",
            "{a} * {b}.sum()",
            "{a}.logical_and({b})",
            "torch.lerp({a}, {b}, 0.5)",
            "{b} - {a}",
        ]
        a, b = (x, y) if r.random() < 0.8 else (y, x)
        return r.choice(ops).format(a=a, b=b)

    def p_reduce(self, prog, env, names, x, t):
        r = self.rng
        d = self.rdim(t)
        k = r.choice([True, False])
        fl = t.is_floating_point()
        ops = [
            "{x}.sum()",
            "{x}.sum(dim={d}, keepdim={k})",
            "{x}.amax(dim={d}, keepdim={k})",
            "{x}.amin(dim={d}, keepdim={k})",
            "{x}.max(dim={d})[0]",
            "{x}.max(dim={d})[1]",
            "{x}.argmax(dim={d})",
            "{x}.argmin(dim={d}, keepdim={k})",
            "{x}.max()",
            "{x}.min()",
            "{x}.argmax()",
            "{x}.prod(dim={d})",
            "{x}.all(dim={d})",
            "{x}.any(dim={d})",
            "{x}.any()",
            "{x}.count_nonzero(dim={d})",
            "{x}.cumsum(dim={d})",
            "{x}.sort(dim={d})[0]",
            "{x}.sort(dim={d}, stable=True)[1]",
            "{x}.argsort(dim={d}, stable=True)",
            "{x}.topk({kk}, dim={d})[0]",
            "{x}.min(dim={d}, keepdim={k})[1]",
            "{x}.cummax(dim={d})[0]",
            "{x}.sum(dim=({d}, {d2}))",
            "{x}.nansum(dim={d})",
            "torch.aminmax({x})[0]",
        ]
        if fl:
            ops += [
                "{x}.mean(dim={d}, keepdim={k})",
                "{x}.mean()",
                "torch.linalg.vector_norm({x}, dim={d})",
                "{x}.norm()",
                "{x}.var(dim={d})",
                "{x}.std(dim={d}, correction=0)",
                "torch.logsumexp({x}, dim={d})",
                "torch.softmax({x}, dim={d})",
                "torch.log_softmax({x}, dim={d})",
                "F.normalize({x}, dim={d})",
                "{x}.var(dim={d}, correction=0, keepdim={k})",
                "{x}.cumprod(dim={d})",
                "torch.linalg.vector_norm({x}, ord=float('inf'), dim={d})",
                "{x}.mean(dim=({d}, {d2}))",
            ]
            if t.dim() >= 1:
                ops += ["F.layer_norm({x}, [{last}])", "F.rms_norm({x}, [{last}])"]
        size = t.shape[d] if t.dim() else 1
        kk = r.randint(0, max(0, size))
        d2 = self.rdim(t)
        op = r.choice(ops)
        return op.format(x=x, d=d, k=k, kk=kk, d2=d2, last=t.shape[-1] if t.dim() else 1)

    def p_shape(self, prog, env, names, x, t):
        r = self.rng
        nd = t.dim()
        shp = list(t.shape)
        ops = []
        if nd <= 2:
            ops.append(f"{x}.t()")
        if nd >= 2:
            i, j = r.sample(range(nd), 2)
            ops += [
                f"{x}.transpose({i}, {j})",
                f"{x}.mT",
                f"{x}.movedim({i}, {j})",
                f"{x}.diagonal()",
                f"torch.tril({x})",
                f"torch.triu({x}, 1)",
                f"{x}.flatten({min(i, j)}, {max(i, j)})",
            ]
        if nd >= 1:
            perm = list(range(nd))
            r.shuffle(perm)
            d = r.randrange(nd)
            sz = shp[d]
            s0 = r.randint(0, sz)
            ln = r.randint(0, sz - s0)
            st = r.choice([1, 1, 2, 3])
            ops += [
                f"{x}.permute({tuple(perm)})",
                f"{x}.flatten()",
                f"{x}.unsqueeze({r.randint(-nd - 1, nd)})",
                f"{x}.narrow({d}, {s0}, {ln})",
                f"{x}.flip({d})",
                f"{x}.roll({r.randint(-3, 3)}, {d})",
                f"{x}.chunk({r.randint(1, 4)}, {d})[0]",
                f"{x}.chunk({r.randint(1, 4)}, {d})[-1]",
                f"{x}.split({r.randint(1, 4)}, {d})[{r.randint(0, 2)}]",
                f"{x}[{':, ' * d}{s0}:{s0 + ln}:{st}]",
                f"{x}[{':, ' * d}{r.randint(-sz, max(sz - 1, -sz))}]" if sz else f"{x}[None]",
                f"{x}[..., None]",
                f"{x}.squeeze()",
                f"{x}.squeeze({d})",
                f"{x}.select({d}, {r.randint(0, max(sz - 1, 0))})",
                f"{x}.unbind({d})[{r.randint(0, max(sz - 1, 0))}]",
                f"{x}.contiguous()",
                f"{x}.repeat({', '.join(str(r.choice([1, 1, 2, 3])) for _ in range(nd))})",
                f"{x}.expand({', '.join('-1' if s != 1 else str(r.choice([1, 2, 3])) for s in shp)})",
                f"{x}.unsqueeze(0).expand({r.choice([2, 3])}, {', '.join(['-1'] * nd)})",
                f"torch.cat([{x}, {x} * 2], dim={d})",
                f"torch.stack([{x}, {x}], dim={r.randint(0, nd)})",
                f"torch.cat([{x}, {r.choice(names)}], dim={d})",
                f"{x}.tile(({r.choice([1, 2])},))",
                f"{x}.unfold({d}, {max(1, min(2, sz))}, 1)" if sz else f"{x}.clone()",
            ]
            n = t.numel()
            ops += [
                f"{x}.reshape({self.factor(n)})",
                f"{x}.view({self.factor(n)})",
                f"{x}.reshape(-1)",
                f"{x}.view(-1)",
            ]
            if shp[d] in (4, 6, 8, 9, 12):
                a = r.choice([f for f in (2, 3, 4) if shp[d] % f == 0] or [1])
                ops.append(f"{x}.unflatten({d}, ({a}, {shp[d] // a}))")
        else:
            ops += [f"{x}.unsqueeze(0)", f"{x}.reshape(1, 1)", f"{x}.view(-1)", f"{x}.expand(3)", f"{x}[None]"]
        return r.choice(ops)

    def factor(self, n):
        r = self.rng
        if n == 0:
            return r.choice(["(0,)", "(0, 3)", "(2, 0)"])
        fs = []
        m = n
        while m > 1 and len(fs) < 3:
            divs = [d for d in range(2, m + 1) if m % d == 0]
            d = r.choice(divs)
            fs.append(d)
            m //= d
        if m > 1:
            fs.append(m)
        if r.random() < 0.3:
            fs.insert(r.randint(0, len(fs)), 1)
        r.shuffle(fs)
        if not fs:
            fs = [1]
        return "(" + ", ".join(map(str, fs)) + ("," if len(fs) == 1 else "") + ")"

    def partner(self, prog, env, shape, dtype, **kw):
        j = self.new_input(prog, shape, dtype, **kw)
        name = self.fresh_name(prog)
        prog["steps"].append(dict(out=name, expr=f"mk({j})"))
        env[name] = eval(f"mk({j})", env)
        return name

    def dtcode(self, t):
        return {
            torch.float32: "f32",
            torch.float64: "f64",
            torch.bfloat16: "bf16",
            torch.float16: "f16",
            torch.int64: "i64",
            torch.int32: "i32",
            torch.bool: "bool",
        }[t.dtype]

    def p_matmul(self, prog, env, names, x, t):
        r = self.rng
        if not t.is_floating_point() or t.dim() == 0 or t.dtype == torch.float16:
            return None
        dt = self.dtcode(t)
        shp = list(t.shape)
        m = r.choice([1, 2, 3, 4, 5, 8])
        if t.dim() == 1:
            y = self.partner(prog, env, [shp[0]] if r.random() < 0.5 else [shp[0], m], dt)
            ops = [f"{x} @ {y}", f"torch.matmul({x}, {y})", f"torch.outer({x}, {x})"]
            if env[y].dim() == 1:
                ops.append(f"torch.dot({x}, {y})")
            return r.choice(ops)
        k = shp[-1]
        c = r.random()
        if c < 0.5:
            y = self.partner(prog, env, [k, m], dt)
            ops = [f"{x} @ {y}", f"torch.matmul({x}, {y})"]
            if t.dim() == 2:
                ops += [f"torch.mm({x}, {y})", f"torch.einsum('ij,jk->ik', {x}, {y})"]
                b = self.partner(prog, env, [shp[0], m] if r.random() < 0.5 else [m], dt)
                ops.append(f"torch.addmm({b}, {x}, {y})")
            return r.choice(ops)
        if c < 0.8:
            w = self.partner(prog, env, [m, k], dt)
            if r.random() < 0.5:
                b = self.partner(prog, env, [m], dt)
                return f"F.linear({x}, {w}, {b})"
            return f"F.linear({x}, {w})"
        if t.dim() == 3:
            y = self.partner(prog, env, [shp[0], k, m], dt)
            return r.choice([f"torch.bmm({x}, {y})", f"{x} @ {y}", f"torch.einsum('bij,bjk->bik', {x}, {y})"])
        if t.dim() == 2:
            y = self.partner(prog, env, [k], dt)
            return f"torch.mv({x}, {y})"
        return f"{x} @ {x}.transpose(-1, -2)"

    def p_index(self, prog, env, names, x, t):
        r = self.rng
        if t.dim() == 0:
            return None
        d = r.randrange(t.dim())
        sz = t.shape[d]
        c = r.random()
        if c < 0.25:
            idx = self.idx_input(prog, env, [r.randint(0, 6)], sz)
            return r.choice([f"{x}.index_select({d}, {idx})", f"{x}[{':, ' * d}{idx}]"])
        if c < 0.45:
            ishape = list(t.shape)
            ishape[d] = r.randint(0, 5)
            ishape = [min(s, r.randint(0, s)) if i != d else s for i, s in enumerate(ishape)]
            idx = self.idx_input(prog, env, ishape, sz)
            if r.random() < 0.5:
                return f"torch.gather({x}, {d}, {idx})"
            src = self.partner(prog, env, ishape, self.dtcode(t))
            return r.choice(
                [
                    f"{x}.scatter_add({d}, {idx}, {src})",
                    f"{x}.scatter_reduce({d}, {idx}, {src}, reduce='amax')",
                    f"{x}.scatter_reduce({d}, {idx}, {src}, reduce='sum', include_self=False)",
                ]
            )
        if c < 0.6 and t.dim() == 2 and t.is_floating_point():
            idx = self.idx_input(prog, env, rand_shape(r, r.choice([1, 2])), t.shape[0])
            return f"F.embedding({idx}, {x})"
        if c < 0.7:
            idx = self.idx_input(prog, env, [r.randint(0, 5)], sz)
            src_shape = list(t.shape)
            src_shape[d] = env[idx].shape[0]
            src = self.partner(prog, env, src_shape, self.dtcode(t))
            return f"{x}.index_add({d}, {idx}, {src})"
        if c < 0.8:
            return r.choice([f"{x}.masked_select({x} > 0)", f"{x}.nonzero()", f"{x}[{x} > 0]"])
        if c < 0.9:
            idx = self.idx_input(prog, env, rand_shape(r, 1), max(1, r.randint(1, 6)))
            return f"F.one_hot({idx}, {max(1, env[idx].max().item() + 1) if env[idx].numel() else 3})"
        return f"{x}.index_fill({d}, {self.idx_input(prog, env, [r.randint(0, 3)], sz)}, 7)"

    def idx_input(self, prog, env, shape, high):
        return self.partner(
            prog,
            env,
            shape,
            "i64",
            kind="index",
            high=high,
            rg=False,
            pl=rand_placements(self.rng, prog["mesh"], len(shape), "bool"),
        )

    def p_inplace(self, prog, env, names, x, t):
        r = self.rng
        same = [n for n in names if tuple(env[n].shape) == tuple(t.shape) and n != x]
        y = r.choice(same) if same else None
        nd = t.dim()
        ops = [f"{x}.mul_(2)", f"{x}.zero_()", f"{x}.fill_(3)", f"{x}.add_(1)", f"{x}.masked_fill_({x} > 0, 1)"]
        if t.is_floating_point() or t.dtype in (torch.int64, torch.int32):
            ops += [f"{x}.clamp_(min=0)", f"{x}.neg_()", f"{x}.abs_()"]
        if t.is_floating_point():
            ops += [f"{x}.relu_()", f"{x}.div_(2)", f"{x}.sigmoid_()"]
        if y:
            ops += [f"{x}.add_({y})", f"{x}.copy_({y})", f"{x}.mul_({y})", f"{x}.sub_({y}, alpha=2)"]
        if nd >= 1 and t.shape[0] > 0:
            i = r.randrange(t.shape[0])
            ops += [f"SETITEM({x}, {i}, 5)", f"SETITEM({x}, slice(0, {i + 1}), 0)"]
        if nd >= 2:
            ops += [f"{x}.transpose_(0, 1)", f"{x}.t_()" if nd == 2 else f"{x}.squeeze_()"]
        if nd >= 1:
            ops += [
                f"{x}.unsqueeze_(0)",
                f"{x}.cumsum_({r.randrange(nd)})" if t.dtype != torch.bool else f"{x}.zero_()",
            ]
        return r.choice(ops)

    def p_redist(self, prog, env, names, x, t):
        pl = rand_placements(self.rng, prog["mesh"], t.dim(), self.dtcode(t))
        return f"R({x}, {pl!r})"

    def p_loss(self, prog, env, names, x, t):
        r = self.rng
        if not t.is_floating_point() or t.dtype in (torch.float16, torch.bfloat16):
            return None
        if t.dim() == 2 and t.shape[1] > 0:
            tgt = self.idx_input(prog, env, [t.shape[0]], t.shape[1])
            return r.choice(
                [
                    f"F.cross_entropy({x}, {tgt})",
                    f"F.nll_loss(torch.log_softmax({x}, -1), {tgt})",
                    f"F.cross_entropy({x}, {tgt}, reduction='sum')",
                    f"F.cross_entropy({x}, {tgt}, reduction='none')",
                ]
            )
        y = self.partner(prog, env, list(t.shape), self.dtcode(t))
        return r.choice(
            [
                f"F.mse_loss({x}, {y})",
                f"F.l1_loss({x}, {y}, reduction='sum')",
                f"F.binary_cross_entropy_with_logits({x}, torch.sigmoid({y}))",
                f"F.smooth_l1_loss({x}, {y})",
            ]
        )

    def p_compile(self, prog, env, names, x, t):
        lam = self.rng.choice(LAMBDAS)
        return f"CMP({lam!r}, {x})"

    def mutate(self, prog):
        r = self.rng
        p = copy.deepcopy(prog)
        m = r.choice(
            [
                "regen_tail",
                "placements",
                "placements",
                "mesh",
                "insert_redist",
                "delete",
                "insert_op",
                "dtype_shape",
                "grad",
                "extend",
            ]
        )
        p["mut"] = m
        if m == "regen_tail" and len(p["steps"]) > 2:
            k = r.randint(1, len(p["steps"]) - 1)
            p["steps"] = p["steps"][:k]
            self.extend(p, r.randint(1, 8))
        elif m == "placements" and p["inputs"]:
            for spec in r.sample(p["inputs"], r.randint(1, len(p["inputs"]))):
                spec["pl"] = rand_placements(
                    r, p["mesh"], len(spec["shape"]), "bool" if spec.get("kind") == "index" else spec["dtype"]
                )
        elif m == "mesh" and len(self.meshes) > 1:
            p["mesh"] = r.choice([mm for mm in self.meshes if mm != p["mesh"]])
            for spec in p["inputs"]:
                spec["pl"] = rand_placements(
                    r, p["mesh"], len(spec["shape"]), "bool" if spec.get("kind") == "index" else spec["dtype"]
                )
            nd = MESH_NDIM[p["mesh"]]
            for st in p["steps"]:
                pr = parse_R(st["expr"])
                if pr and r.random() < 0.9:
                    st["expr"] = f"R({pr[0]}, {(pr[1] + ['R'] * nd)[:nd]!r})"
        elif m == "insert_redist" and len(p["steps"]) > 1:
            env = self.replay(p)[0]
            k = r.randrange(len(p["steps"]))
            src = p["steps"][k]["out"]
            v = env.get(src)
            if isinstance(v, torch.Tensor):
                name = self.fresh_name(p)
                pl = rand_placements(r, p["mesh"], v.dim(), self.dtcode(v))
                pat = re.compile(rf"\b{re.escape(src)}\b")
                for st in p["steps"][k + 1 :]:
                    st["expr"] = pat.sub(name, st["expr"])
                p["steps"].insert(k + 1, dict(out=name, expr=f"R({src}, {pl!r})"))
        elif m == "delete" and len(p["steps"]) > 2:
            del p["steps"][r.randrange(len(p["steps"]))]
        elif m == "insert_op" and len(p["steps"]) > 1:
            k = r.randint(1, len(p["steps"]))
            head, tail = p["steps"][:k], p["steps"][k:]
            p["steps"] = head
            self.extend(p, 1)
            p["steps"] += tail
        elif m == "dtype_shape" and p["inputs"]:
            spec = r.choice(p["inputs"])
            if spec.get("kind") != "index":
                if r.random() < 0.5:
                    spec["dtype"] = rand_dtype(r)
                    spec["rg"] = spec["rg"] and spec["dtype"] in ("f32", "f64")
                    if any(pl.startswith("P:") for pl in spec["pl"]):
                        spec["pl"] = rand_placements(r, p["mesh"], len(spec["shape"]), spec["dtype"])
                else:
                    spec["shape"] = [max(0, s + r.choice([-1, 1, 2])) for s in spec["shape"]]
                    spec["pl"] = rand_placements(r, p["mesh"], len(spec["shape"]), spec["dtype"])
        elif m == "grad":
            p["grad"] = True
            for spec in p["inputs"]:
                if spec["dtype"] in ("f32", "f64") and spec.get("kind") != "index":
                    spec["rg"] = True
            p["steps"] = [s for s in p["steps"] if not s["expr"].startswith("BW(")]
            self.repair(p)
            self.add_backward(p)
        else:
            self.extend(p, r.randint(1, 5))
        self.repair(p)
        return p
