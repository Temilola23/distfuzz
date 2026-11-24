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
MUT_RE = re.compile(r"_\(|SETITEM|BW\(|inplace=True")
MESH_NDIM = {"1d": 1, "2d": 2}
FOREACH_UNARY = (
    "abs acos asin atan ceil cos cosh erf erfc exp expm1 floor frac lgamma log log10 log1p log2 neg reciprocal round "
    "rsqrt sigmoid sign sin sinh sqrt tan tanh trunc"
).split()

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
            [
                "unary",
                "binary",
                "reduce",
                "shape",
                "matmul",
                "index",
                "inplace",
                "redist",
                "loss",
                "nn",
                "linalg",
                "tail",
                "compile",
            ],
            [12, 10, 12, 16, 7, 6, 8, 14, 4, 9, 6, 6, 2 if self.allow_compile else 0],
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
                "torch.sin({x})",
                "torch.log1p({x}.abs())",
                "torch.expm1({x})",
                "{x}.trunc()",
                "{x}.frac()",
                "{x}.square()",
                "{x}.isfinite()",
                "F.leaky_relu({x}, 0.25)",
                "F.elu({x})",
                "F.hardsigmoid({x})",
                "F.relu6({x})",
                # no copysign/signbit: a Partial(sum) cannot keep the sign of a zero, so -0.0 vs +0.0 would be
                # reported as a value mismatch
            ]
            # the long tail, drawn at a fixed rate; domains kept valid (clamp, abs + 1, sigmoid) so most outputs
            # are finite
            bulk = [
                "torch.acos({x}.clamp(-1, 1))",
                "torch.asin({x}.clamp(-1, 1))",
                "torch.atan({x})",
                "torch.acosh({x}.abs() + 1)",
                "torch.asinh({x})",
                "torch.atanh({x}.clamp(-0.5, 0.5))",
                "torch.cosh({x})",
                "torch.sinh({x})",
                "torch.tan({x})",
                "{x}.ceil()",
                "torch.deg2rad({x})",
                "torch.rad2deg({x})",
                "torch.erfc({x})",
                "torch.erfinv({x}.clamp(-0.5, 0.5))",
                "torch.exp2({x})",
                "torch.log2({x}.abs() + 1)",
                "torch.log10({x}.abs() + 1)",
                "torch.lgamma({x}.abs() + 0.5)",
                "torch.digamma({x}.abs() + 0.5)",
                "torch.logit(torch.sigmoid({x}))",
                "torch.rsqrt({x}.abs() + 1)",
                "torch.sinc({x})",
                "torch.i0({x})",
                "F.mish({x})",
                "F.selu({x})",
                "F.celu({x})",
                "F.hardshrink({x}, 1.5)",
                "F.softshrink({x}, 1.5)",
                "F.threshold({x}, 0.5, -2.0)",
                "{x}.clamp_max(1)",
                "{x}.clamp_min(0)",
                "{x}.clip(-2, 2)",
                "{x}.sgn()",
                "torch.positive({x})",
                "torch.isinf({x}.reciprocal())",
                "torch.isposinf({x}.reciprocal())",
                "torch.isneginf({x}.reciprocal())",
                "torch.nan_to_num({x}.reciprocal(), posinf=9.0, neginf=-9.0)",
                "torch.float_power({x}, 2)",
                "2 - {x}",
                "torch.true_divide({x}, 4)",
                "torch.frexp({x})[0]",
                "torch.special.i0e({x})",
                "torch.special.i1({x})",
                "torch.special.i1e({x})",
                "torch.special.erfcx({x})",
                "torch.special.ndtri(torch.sigmoid({x}))",
                "torch.special.log_ndtr({x})",
                "torch.special.entr({x}.abs())",
                "torch.special.bessel_j0({x})",
                "torch.special.bessel_j1({x})",
                "torch.special.chebyshev_polynomial_t({x}.clamp(-1, 1), 3)",
                "torch.special.hermite_polynomial_h({x}, 2)",
                "torch.special.legendre_polynomial_p({x}.clamp(-1, 1), 2)",
                "torch.special.modified_bessel_i0({x})",
                "torch.special.spherical_bessel_j0({x})",
                "torch.special.zeta({x}.abs() + 2, 1.0)",
                "torch.mvlgamma({x}.abs() + 2, 1)",
                "torch.polygamma(1, {x}.abs() + 1)",
                "{x}.new_full((2, 3), 7)",
                "{x}.new_ones(({n},))".replace("{n}", str(r.randint(0, 5))),
                "torch.fill({x}, 2)",
                "torch.histc({x}.float(), bins=4, min=-4, max=4)",
            ]
            if r.random() < 0.4:
                ops = bulk
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
                "{x} ^ 3",
                "{x} | 1",
                "~{x}",
                "{x} << 1",
                "{x} >> 1",
                "torch.bitwise_right_shift({x}, 1)",
                "torch.gcd({x}, {x} + 6)",
                "torch.lcm({x}, {x}.abs() % 3 + 1)",
                "torch.remainder({x}, 3)",
                "{x}.clamp_min(0)",
                "{x}.clamp_max(1)",
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
            "torch.fmod({a}, 3)",
            "torch.fmax({a}.float(), {b}.float())",
            "torch.hypot({a}.float(), {b}.float())",
            "torch.logaddexp({a}.float(), {b}.float())",
            "torch.xlogy({a}.float().abs(), {b}.float().abs() + 1)",
            "torch.heaviside({a}.float(), {b}.float())",
            "{a} != {b}",
            "{a} >= {b}",
            "{a} > {b}",
            "{a} <= {b}",
            "torch.logical_xor({a}, {b})",
            "{a}.bool() | {b}.bool()",
            "{a}.bool() ^ {b}.bool()",
            "torch.fmin({a}.float(), {b}.float())",
            "torch.logaddexp2({a}.float(), {b}.float())",
            "torch.nextafter({a}.float(), {b}.float())",
            "torch.float_power({a}, {b}.abs() % 3)",
            "torch.addcmul({a}, {a}, {b}, value=2)",
            "torch.addcdiv({a}.float(), {b}.float(), {b}.float().abs() + 1)",
            "torch.special.xlog1py({a}.float(), {b}.float().abs())",
            "torch.ldexp({a}.float(), {b}.abs().clamp(max=4).int())",
            "torch.remainder({a}, {b}.abs() + 1)",
        ]
        if torch.bfloat16 in (t.dtype, env[y].dtype) or torch.float16 in (t.dtype, env[y].dtype):
            # a half value that matched at half tolerance is compared at float32 tolerance after .float(), and the
            # log terms amplify the difference
            ops = [o for o in ops if "xlog" not in o]
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
                "torch.norm({x}, p=2, dim={d})",
                "{x}.nanmedian(dim={d})[0]",
                "{x}.nanmedian()",
                "torch.linalg.norm({x})",
            ]
            if 1 <= t.dim() <= 2:
                ops.append("torch.cov({x})")
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
                f"{x}.reshape({self.factor(n, shp)})",
                f"{x}.view({self.factor(n, shp)})",
                f"{x}.reshape(-1)",
                f"{x}.view(-1)",
                f"torch.broadcast_to({x}, ({', '.join(str(s if s != 1 else r.choice([1, 2, 3])) for s in shp)}))",
                f"torch.tensor_split({x}, {r.randint(1, 4)}, dim={d})[0]",
            ]
            if n == 0:
                # zero-size views take their own path through view_groups (two hangs so far), so draw them more often
                ops += [f"{x}.{r.choice(['reshape', 'view'])}({self.factor(0, shp)})" for _ in range(4)]
                if sz == 0:
                    zs = [r.choice([0, 0, 1, 2, 3]) for _ in range(r.randint(1, 4))]
                    zs[r.randrange(len(zs))] = 0
                    ops.append(f"{x}.unflatten({d}, ({', '.join(map(str, zs))}{',' if len(zs) == 1 else ''}))")
            if shp[d] in (4, 6, 8, 9, 12):
                a = r.choice([f for f in (2, 3, 4) if shp[d] % f == 0] or [1])
                ops.append(f"{x}.unflatten({d}, ({a}, {shp[d] // a}))")
        else:
            ops += [f"{x}.unsqueeze(0)", f"{x}.reshape(1, 1)", f"{x}.view(-1)", f"{x}.expand(3)", f"{x}[None]"]
        return r.choice(ops)

    def factor(self, n, src=()):
        r = self.rng
        if n == 0:
            return self.zero_shape(src)
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

    def zero_shape(self, src):
        """A target shape with numel 0 for reshaping a zero-size tensor of shape src: several zero dims, zero dims
        after the source's dims ((4, 0) -> (4, 2, 0, 0, 0)), or a zero followed by larger dims ((0,) -> (0, 5))."""
        r = self.rng
        c = r.random()
        if c < 0.2:
            return r.choice(["(0,)", "(0, 3)", "(2, 0)"])
        if c < 0.55 and src:
            # keep a prefix of the source shape, then a tail that contains a zero
            head = list(src[: r.randint(0, len(src))])
            tail = [r.choice([0, 0, 1, 2, 3, 5]) for _ in range(r.randint(1, 4))]
            if 0 not in head + tail:
                tail[r.randrange(len(tail))] = 0
            dims = head + tail
        else:
            dims = [r.choice([0, 0, 1, 2, 3, 4, 5, 7]) for _ in range(r.randint(1, 5))]
            if 0 not in dims:
                dims[r.randrange(len(dims))] = 0
        dims = dims[:6]
        if 0 not in dims:
            dims[-1] = 0
        return "(" + ", ".join(map(str, dims)) + ("," if len(dims) == 1 else "") + ")"

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
        # int64/int32 too: CPU matmul kernels take them, and integer Partial(sum) outputs have their own bugs
        if t.dtype in (torch.bool, torch.float16) or t.dim() == 0:
            return None
        dt = self.dtcode(t)
        shp = list(t.shape)
        m = r.choice([1, 2, 3, 4, 5, 8])
        ab = r.choice(["", ", beta=2, alpha=3", ", beta=0", ", beta=0.5, alpha=2"])
        if t.dim() == 1:
            y = self.partner(prog, env, [shp[0]] if r.random() < 0.5 else [shp[0], m], dt)
            ops = [f"{x} @ {y}", f"torch.matmul({x}, {y})", f"torch.outer({x}, {x})"]
            if env[y].dim() == 1:
                ops.append(f"torch.dot({x}, {y})")
                b = self.partner(prog, env, [shp[0], shp[0]] if r.random() < 0.7 else [1], dt)
                ops.append(f"torch.addr({b}, {x}, {y}{ab})")
            return r.choice(ops)
        k = shp[-1]
        c = r.random()
        if c < 0.4:
            y = self.partner(prog, env, [k, m], dt)
            ops = [f"{x} @ {y}", f"torch.matmul({x}, {y})", f"torch.tensordot({x}, {y}, dims=1)"]
            if t.dim() == 2:
                ops += [f"torch.mm({x}, {y})", f"torch.einsum('ij,jk->ik', {x}, {y})"]
                b = self.partner(prog, env, [shp[0], m] if r.random() < 0.5 else [m], dt)
                ops += [f"torch.addmm({b}, {x}, {y}{ab})", f"{b}.addmm_({x}, {y}{ab})"]
            return r.choice(ops)
        if c < 0.65:
            w = self.partner(prog, env, [m, k], dt)
            if r.random() < 0.5:
                b = self.partner(prog, env, [m], dt)
                return f"F.linear({x}, {w}, {b})"
            return f"F.linear({x}, {w})"
        if t.dim() == 3:
            y = self.partner(prog, env, [shp[0], k, m], dt)
            ops = [f"torch.bmm({x}, {y})", f"{x} @ {y}", f"torch.einsum('bij,bjk->bik', {x}, {y})"]
            bshape = r.choice([[shp[0], shp[1], m], [shp[1], m], [1, m], [m]])
            b = self.partner(prog, env, bshape, dt)
            ops += [f"torch.baddbmm({b}, {x}, {y}{ab})"]
            if len(bshape) == 3:
                ops.append(f"{b}.baddbmm_({x}, {y}{ab})")
            b2 = self.partner(prog, env, [shp[1], m] if r.random() < 0.7 else [m], dt)
            ops.append(f"torch.addbmm({b2}, {x}, {y}{ab})")
            return r.choice(ops)
        if t.dim() == 2:
            y = self.partner(prog, env, [k], dt)
            b = self.partner(prog, env, [shp[0]] if r.random() < 0.7 else [1], dt)
            return r.choice([f"torch.mv({x}, {y})", f"torch.addmv({b}, {x}, {y}{ab})"])
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
                return r.choice([f"torch.gather({x}, {d}, {idx})", f"torch.take_along_dim({x}, {idx}, dim={d})"])
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
            return r.choice(
                [
                    f"{x}.index_add({d}, {idx}, {src})",
                    f"{x}.index_reduce({d}, {idx}, {src}, '{r.choice(['prod', 'amax', 'amin', 'mean'])}')",
                    f"{x}.index_reduce_({d}, {idx}, {src}, 'amax', include_self=False)",
                ]
            )
        if c < 0.8:
            return r.choice([f"{x}.masked_select({x} > 0)", f"{x}.nonzero()", f"{x}[{x} > 0]"])
        if c < 0.9:
            idx = self.idx_input(prog, env, rand_shape(r, 1), max(1, r.randint(1, 6)))
            return f"F.one_hot({idx}, {max(1, env[idx].max().item() + 1) if env[idx].numel() else 3})"
        if c < 0.95:
            idx = self.idx_input(prog, env, [r.randint(0, 4)], t.shape[0])
            v = self.partner(prog, env, [env[idx].shape[0], *t.shape[1:]] if r.random() < 0.5 else [], self.dtcode(t))
            return r.choice(
                [
                    f"{x}.index_put(({idx},), {v})",
                    f"{x}.index_put(({idx},), {v}, accumulate=True)",
                    f"{x}.index_put_(({idx},), {v}, accumulate=True)",
                ]
            )
        fi = self.idx_input(prog, env, [r.randint(0, 3)], sz)
        return r.choice([f"{x}.index_fill({d}, {fi}, 7)", f"{x}.index_fill_({d}, {fi}, 1)"])

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
        bulk = []
        ops = [f"{x}.mul_(2)", f"{x}.zero_()", f"{x}.fill_(3)", f"{x}.add_(1)", f"{x}.masked_fill_({x} > 0, 1)"]
        if t.is_floating_point() or t.dtype in (torch.int64, torch.int32):
            ops += [f"{x}.clamp_(min=0)", f"{x}.neg_()", f"{x}.abs_()"]
        if t.dtype in (torch.int64, torch.int32):
            ops += [
                f"{x}.bitwise_not_()",
                f"{x}.bitwise_or_(1)",
                f"{x}.bitwise_xor_(3)",
                f"{x}.bitwise_left_shift_(1)",
                f"{x}.bitwise_right_shift_(1)",
                f"{x}.remainder_(3)",
                f"{x}.fmod_(3)",
                f"{x}.gcd_({x} + 6)",
                f"{x}.lcm_({x}.abs() % 3 + 1)",
                f"{x}.clamp_min_(0)",
                f"{x}.clamp_max_(1)",
            ]
        if t.dtype == torch.bool:
            ops += [f"{x}.logical_not_()", f"{x}.logical_xor_({x}.logical_not())", f"{x}.bitwise_not_()"]
        if t.is_floating_point():
            ops += [f"{x}.relu_()", f"{x}.div_(2)", f"{x}.sigmoid_()"]
            # one in-place call per step, so opname() names it; out-of-domain inputs give NaN on both sides
            bulk = [
                f"{x}.{op}()"
                for op in (
                    "sqrt_ exp_ log_ log1p_ log2_ log10_ rsqrt_ sin_ cos_ tan_ sinh_ cosh_ tanh_ atan_ asinh_ asin_ "
                    "acos_ atanh_ acosh_ erf_ erfc_ erfinv_ expm1_ exp2_ lgamma_ digamma_ floor_ ceil_ round_ trunc_ "
                    "frac_ sign_ sgn_ reciprocal_ square_ deg2rad_ rad2deg_ sinc_ i0_ logit_ nan_to_num_ detach_"
                ).split()
            ]
            bulk += [f"{x}.pow_(2)", f"{x}.clamp_min_(0)", f"{x}.clamp_max_(1)", f"{x}.clip_(-1, 1)"]
            bulk += [f"{x}.true_divide_(4)", f"{x}.fmod_(3)", f"{x}.remainder_(3)"]
            if nd >= 2:
                bulk += [f"{x}.tril_()", f"{x}.triu_(1)"]
            bulk += [
                f"F.hardtanh({x}, inplace=True)",
                f"F.relu6({x}, inplace=True)",
                f"F.leaky_relu_({x}, 0.25)",
                f"F.elu_({x})",
                f"F.celu({x}, inplace=True)",
                f"F.selu({x}, inplace=True)",
                f"F.silu({x}, inplace=True)",
                f"F.mish({x}, inplace=True)",
                f"F.hardsigmoid({x}, inplace=True)",
                f"F.threshold_({x}, 0.5, -2.0)",
            ]
            if nd >= 1:
                ops.append(f"{x}.cumprod_({r.randrange(nd)})")
        if y:
            ops += [f"{x}.add_({y})", f"{x}.copy_({y})", f"{x}.mul_({y})", f"{x}.sub_({y}, alpha=2)"]
            ops += [f"{x}.{op}_({y})" for op in ("eq", "ne", "ge", "gt", "le", "lt")]
            if t.is_floating_point():
                ops += [
                    f"{x}.lerp_({y}, 0.5)",
                    f"{x}.atan2_({y})",
                    f"{x}.hypot_({y})",
                    f"{x}.heaviside_({y})",
                    f"{x}.xlogy_({y})",
                    f"{x}.addcmul_({y}, {y}, value=0.5)",
                    f"{x}.addcdiv_({y}, {y})",
                    f"{x}.nextafter_({y})",
                    f"{x}.float_power_(2)",
                ]
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
        # the long tail of pointwise in-place ops is drawn at a fixed rate so it does not crowd out the rest
        return r.choice(bulk if bulk and r.random() < 0.4 else ops)

    def p_redist(self, prog, env, names, x, t):
        pl = rand_placements(self.rng, prog["mesh"], t.dim(), self.dtcode(t))
        return f"R({x}, {pl!r})"

    def p_loss(self, prog, env, names, x, t):
        r = self.rng
        if not t.is_floating_point() or t.dtype in (torch.float16, torch.bfloat16):
            return None
        red = r.choice(["mean", "mean", "sum", "none"])
        if t.dim() == 2 and t.shape[1] > 0 and r.random() < 0.5:
            tgt = self.idx_input(prog, env, [t.shape[0]], t.shape[1])
            ops = [
                f"F.cross_entropy({x}, {tgt})",
                f"F.nll_loss(torch.log_softmax({x}, -1), {tgt})",
                f"F.cross_entropy({x}, {tgt}, reduction='sum')",
                f"F.cross_entropy({x}, {tgt}, reduction='none')",
                f"F.cross_entropy({x}, {tgt}, label_smoothing=0.1, reduction='{red}')",
                f"F.cross_entropy({x}, torch.softmax({x} * 0.5, -1), reduction='{red}')",
                f"F.multi_margin_loss({x}, {tgt}, reduction='{red}')",
                f"F.multilabel_soft_margin_loss({x}, ({x} > 0).to({x}.dtype), reduction='{red}')",
            ]
            if r.random() < 0.3:
                w = self.partner(prog, env, [t.shape[1]], self.dtcode(t))
                ops += [
                    f"F.cross_entropy({x}, {tgt}, weight={w}.abs() + 1, reduction='{red}')",
                    f"F.nll_loss(torch.log_softmax({x}, -1), {tgt}, weight={w}.abs() + 1, reduction='{red}')",
                ]
            return r.choice(ops)
        if t.dim() == 4 and t.shape[1] > 0 and r.random() < 0.5:
            # (N, C, H, W) with an (N, H, W) target: nll_loss2d
            tgt = self.idx_input(prog, env, [t.shape[0], t.shape[2], t.shape[3]], t.shape[1])
            return f"F.cross_entropy({x}, {tgt}, reduction='{red}')"
        y = self.partner(prog, env, list(t.shape), self.dtcode(t))
        ops = [
            f"F.mse_loss({x}, {y})",
            f"F.l1_loss({x}, {y}, reduction='sum')",
            f"F.binary_cross_entropy_with_logits({x}, torch.sigmoid({y}))",
            f"F.smooth_l1_loss({x}, {y})",
            f"F.mse_loss({x}, {y}, reduction='{red}')",
            f"F.l1_loss({x}, {y}, reduction='{red}')",
            f"F.huber_loss({x}, {y}, delta={r.choice([0.5, 1.0, 2.0])}, reduction='{red}')",
            f"F.smooth_l1_loss({x}, {y}, beta=0.5, reduction='{red}')",
            f"F.binary_cross_entropy(torch.sigmoid({x}), torch.sigmoid({y}), reduction='{red}')",
            f"F.binary_cross_entropy(torch.sigmoid({x}), ({y} > 0).to({x}.dtype), reduction='{red}')",
            f"F.binary_cross_entropy_with_logits({x}, ({y} > 0).to({x}.dtype), reduction='{red}')",
            f"F.soft_margin_loss({x}, {y}.sign(), reduction='{red}')",
            f"F.hinge_embedding_loss({x}, ({y} > 0).to({x}.dtype) * 2 - 1, reduction='{red}')",
            f"F.poisson_nll_loss({x}, {y}.abs(), reduction='{red}')",
            f"F.kl_div(torch.log_softmax({x}, -1), torch.log_softmax({y}, -1), log_target=True, reduction='{red}')",
            f"F.kl_div(torch.log_softmax({x}, -1), torch.softmax({y}, -1), reduction='batchmean')",
            f"F.gaussian_nll_loss({x}, {y}, {y}.abs() + 1, reduction='{red}')",
            f"F.margin_ranking_loss({x}, {y}, ({x} > {y}).to({x}.dtype) * 2 - 1, reduction='{red}')",
        ]
        if t.dim() == 1:
            ops.append(f"F.binary_cross_entropy_with_logits({x}, torch.sigmoid({y}), pos_weight={y}.abs() + 1)")
        if t.dim() == 2:
            ops += [
                f"F.cosine_embedding_loss({x}, {y}, ({y}.sum(-1) > 0).to({x}.dtype) * 2 - 1, reduction='{red}')",
                f"F.triplet_margin_loss({x}, {y}, {x} * 0.5, reduction='{red}')",
                f"F.multilabel_margin_loss({x}, ({y}.abs() % {t.shape[1] + 1} - 1).long(), reduction='{red}')"
                if t.shape[1] > 0
                else f"F.mse_loss({x}, {y})",
            ]
        return r.choice(ops)

    def p_nn(self, prog, env, names, x, t):
        """Ops with their own sharding strategies: foreach lists, norms, conv, pooling, padding, interpolation,
        order statistics and scatter-views. No attention: on CPU, SDPA dispatches to an op with no strategy."""
        r = self.rng
        nd = t.dim()
        fl = t.is_floating_point() and t.dtype not in (torch.float16, torch.bfloat16)
        groups = []
        if nd >= 1:
            groups += ["order", "scatter"]
        if t.is_floating_point() or t.dtype in (torch.int64, torch.int32):
            groups.append("foreach")
        if fl and nd >= 1:
            groups.append("float")
        if fl and nd == 3:
            # twice: 3-D floats also take "img" (lifted to 4-D), which should not halve conv1d and 1-D upsampling
            groups += ["seq", "seq"]
        if fl and 2 <= nd <= 4:
            groups.append("img")
        if not groups:
            return None
        g = r.choice(groups)
        if g in ("order", "scatter", "float"):
            d = r.randrange(nd)
            sz = t.shape[d]
        if g == "order":
            # values only: the indices of tied integer values are not unique
            ops = [f"{x}.median(dim={d})[0]", f"{x}.cummin(dim={d})[0]", f"{x}.mode(dim={d})[0]"]
            if sz:
                ops.append(f"{x}.kthvalue({r.randint(1, sz)}, dim={d})[0]")
            return r.choice(ops)
        if g == "scatter":
            if not sz:
                return f"F.pad({x}, ({r.randint(0, 2)}, {r.randint(0, 2)}))"
            i, s0 = r.randrange(sz), r.randrange(sz)
            return r.choice(
                [
                    f"F.pad({x}, ({r.randint(0, 2)}, {r.randint(0, 2)}))",
                    f"torch.select_scatter({x}, {x}.select({d}, {i}) * 2, {d}, {i})",
                    f"torch.slice_scatter({x}, {x}.narrow({d}, {s0}, {sz - s0}) + 1, {d}, {s0})",
                ]
            )
        if g == "foreach":
            same = [n for n in names if tuple(env[n].shape) == tuple(t.shape) and env[n].dtype == t.dtype]
            y, k = r.choice(same), r.randrange(2)
            ops = [
                f"torch._foreach_add([{x}, {y}], [{y}, {x}])[{k}]",
                f"torch._foreach_mul([{x}, {y}], 2)[{k}]",
                f"torch._foreach_maximum([{x}, {y}], [{y}, {x}])[{k}]",
                f"torch._foreach_minimum([{x}, {y}], [{y}, {x}])[{k}]",
                f"torch._foreach_sub([{x}, {y}], [{y}, {x}])[{k}]",
                f"torch._foreach_abs([{x}, {y}])[{k}]",
                f"torch._foreach_neg([{x}, {y}])[{k}]",
                f"torch._foreach_clone([{x}, {y}])[{k}]",
                f"torch._foreach_sign([{x}, {y}])[{k}]",
                f"torch._foreach_clamp_min([{x}, {y}], 0)[{k}]",
                f"torch._foreach_clamp_max([{x}, {y}], 1)[{k}]",
            ]
            un = r.choice(FOREACH_UNARY)
            if fl:
                ops += [
                    f"torch._foreach_norm([{x}, {y}])[{k}]",
                    f"torch._foreach_norm([{x}, {y}], ord=float('inf'))[{k}]",
                    f"torch._foreach_lerp([{x}, {y}], [{y}, {x}], 0.5)[{k}]",
                    f"torch._foreach_addcmul([{x}, {y}], [{y}, {x}], [{x}, {y}], value=0.5)[{k}]",
                    f"torch._foreach_addcdiv([{x}, {y}], [{y}, {x}], [{x}.abs() + 1, {y}.abs() + 1])[{k}]",
                    f"torch._foreach_div([{x}, {y}], [{y}.abs() + 1, {x}.abs() + 1])[{k}]",
                    f"torch._foreach_pow([{x}, {y}], 2)[{k}]",
                    f"torch._foreach_sqrt([{x}.abs(), {y}.abs()])[{k}]",
                    f"torch._foreach_max([{x}, {y}])[{k}]",
                    f"torch._foreach_{un}([{x}, {y}])[{k}]",
                ]
            if y != x:
                # in place on both list members (returns the list in torch 2.14). Only for two distinct tensors, and
                # the other lists are clones: with aliasing the result depends on the kernel's update order (the
                # CPU loop and DTensor's foreach differ), which is not a DTensor bug
                b_op = r.choice(["add_", "mul_", "sub_", "maximum_", "clamp_min_"])
                u_op = r.choice(["abs_", "neg_", "zero_", "sign_"])
                ops += [f"torch._foreach_{b_op}([{x}, {y}], 1)[{k}]", f"torch._foreach_{u_op}([{x}, {y}])[{k}]"]
                if fl:
                    yc, xc = f"{y}.clone()", f"{x}.clone()"
                    ops += [
                        f"torch._foreach_{un}_([{x}, {y}])[{k}]",
                        f"torch._foreach_lerp_([{x}, {y}], [{yc}, {xc}], 0.5)[{k}]",
                        f"torch._foreach_addcmul_([{x}, {y}], [{yc}, {xc}], [{xc}, {yc}], value=0.5)[{k}]",
                        f"torch._foreach_addcdiv_([{x}, {y}], [{yc}, {xc}], [{x}.abs() + 1, {y}.abs() + 1])[{k}]",
                        f"torch._foreach_div_([{x}, {y}], 2)[{k}]",
                        f"torch._foreach_pow_([{x}, {y}], 2)[{k}]",
                    ]
            return r.choice(ops)
        if g == "float":
            ops = [f"torch.logcumsumexp({x}, dim={d})", f"torch.var_mean({x}, dim={d})[0]"]
            if nd in (2, 3):
                ops.append(f"torch.diag_embed({x})")
            if nd == 2:
                ops += [f"torch.diag({x})", f"torch.diagonal_scatter({x}, torch.diagonal({x}) + 1)"]
            return r.choice(ops)
        if g == "img":
            # lower-rank inputs are lifted to (N, C, H, W) inside the expression
            lift = "[None]" * (4 - nd)
            return self.img_op(prog, env, f"{x}{lift}", t[(None,) * (4 - nd)])
        # "seq": (batch, channels, length)
        c = t.shape[1]
        if c and r.random() < 0.2:
            w = self.partner(prog, env, [r.choice([1, 2, 3]), c, r.choice([1, 2])], self.dtcode(t))
            return f"F.conv1d({x}, {w})"
        ops = [
            f"F.avg_pool1d({x}, {r.choice([1, 2])})",
            f"F.max_pool1d({x}, {r.choice([1, 2])})",
            f"F.interpolate({x}, scale_factor=2, mode='nearest')",
            f"F.interpolate({x}, size={r.choice([2, 4, 5])}, mode='linear')",
            f"F.interpolate({x}, scale_factor=2, mode='nearest-exact')",
            f"F.adaptive_avg_pool1d({x}, {r.choice([1, 2, 3])})",
            f"F.adaptive_max_pool1d({x}, {r.choice([1, 2, 3])})",
            f"F.max_pool1d({x}, 2, return_indices=True)[1]",
            f"F.instance_norm({x})",
        ]
        if c:
            ops += [
                f"F.group_norm({x}, {r.choice([n for n in (1, 2, 3) if c % n == 0])})",
                f"F.batch_norm({x}, None, None, training=True)",
            ]
        return r.choice(ops)

    def img_op(self, prog, env, x, t):
        """(N, C, H, W) inputs: 2-D pooling, upsampling, pixel shuffle, grid_sample; 3-D ops on the same tensor read
        as an unbatched (C, D, H, W) volume or batched through x[None]."""
        r = self.rng
        n, c, h, w = t.shape
        sz = r.choice([1, 2, 3])
        mode = r.choice(["nearest", "nearest-exact", "bilinear", "bicubic"])
        ops = [
            f"F.adaptive_avg_pool2d({x}, {sz})",
            f"F.adaptive_max_pool2d({x}, {sz})",
            f"F.adaptive_max_pool2d({x}, {sz}, return_indices=True)[1]",
            f"F.avg_pool2d({x}, {r.choice([1, 2])})",
            f"F.max_pool2d({x}, {r.choice([1, 2])})",
            f"F.lp_pool2d({x}, 2, {r.choice([1, 2])})",
            f"F.interpolate({x}, scale_factor=2, mode='{mode}')",
            f"F.interpolate({x}, size=({r.choice([2, 3, 5])}, {r.choice([2, 4])}), mode='bilinear', antialias=True)",
            f"F.interpolate({x}, size=(2, 3), mode='bicubic', antialias=True)",
            f"F.interpolate({x}, scale_factor=0.5, mode='area')",
            f"F.adaptive_avg_pool3d({x}, {sz})",
            f"F.max_pool3d({x}, 1)",
            f"F.avg_pool3d({x}, 1)",
            f"F.adaptive_max_pool3d({x}, {sz})",
            f"F.interpolate({x}[None], scale_factor=2, mode='{r.choice(['nearest', 'trilinear', 'nearest-exact'])}')",
            f"F.instance_norm({x})",
        ]
        if c % 4 == 0 and c:
            ops.append(f"F.pixel_shuffle({x}, 2)")
        if h % 2 == 0 and w % 2 == 0:
            ops.append(f"F.pixel_unshuffle({x}, 2)")
            ops.append(f"F.max_unpool2d(*F.max_pool2d({x}, 2, return_indices=True), 2)")
        if r.random() < 0.15:
            grid = self.partner(prog, env, [n, r.choice([1, 2, 3]), r.choice([1, 2, 3]), 2], self.dtcode(t))
            gmode = r.choice(["bilinear", "nearest", "bicubic"])
            ops.append(f"F.grid_sample({x}, torch.tanh({grid}), mode='{gmode}', align_corners=False)")
        if c and r.random() < 0.15:
            wt = self.partner(prog, env, [r.choice([1, 2]), c, 1, r.choice([1, 2])], self.dtcode(t))
            ops.append(f"F.conv2d({x}, {wt})")
        return r.choice(ops)

    def bcast_like(self, shp, keep=(), drop_max=None):
        """A shape that broadcasts with shp: dims in keep (negative indices) are copied, a size-s dim becomes s or 1,
        a size-1 dim becomes 1 or a larger size (the other operand is then the broadcast one), and up to drop_max
        leading dims are dropped."""
        r = self.rng
        nd = len(shp)
        out = []
        for i, s in enumerate(shp):
            if i - nd in keep:
                out.append(s)
            elif s == 1:
                out.append(r.choice([1, 2, 3, 4, 5]))
            else:
                out.append(r.choice([s, 1]))
        k = r.randint(0, nd if drop_max is None else min(drop_max, nd))
        return out[k:]

    def batch_dims(self):
        r = self.rng
        return [r.choice([1, 1, 2, 3, 4, 5, 0]) for _ in range(r.choice([0, 1, 1, 2]))]

    def p_linalg(self, prog, env, names, x, t):
        """Broadcasting cross products and batched linalg. Both have their own strategies (linalg_cross_strategy,
        linalg_batch_dim_strategy) that shard batch dims, so operands get size-1 batch dims to broadcast."""
        r = self.rng
        if t.dtype not in (torch.float32, torch.float64):
            return None
        c = r.random()
        if c < 0.3:
            return self.cross_op(prog, env, x, t)
        if c < 0.65:
            return self.solve_op(prog, env, x, t)
        return self.decomp_op(prog, env, x, t)

    def cross_op(self, prog, env, x, t):
        r = self.rng
        dt = self.dtcode(t)
        threes = [i - t.dim() for i, s in enumerate(t.shape) if s == 3]
        if threes and r.random() < 0.6:
            a, shp, d = x, list(t.shape), r.choice(threes)
        else:
            shp = [r.choice([1, 1, 2, 3, 4, 5, 0]) for _ in range(r.randint(0, 3))]
            pos = r.randint(0, len(shp))
            shp.insert(pos, 3)
            d = pos - len(shp)
            a = self.partner(prog, env, shp, dt)
        op = r.choice(["torch.linalg.cross", "torch.linalg.cross", "torch.cross", "torch.linalg.vecdot"])
        if r.random() < 0.15:
            op = "F.cosine_similarity"
        # cross needs equal ranks; the others also broadcast missing leading dims
        b = self.partner(prog, env, self.bcast_like(shp, keep=(d,), drop_max=0 if "cross" in op else len(shp) + d), dt)
        if r.random() < 0.5:
            a, b = b, a
        return f"{op}({a}, {b}, dim={d})"

    def square(self, prog, env, batch, n, dt, kind):
        return self.partner(prog, env, [*batch, n, n], dt, kind=kind, high=None)

    def solve_op(self, prog, env, x, t):
        """A (*a, n, n) and B (*b, n, k) with broadcasting batch dims. B is x when x is a matrix."""
        r = self.rng
        dt = self.dtcode(t)
        if t.dim() >= 2 and r.random() < 0.6:
            b, bshape = x, list(t.shape)
        else:
            bshape = [*self.batch_dims(), r.choice([1, 2, 3, 4]), r.choice([1, 2, 3])]
            b = self.partner(prog, env, bshape, dt)
        n = bshape[-2]
        u = r.choice([True, False])
        op = r.choice(["tri", "tri", "trs", "solve", "lu", "chol", "ldl", "ormqr"])
        # ormqr needs equal batches; ldl_solve with a broadcast LD batch reads out of bounds on plain CPU tensors
        # (wrong values or SIGSEGV in torch 2.14), which would kill the generator's own process
        abatch = list(bshape[:-2]) if op in ("ldl", "ormqr") else self.bcast_like(bshape[:-2])
        if op in ("tri", "trs", "ormqr"):
            a = self.square(prog, env, abatch, n, dt, None)
        else:
            a = self.square(prog, env, abatch, n, dt, "spd" if op in ("chol", "ldl") else "dd")
        if op == "tri" and r.random() < 0.3:
            return f"torch.linalg.solve_triangular({a}, {b}.mT, upper={u}, left=False, unitriangular=True)"
        return {
            "tri": f"torch.linalg.solve_triangular({a}, {b}, upper={u}, unitriangular=True)",
            "trs": f"torch.triangular_solve({b}, {a}, upper={u}, unitriangular=True)[0]",
            "solve": f"torch.linalg.solve({a}, {b})",
            "lu": f"torch.linalg.lu_solve(*torch.linalg.lu_factor({a}), {b})",
            "chol": f"torch.cholesky_solve({b}, torch.linalg.cholesky({a}))",
            "ldl": f"torch.linalg.ldl_solve(*torch.linalg.ldl_factor({a}), {b})",
            "ormqr": f"torch.ormqr(*torch.geqrf({a}), {b}, left=True, transpose={u})",
        }[op]

    def decomp_op(self, prog, env, x, t):
        """One square (batch of) matrices in, from x when x is square, else a fresh well-conditioned one."""
        r = self.rng
        dt = self.dtcode(t)
        kind = r.choice(["dd", "spd"])
        if t.dim() >= 2 and t.shape[-1] == t.shape[-2] and r.random() < 0.4:
            a, kind = x, None
        else:
            a = self.square(prog, env, self.batch_dims(), r.choice([0, 1, 2, 3, 4]), dt, kind)
        ops = [
            f"torch.linalg.det({a})",
            f"torch.linalg.slogdet({a})[1]",
            f"torch.linalg.qr({a})[1]",
            f"torch.geqrf({a})[0]",
            f"torch.linalg.householder_product(*torch.geqrf({a}))",
            f"torch.linalg.svdvals({a})",
            f"torch.linalg.matrix_norm({a})",
            f"torch.linalg.matrix_norm({a}, ord=1)",
            f"torch.linalg.lu_factor({a})[0]",
            f"torch.linalg.lu({a})[2]",
            f"torch.lu_unpack(*torch.linalg.lu_factor({a}))[1]",
            f"torch.linalg.matrix_exp({a} * 0.1)",
            f"torch.linalg.pinv({a})",
            f"{a}.diagonal(dim1=-2, dim2=-1).sum(-1)",
        ]
        if kind is not None:
            ops += [
                f"torch.linalg.inv({a})",
                f"torch.linalg.inv_ex({a})[0]",
                f"torch.linalg.solve_ex({a}, {a})[0]",
            ]
        if kind == "spd":
            ops += [
                f"torch.linalg.cholesky({a})",
                f"torch.linalg.cholesky_ex({a})[0]",
                f"torch.cholesky_inverse(torch.linalg.cholesky({a}))",
                f"torch.linalg.eigvalsh({a})",
                f"torch.linalg.ldl_factor({a})[0]",
            ]
        return r.choice(ops)

    def p_tail(self, prog, env, names, x, t):
        """Strategy-table ops no other group emits, drawn uniformly so each one is reached: torch.special functions,
        the remaining _foreach variants, fused Adam, functional batch-norm variants, prims, *_copy views and aten
        functional forms. Domains are restricted where an out-of-domain value would raise on some shards only."""
        r = self.rng
        nd = t.dim()
        dt = self.dtcode(t)
        ops = []
        if t.dtype in (torch.int64, torch.int32):
            ops += [f"{x}.__ilshift__(1)", f"{x}.__irshift__(1)"]
        if not t.is_floating_point():
            return r.choice(ops) if ops else None
        full = t.dtype in (torch.float32, torch.float64)
        pos = t.numel() > 0 and bool((t > 0).all())
        a = f"{x}.abs() + 0.5"
        ops += [
            f"torch.angle({x})",
            f"torch.ops.aten.gelu_({x})",
            f"torch.ops.aten.conj_physical_({x})",
            f"{x}.new_empty((2, 3)).fill_(1)",
            f"{x} * torch.ops.aten.is_same_size({x}, {x} + 1)",
            # signs of zeros are not kept by Partial(sum) (see p_unary), so no zero reaches signbit and copysign
            # takes its sign from a positive tensor
            f"torch.signbit(torch.where({x} == 0, 1.0, {x}))",
            f"torch.copysign({x}, {a})",
            f"{x}.copysign_({a})",
            f"torch.ops.prims.convert_element_type({x}, torch.float64)",
            f"torch.ops.prims.view_of({x})",
            f"torch.ops.prims.copy_to({x}.clone(), {x} * 3)",
            f"torch.ops.aten.native_dropout({x}, 0.0, True)[0]",
            f"torch.ops.aten.native_dropout_backward({x}, {x} == {x}, 1.0)",
            f"{x}.ldexp_({x}.abs().floor() % 3)",
            f"{x}.polygamma_(1)",
            f"torch.view_copy({x}, (-1,))",
            f"torch.ops.aten.norm({x}, 2)",
        ]
        if pos:
            ops += [f"{x}.mvlgamma_(1)", f"{x}.igamma_({a})", f"{x}.igammac_({a})"]
        if full:
            sp = "torch.special."
            ops += [
                f"{sp}airy_ai({x})",
                f"{sp}bessel_y0({a})",
                f"{sp}bessel_y1({a})",
                f"{sp}modified_bessel_i1({x})",
                f"{sp}modified_bessel_k0({a})",
                f"{sp}modified_bessel_k1({a})",
                f"{sp}scaled_modified_bessel_k0({a})",
                f"{sp}scaled_modified_bessel_k1({a})",
                f"{sp}hermite_polynomial_he({x}, {r.randint(0, 4)})",
                f"{sp}laguerre_polynomial_l({x}, {r.randint(0, 4)})",
                f"torch.igamma({a}, {x}.abs() * 0.5 + 0.1)",
                f"torch.igammac({a}, {x}.abs() * 0.5 + 0.1)",
                f"torch.view_as_real(torch.complex({x}, {x} * 2))",
                f"torch.ops.aten.linalg__powsum({x}, {r.choice([1, 2, 3])})",
                # these have strategies only as prims (torch.special decomposes to other ops)
                f"torch.ops.prims.zeta({a}, {x}.abs() + 1)",
            ]
            ops += [f"torch.ops.prims.{f}({x})" for f in "erfcx bessel_j0 bessel_j1 spherical_bessel_j0".split()]
            ops += [f"torch.ops.prims.{f}({x})" for f in "bessel_i0e bessel_i1 bessel_i1e".split()]
            ops.append(f"torch.ops.prims.ndtri(torch.sigmoid({x}))")
            ops += [f"{sp}chebyshev_polynomial_{k}({x}.clamp(-1, 1), {r.randint(0, 4)})" for k in "uvw"]
            ops += [f"{sp}shifted_chebyshev_polynomial_{k}(torch.sigmoid({x}), {r.randint(0, 4)})" for k in "tuvw"]
        if nd >= 1:
            d = r.randrange(nd)
            sz = t.shape[d]
            shp = list(t.shape)
            grown = shp[:d] + ([-1] if sz != 1 else [3]) + shp[d + 1 :]
            ops += [
                f"torch.ops.aten._safe_softmax({x}, {d})",
                f"torch.ops.aten.norm({x}, {r.choice([1, 2, 3])}, [{d}])",
                f"torch.expand_copy({x}.narrow({d}, 0, {min(sz, 1)}), {grown})",
            ]
            if sz:
                cut = r.randint(0, sz)
                ops.append(f"torch.split_with_sizes_copy({x}, [{cut}, {sz - cut}], {d})[{r.randrange(2)}]")

            # partner inputs are only made for the op that is drawn
            def bucketize():
                b = self.partner(prog, env, [r.randint(1, 4)], dt)
                return f"torch.bucketize({x}, {b}.sort()[0])"

            def index_put_impl():
                idx = self.idx_input(prog, env, [r.randint(0, 4)], t.shape[0])
                v = self.partner(prog, env, [env[idx].shape[0], *shp[1:]], dt)
                return f"torch.ops.aten._index_put_impl_({x}, [{idx}], {v}, True)"

            def scatter_add_():
                gidx = self.idx_input(prog, env, shp, sz)
                return f"{x}.scatter_add_({d}, {gidx}, {x} * 2)"

            ops.append(bucketize)
            if sz:
                ops.append(scatter_add_)
            if t.shape[0]:
                ops.append(index_put_impl)
        if full:
            same = [n for n in names if tuple(env[n].shape) == tuple(t.shape) and env[n].dtype == t.dtype and n != x]
            if same:
                y, k = r.choice(same), r.randrange(2)
                xc, yc = f"{x}.clone()", f"{y}.clone()"
                if r.random() < 0.3:
                    # every unary _foreach op, plain and in place; p_nn draws one at random among many other ops,
                    # so single variants are otherwise missed for whole campaigns
                    un = r.choice(FOREACH_UNARY)
                    return f"torch._foreach_{un}{r.choice(['', '_'])}([{x}, {y}])[{k}]"
                for op in "atan erfc lgamma log2 round sigmoid sin".split():
                    ops.append(f"torch._foreach_{op}([{x}, {y}])[{k}]")
                for op in (
                    "asin_ ceil_ cos_ cosh_ erf_ exp_ expm1_ floor_ frac_ log1p_ log_ reciprocal_ rsqrt_ sin_ "
                    "tan_ tanh_ trunc_".split()
                ):
                    ops.append(f"torch._foreach_{op}([{x}, {y}])[{k}]")
                ops += [
                    f"torch._foreach_clamp_max_([{x}, {y}], 1)[{k}]",
                    f"torch._foreach_maximum_([{x}, {y}], [{yc}, {xc}])[{k}]",
                    f"torch._foreach_minimum_([{x}, {y}], [{yc}, {xc}])[{k}]",
                    f"torch._foreach_powsum([{x}, {y}], 2)[{k}]",
                    f"torch._amp_foreach_non_finite_check_and_unscale_([{x}, {y}], {x}.new_zeros(1), "
                    f"{x}.new_full((1,), 0.5))",
                ]
                for fused in ("_fused_adam", "_fused_adamw"):
                    # params x, y; grads, exp_avg, exp_avg_sq (and max_exp_avg_sq) from fresh clones; step 1
                    ams = r.random() < 0.3
                    args = (
                        f"[{yc}, {xc}], [{xc} * 0.5, {yc} * 0.5], [{xc}.abs(), {yc}.abs()], "
                        f"[{f'{xc}.abs() + 1, {yc}.abs() + 1' if ams else ''}], "
                        f"[{x}.new_ones(()), {x}.new_ones(())], lr=0.1, beta1=0.9, beta2=0.99, "
                        f"weight_decay={r.choice([0.0, 0.01])}, eps=1e-8, amsgrad={ams}, maximize={r.random() < 0.2}"
                    )
                    ops.append(f"torch.{fused}_([{x}, {y}], {args})")
                    ops.append(f"torch.ops.aten.{fused}([{x}, {y}], {args})[{r.randrange(4)}][{k}]")
                if nd == 2:

                    def foreach_mm():
                        w = self.partner(prog, env, [t.shape[1], r.randint(1, 4)], dt)
                        return f"torch._foreach_mm([{x}, {y}], [{w}, {w}])[{k}]"

                    ops.append(foreach_mm)
        if full and 3 <= nd <= 5 and t.shape[1]:
            c = t.shape[1]
            rm, rv = f"{x}.new_zeros({c})", f"{x}.new_ones({c})"
            eps = r.choice([1e-5, 0.1])
            ops += [
                f"torch.ops.aten._native_batch_norm_legit_no_training({x}, None, None, {rm}, {rv}, 0.1, {eps})[0]",
                f"torch.ops.aten._native_batch_norm_legit({x}, None, None, {rm}, {rv}, True, 0.1, {eps})[0]",
                f"torch.ops.aten._native_batch_norm_legit_functional({x}, None, None, {rm}, {rv}, True, 0.1, {eps})[0]",
                f"torch.ops.aten._batch_norm_with_update({x}, None, None, {rm}, {rv}, 0.1, {eps})[0]",
                f"torch.ops.aten._batch_norm_with_update_functional({x}, None, None, {rm}, {rv}, 0.1, {eps})[0]",
            ]
        if full and nd in (3, 4, 5) and t.shape[1]:
            # 3-D ops take x lifted to (N, C, D, H, W); 2-D ones take 4-D inputs as they are
            v = f"{x}{'[None]' * (5 - nd)}"
            n, c, *sp = [1] * (5 - nd) + list(t.shape)
            # fractional pooling needs output + kernel - 1 < input in every dim, so no dim of size 0 or 1
            if all(s >= 2 for s in sp):
                k3 = [1 if s == 2 else 2 for s in sp]
                out3 = [s - k for s, k in zip(sp, k3)]
                ops.append(
                    f"torch.ops.aten.fractional_max_pool3d({v}, {k3}, {out3}, {v}.new_full(({n}, {c}, 3), 0.5))[0]"
                )
            if all(sp):
                kp = [2 if s % 2 == 0 else 1 for s in sp]
                ops.append(f"F.max_unpool3d(*F.max_pool3d({v}, {kp}, return_indices=True), {kp})")
            if nd == 4:
                if all(s >= 2 for s in t.shape[2:]):
                    k2 = [1 if s == 2 else 2 for s in t.shape[2:]]
                    out2 = [s - k for s, k in zip(t.shape[2:], k2)]
                    ops.append(
                        f"torch.ops.aten.fractional_max_pool2d({x}, {k2}, {out2}, "
                        f"{x}.new_full(({t.shape[0]}, {t.shape[1]}, 2), 0.5))[0]"
                    )
                ops.append(f"torch.ops.aten._upsample_lanczos2d_aa({x}, [{r.randint(1, 5)}, {r.randint(1, 5)}], False)")

            def grid_sample_3d():
                g = self.partner(prog, env, [n, 2, 2, 2, 3], dt)
                return f"F.grid_sample({v}, torch.tanh({g}), align_corners=False)"

            ops.append(grid_sample_3d)
        if not ops:
            return None
        op = r.choice(ops)
        return op() if callable(op) else op

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
