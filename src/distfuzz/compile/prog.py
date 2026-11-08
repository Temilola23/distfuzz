"""Compile-track programs.

An op program is a DTensor generator program (``distfuzz.dtensor.gen``) plus a ``cfg`` that says how to compile it.
The worker turns the program's steps into one Python function: every ``mk(j)`` step becomes an argument, every
other step a line of the body, and the function returns the step outputs. The same source runs as the plain-tensor
reference, as eager DTensor and under ``torch.compile`` (plain and DTensor), so a difference between the legs points
at the compile path.
"""

import copy
import re

from distfuzz.dtensor.gen import MUT_RE, NAME_RE, Gen, parse_R

BACKENDS = ("aot_eager", "inductor")
DYNAMIC = ("static", "dynamic", "mark")
MK_STEP_RE = re.compile(r"^mk\((\d+)\)$")


def pl_code(p):
    if p == "R":
        return "Replicate()"
    if p.startswith("S"):
        return f"Shard({int(p[1:])})"
    if p.startswith("P:"):
        return f"Partial({p[2:]!r})"
    raise ValueError(p)


def compile_steps(prog):
    """Steps that go into the compiled function, and the BW targets that run after it.

    The function stops at the first BW step: backward runs outside the compiled region (as in training), on the
    returned output. Steps after the first BW are dropped.
    """
    body, bws = [], []
    for st in prog["steps"]:
        if st["expr"].startswith("BW("):
            bws.append(st["expr"][3:-1])
            break
        body.append(st)
    return body, bws


def outputs(body, bws, mode):
    """Names the function returns. "all": every step output. "leaves": outputs no later step reads, plus BW targets
    and inputs (mutated inputs are compared through the returned argument)."""
    names = [st["out"] for st in body]
    if mode == "all":
        return names
    used = set()
    for st in body:
        if not MK_STEP_RE.match(st["expr"]):
            used |= set(NAME_RE.findall(st["expr"]))
    keep = []
    for st in body:
        n = st["out"]
        if n not in used or n in bws or MK_STEP_RE.match(st["expr"]) or MUT_RE.search(st["expr"]):
            keep.append(n)
    return keep


def build_source(prog):
    """Source of ``def prog_fn(...)`` and the argument list: [(arg_name, input_index)].

    ``R(v, [...])`` becomes ``R(v, (Shard(0), ...))`` so placements are constants the tracer sees directly.
    """
    body, bws = compile_steps(prog)
    args, lines = [], []
    for st in body:
        m = MK_STEP_RE.match(st["expr"])
        if m:
            args.append((st["out"], int(m.group(1))))
            continue
        expr = st["expr"]
        pr = parse_R(expr)
        if pr:
            pls = ", ".join(pl_code(p) for p in pr[1])
            expr = f"R({pr[0]}, ({pls},))"
        lines.append(f"    {st['out']} = {expr}")
    outs = outputs(body, bws, prog["cfg"].get("ret", "all"))
    # outputs that backward does not start from are detached: AOTAutograd guesses a DTensor tangent for every
    # output that requires grad and fails when autograd hands it none (documented limitation, pytorch#172556)
    items = [n if not bws or n in bws else f"_d({n})" for n in outs]
    ret = ", ".join(items) + ("," if len(items) == 1 else "")
    src = "def prog_fn(" + ", ".join(a for a, _ in args) + "):\n"
    src += "\n".join(lines) + ("\n" if lines else "")
    src += f"    return ({ret})\n"
    return src, args, outs, bws


def rand_cfg(rng, backends=BACKENDS):
    return dict(
        backend=rng.choice(list(backends)),
        fullgraph=rng.random() < 0.5,
        dynamic=rng.choice(list(DYNAMIC)),
        ret=rng.choice(["all", "all", "leaves"]),
        variant=None,
    )


def variant_inputs(prog, rng, world):
    """A second input list for dynamic configs: one size used by a sharded dim is mapped to another size in every
    dim of every input that has it, so shapes stay consistent across operands. The compiled function is called on
    it without being rebuilt. Returns None when no input has a sharded dim of size >= 2."""
    sizes = []
    for spec in prog["inputs"]:
        if spec.get("kind") == "index":
            continue
        for p in spec["pl"]:
            if p.startswith("S"):
                d = int(p[1:])
                if d < len(spec["shape"]) and spec["shape"][d] >= 2:
                    sizes.append(spec["shape"][d])
    if not sizes:
        return None
    s = rng.choice(sizes)
    cands = sorted({s + 1, s + 2, s - 1, 2 * world + 1, world - 1, world + 1} - {s, 0, 1})
    if not cands:
        return None
    s2 = rng.choice(cands)
    out = copy.deepcopy(prog["inputs"])
    for spec in out:
        spec["shape"] = [s2 if x == s else x for x in spec["shape"]]
        if spec.get("kind") == "index" and spec.get("high") == s:
            spec["high"] = s2
    return dict(old=s, new=s2, inputs=out)


class CompileGen:
    """Wraps the DTensor generator: same programs (no nested CMP steps), plus a compile config."""

    def __init__(self, world, rng, backends=BACKENDS, module_rate=0.0):
        self.world, self.rng = world, rng
        self.backends = backends
        self.module_rate = module_rate
        self.gen = Gen(world, rng, allow_compile=False)

    def finish(self, prog):
        prog.setdefault("cfg", rand_cfg(self.rng, self.backends))
        prog["track"] = "op"
        if prog["cfg"]["dynamic"] != "static":
            prog["cfg"]["variant"] = variant_inputs(prog, self.rng, self.world)
        else:
            prog["cfg"]["variant"] = None
        return prog

    def generate(self):
        if self.module_rate and self.rng.random() < self.module_rate:
            from distfuzz.compile.module import rand_module_prog

            return rand_module_prog(self.rng, self.world, self.backends)
        p = self.gen.generate()
        # keep compile programs short: compile time grows with graph size, and the bugs here need few ops
        p["steps"] = p["steps"][:14]
        self.gen.repair(p)
        return self.finish(p)

    def mutate(self, prog):
        if prog.get("track") == "module":
            from distfuzz.compile.module import mutate_module_prog

            return mutate_module_prog(prog, self.rng, self.world, self.backends)
        r = self.rng
        if r.random() < 0.3:
            p = copy.deepcopy(prog)
            k = r.choice(["backend", "fullgraph", "dynamic", "ret"])
            c = p["cfg"]
            if k == "backend":
                c["backend"] = r.choice(list(self.backends))
            elif k == "fullgraph":
                c["fullgraph"] = not c["fullgraph"]
            elif k == "dynamic":
                c["dynamic"] = r.choice(list(DYNAMIC))
            else:
                c["ret"] = "leaves" if c.get("ret") == "all" else "all"
            p["mut"] = "cfg:" + k
            return self.finish(p)
        cfg = copy.deepcopy(prog["cfg"])
        p = self.gen.mutate({k: v for k, v in prog.items() if k not in ("cfg", "track")})
        p["steps"] = p["steps"][:14]
        p["cfg"] = cfg
        return self.finish(p)
