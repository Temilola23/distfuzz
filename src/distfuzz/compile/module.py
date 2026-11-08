"""Module track: small FSDP2 / TP / FSDP2+TP models trained for a few steps, eager vs compiled vs single process.

The parallelization follows torchtitan's order: tensor parallel first, then ``torch.compile`` on each block
("per_module", torchtitan's layout) or on the whole model after sharding ("whole"), then ``fully_shard`` on each block
and the root. Every rank feeds the same batch, so data-parallel gradient averaging leaves the gradient unchanged and
the single-process model trained on that batch is the reference.
"""

import copy
import re
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

from distfuzz.dtensor.runtime import compare

DIMS = [1, 2, 3, 4, 5, 6, 8, 9]


def rand_module_prog(rng, world, backends):
    par = rng.choice(["fsdp", "tp", "fsdp_tp"] if world == 4 else ["fsdp", "tp"])
    kind = rng.choice(["mlp", "emb"])
    vary = rng.random() < 0.4
    b0 = rng.randint(1, 5)
    steps = rng.randint(2, 4)
    prog = dict(
        track="module",
        mesh="2d" if par == "fsdp_tp" else "1d",
        model=dict(
            kind=kind,
            I=rng.choice(DIMS),
            D=rng.choice(DIMS),
            H=rng.choice(DIMS),
            O=rng.choice(DIMS),
            V=rng.choice([3, 5, 7, 8, 9, 12]),
            S=rng.choice([1, 2, 3, 4, 5, 6]),
            blocks=rng.choice([1, 2]),
            norm=rng.choice(["ln", "rms", "none"]),
            act=rng.choice(["relu", "gelu", "silu"]),
            bias=rng.random() < 0.7,
        ),
        cfg=dict(
            par=par,
            sp=par != "fsdp" and rng.random() < 0.4,
            wrap=rng.choice(["per_module", "whole"]),
            backend=rng.choice(list(backends)),
            fullgraph=rng.random() < 0.5,
            dynamic=rng.choice(["static", "dynamic", "mark"]),
            raf=rng.random() < 0.7,
            opt=rng.choice(["sgd", "adam"]),
            batches=[rng.randint(1, 5) if vary else b0 for _ in range(steps)],
            seed=rng.randrange(1 << 20),
        ),
        inputs=[],
        steps=[],
    )
    return legalize(prog, rng, world)


def tp_degree(prog, world):
    return {"fsdp": 1, "tp": world, "fsdp_tp": 2}[prog["cfg"]["par"]]


def no_empty_shard(n, k):
    # torch.chunk semantics: chunks of ceil(n / k); the last rank is empty when the first k - 1 chunks cover n
    return n >= k and -(-n // k) * (k - 1) < n


def legalize(prog, rng, world):
    """Dims that tensor parallel shards get at least one row per rank. An empty TP shard already fails in eager
    mode (VERIFIED.md section 9 family) and a rank-asymmetric eager failure costs a full collective timeout.
    Uneven shards stay possible."""
    k = tp_degree(prog, world)
    m = prog["model"]
    if k > 1:
        ok = [x for x in DIMS + [7, 10, 11, 12] if no_empty_shard(x, k)]
        # uneven TP shards fail in eager mode already: the TP styles call DTensor.from_local without a shape, which
        # assumes even shards (mm shape errors). Uneven shards stay in the FSDP2 dim-0 shards and in the op track.
        ok = [x for x in ok if x % k == 0]
        for d in ("D", "H", "O", "V"):
            if m[d] not in ok:
                m[d] = rng.choice(ok)
        if prog["cfg"]["sp"] and m["S"] not in ok:
            m["S"] = rng.choice([s for s in ok if s <= 12])
    else:
        prog["cfg"]["sp"] = False
    if prog["cfg"]["par"] != "tp" and prog["cfg"]["wrap"] == "whole":
        # FSDP2's hooks are dynamo-disabled, so a whole-model compile cannot be fullgraph (torchtitan compiles blocks)
        prog["cfg"]["fullgraph"] = False
    return prog


def mutate_module_prog(prog, rng, world, backends):
    p = copy.deepcopy(prog)
    other = rand_module_prog(rng, world, backends)
    other["cfg"]["fullgraph"] = rng.random() < 0.5
    k = rng.choice(["model", "par", "wrap", "backend", "fullgraph", "dynamic", "raf", "opt", "batches", "sp", "dim"])
    if k == "model":
        p["model"] = other["model"]
    elif k == "par":
        p["cfg"]["par"], p["mesh"], p["cfg"]["sp"] = other["cfg"]["par"], other["mesh"], other["cfg"]["sp"]
    elif k == "dim":
        d = rng.choice(["I", "D", "H", "O", "V", "S"])
        p["model"][d] = other["model"][d]
    elif k in p["cfg"]:
        p["cfg"][k] = other["cfg"][k]
    p["mut"] = "module:" + k
    return legalize(p, rng, world)


class RMS(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * self.weight


class Block(nn.Module):
    def __init__(self, m):
        super().__init__()
        D, H = m["D"], m["H"]
        self.norm = {"ln": lambda: nn.LayerNorm(D), "rms": lambda: RMS(D), "none": nn.Identity}[m["norm"]]()
        self.fc1 = nn.Linear(D, H, bias=m["bias"])
        self.fc2 = nn.Linear(H, D, bias=m["bias"])
        self.act = m["act"]

    def forward(self, x):
        h = self.fc1(self.norm(x))
        h = {"relu": F.relu, "gelu": F.gelu, "silu": F.silu}[self.act](h)
        return x + self.fc2(h)


class Net(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.inp = nn.Embedding(m["V"], m["D"]) if m["kind"] == "emb" else nn.Linear(m["I"], m["D"], bias=m["bias"])
        self.layers = nn.ModuleDict({str(i): Block(m) for i in range(m["blocks"])})
        self.head = nn.Linear(m["D"], m["O"], bias=m["bias"])

    def forward(self, x):
        h = self.inp(x)
        for b in self.layers.values():
            h = b(h)
        return self.head(h)


def build(prog):
    torch.manual_seed(prog["cfg"]["seed"])
    return Net(prog["model"])


def batch(prog, step):
    m, c = prog["model"], prog["cfg"]
    g = torch.Generator().manual_seed(c["seed"] * 31 + step)
    B, S = c["batches"][step], m["S"]
    if m["kind"] == "emb":
        x = torch.randint(0, m["V"], (B, S), generator=g)
    else:
        x = torch.randint(-3, 4, (B, S, m["I"]), generator=g).float()
    y = torch.randint(-3, 4, (B, S, m["O"]), generator=g).float()
    return x, y


def tp_plan(prog):
    from torch.distributed.tensor import Replicate, Shard
    from torch.distributed.tensor.parallel import ColwiseParallel, RowwiseParallel, SequenceParallel

    sp = prog["cfg"]["sp"]
    act = Shard(1) if sp else Replicate()
    plan = {}
    if prog["model"]["kind"] == "emb":
        plan["inp"] = RowwiseParallel(input_layouts=Replicate(), output_layouts=act)
    else:
        plan["inp"] = ColwiseParallel(output_layouts=act)
    for i in range(prog["model"]["blocks"]):
        if sp and prog["model"]["norm"] != "none":
            plan[f"layers.{i}.norm"] = SequenceParallel()
        plan[f"layers.{i}.fc1"] = ColwiseParallel(input_layouts=act)
        plan[f"layers.{i}.fc2"] = RowwiseParallel(output_layouts=act)
    plan["head"] = ColwiseParallel(input_layouts=act, output_layouts=Replicate())
    return plan


def compile_kw(cfg):
    return dict(
        backend=cfg["backend"],
        fullgraph=cfg["fullgraph"],
        dynamic={"static": False, "dynamic": True, "mark": None}[cfg["dynamic"]],
    )


def parallelize(prog, model, meshes, compiled):
    from torch.distributed.fsdp import fully_shard
    from torch.distributed.tensor.parallel import parallelize_module

    c = prog["cfg"]
    tp_mesh = dp_mesh = None
    if c["par"] == "tp":
        tp_mesh = meshes["1d"]
    elif c["par"] == "fsdp":
        dp_mesh = meshes["1d"]
    else:
        m2 = meshes["2d"]
        dp_mesh, tp_mesh = m2["dp"], m2["tp"]
    if tp_mesh is not None:
        parallelize_module(model, tp_mesh, tp_plan(prog))
    if compiled and c["wrap"] == "per_module":
        for k in list(model.layers.keys()):
            model.layers[k] = torch.compile(model.layers[k], **compile_kw(c))
    if dp_mesh is not None:
        for k in list(model.layers.keys()):
            fully_shard(model.layers[k], mesh=dp_mesh, reshard_after_forward=c["raf"])
        fully_shard(model, mesh=dp_mesh, reshard_after_forward=c["raf"])
    if compiled and c["wrap"] == "whole":
        model = torch.compile(model, **compile_kw(c))
    return model


def make_opt(prog, params):
    if prog["cfg"]["opt"] == "adam":
        return torch.optim.Adam(params, lr=0.01)
    return torch.optim.SGD(params, lr=0.05)


def canon(name):
    return re.sub(r"(^|\.)_orig_mod\.", r"\1", name).replace("_fsdp_wrapped_module.", "")


def full_params(model):
    from torch.distributed.tensor import DTensor

    out = {}
    for n, p in model.named_parameters():
        out[canon(n)] = (p.full_tensor() if isinstance(p, DTensor) else p).detach().clone()
    return out


def full_val(t):
    from torch.distributed.tensor import DTensor

    return t.full_tensor() if isinstance(t, DTensor) else t


def train(prog, model, opt, mark_dyn, losses, graphs_fn=None, graph_log=None):
    for step in range(len(prog["cfg"]["batches"])):
        x, y = batch(prog, step)
        if mark_dyn:
            torch._dynamo.maybe_mark_dynamic(x, 0)
        g0 = graphs_fn() if graphs_fn else 0
        out = model(x)
        loss = F.mse_loss(full_val(out), y)
        loss.backward()
        opt.step()
        opt.zero_grad()
        losses.append(float(loss.detach()))
        if graph_log is not None and graphs_fn:
            graph_log.append(graphs_fn() - g0)


def run_module(prog, meshes, findings, sync_flag, set_leg, cmsg, err_op, Abort):
    from torch._dynamo.utils import counters

    def graphs():
        return int(counters["stats"]["unique_graphs"])

    c = prog["cfg"]
    info: dict = dict(graphs=0, graph_breaks=0, recompiles=0, nondist=0)
    base = build(prog)
    mark_dyn = c["dynamic"] == "mark"

    def add(kind, leg, where, msg, **kw):
        findings.append(dict(kind=kind, leg=leg, expr=where, msg=msg, **kw))

    set_leg("mod_ref")
    ref = copy.deepcopy(base)
    rl: list = []
    try:
        train(prog, ref, make_opt(prog, ref.parameters()), False, rl)
    except Exception as e:
        info["skip"] = "ref:" + cmsg(e)
        return [], info
    rp = full_params(ref)

    def dist_leg(leg, compiled):
        set_leg(leg)
        torch._dynamo.reset()
        counters.clear()
        losses, glog = [], []
        err, model = None, None
        try:
            model = parallelize(prog, copy.deepcopy(base), meshes, compiled)
            train(prog, model, make_opt(prog, model.parameters()), mark_dyn and compiled, losses, graphs, glog)
        except Exception as e:
            err = e
            print(f"[{leg}] step {len(losses)}: {cmsg(e)}", file=sys.stderr, flush=True)
        flags = sync_flag(err is not None)
        if any(flags) and not all(flags):
            add(
                "RANK_DIVERGENT",
                leg,
                f"step {len(losses)}",
                cmsg(err) if err else "other rank failed",
                op=err_op(err or ""),
            )
            raise Abort()
        params = None
        if err is None:
            try:
                params = full_params(model)
            except Exception as e:
                err = e
            flags = sync_flag(err is not None)
            if any(flags) and not all(flags):
                add("RANK_DIVERGENT", leg, "full_params", cmsg(err) if err else "other rank failed")
                raise Abort()
        return losses, params, err, glog

    el, ep, eerr, _ = dist_leg("mod_eager", False)
    if eerr is not None:
        info["skip"] = "eager:" + cmsg(eerr)
        return [], info
    eager_bad = [k for k, (a, b) in enumerate(zip(rl, el)) if not _close(a, b)]
    eager_pbad = [n for n in rp if n in ep and compare(rp[n], ep[n]) is not None]
    if eager_bad or eager_pbad:
        where = f"step {eager_bad[0]}" if eager_bad else eager_pbad[0]
        add("EAGER_MISMATCH", "mod_eager", where, f"ref={rl} eager={el}")
        return [], info

    # control: the single-process model compiled the same way (whole model, or each block)
    set_leg("mod_cplain")
    torch._dynamo.reset()
    cpl: list = []
    cpp = None
    try:
        m = copy.deepcopy(base)
        if c["wrap"] == "per_module":
            for k in list(m.layers.keys()):
                m.layers[k] = torch.compile(m.layers[k], **compile_kw(c))
        else:
            m = torch.compile(m, **compile_kw(c))
        train(prog, m, make_opt(prog, m.parameters()), mark_dyn, cpl)
        cpp = full_params(m)
    except Exception as e:
        info["cplain_err"] = cmsg(e)

    cl, cp, cerr, glog = dist_leg("mod_cdist", True)
    info["graphs"] = graphs()
    info["graph_breaks"] = int(sum(counters["graph_break"].values()))
    info["graph_log"] = glog
    if cerr is not None:
        kind = "FULLGRAPH_BREAK" if type(cerr).__name__ == "Unsupported" or "Unsupported" in cmsg(cerr) else "ERR"
        if "cplain_err" in info and info["cplain_err"] == cmsg(cerr):
            info["nondist"] += 1
            info["skip"] = "nondist_err"
            return [], info
        add(kind, "mod_cdist", f"step {len(cl)}", cmsg(cerr), op=err_op(cerr))
        return [], info
    for k, (a, b) in enumerate(zip(rl, cl)):
        if not _close(a, b):
            if k < len(cpl) and _close(cpl[k], b):
                info["nondist"] += 1
                continue
            add("LOSS_MISMATCH", "mod_cdist", f"step {k}", f"ref={a:.6g} eager={el[k]:.6g} compiled={b:.6g}")
            break
    for n in rp:
        if n not in cp:
            add("PARAM_MISSING", "mod_cdist", n, f"have {sorted(cp)[:4]}")
            break
        d = compare(rp[n], cp[n])
        if d is not None:
            if cpp is not None and n in cpp and compare(cpp[n], cp[n]) is None:
                info["nondist"] += 1
                continue
            add("PARAM_MISMATCH", "mod_cdist", re.sub(r"\d+", "N", n), d)
            break
    # recompiles at a step whose batch shape was already seen
    seen = set()
    for k, b in enumerate(c["batches"]):
        if k >= 2 and b in seen and k < len(glog) and glog[k] > 0:
            info["recompiles"] += glog[k]
        seen.add(b)
    if info["recompiles"]:
        add("RECOMPILE", "mod_cdist", "", f"{info['recompiles']} new graphs at repeated batch shapes {glog}")
    if info["recompiles"] >= 8:
        add("RECOMPILE_STORM", "mod_cdist", "", f"{info['recompiles']} recompiles at repeated shapes")
    return [], info


def _close(a, b):
    if a != a and b != b:
        return True
    return abs(a - b) <= 1e-4 * max(1.0, abs(a))
