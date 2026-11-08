"""Rank process for the compile fuzzer.

Reads one program per line on stdin, runs it on this rank and writes one JSON result line. Every rank runs the same
legs in the same order, so collectives inside DTensor ops and the control-plane ``sync_flag`` calls line up.
The current leg is written to ``<legdir>/rank<r>.leg`` before it starts, so a hang can be attributed to a leg.
"""

import argparse
import datetime
import hashlib
import json
import math
import os
import re
import sys
import time
import traceback
import warnings

warnings.filterwarnings("ignore")

ap = argparse.ArgumentParser()
ap.add_argument("--rank", type=int)
ap.add_argument("--world", type=int)
ap.add_argument("--port", type=int)
ap.add_argument("--legdir", default="")
ap.add_argument("--pg-timeout", type=float, default=180)
args = ap.parse_args()

proto = os.fdopen(os.dup(1), "w", buffering=1)
os.dup2(2, 1)

import torch  # noqa: E402
import torch._dynamo  # noqa: E402
import torch.distributed as dist  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch._dynamo.utils import counters  # noqa: E402
from torch.distributed.device_mesh import init_device_mesh  # noqa: E402
from torch.distributed.tensor import DTensor, Partial, Replicate, Shard  # noqa: E402
from torch.distributed.tensor._utils import compute_local_shape_and_global_offset  # noqa: E402

from distfuzz.compile.prog import build_source  # noqa: E402
from distfuzz.dtensor.runtime import Ctx, compare, make_input  # noqa: E402

torch.set_num_threads(1)
torch._dynamo.config.cache_size_limit = 8
torch._dynamo.config.accumulated_cache_size_limit = 256

dist.init_process_group(
    "gloo",
    init_method=f"tcp://127.0.0.1:{args.port}",
    rank=args.rank,
    world_size=args.world,
    # compile time differs per rank (uneven shards give different local graphs, cache hits differ), so a rank can
    # wait in a collective while another still compiles. Hangs are caught by the driver's program timeout.
    timeout=datetime.timedelta(seconds=args.pg_timeout),
)
W = args.world
SYNC_PG = dist.new_group(backend="gloo", timeout=datetime.timedelta(seconds=args.pg_timeout))
MESHES = {"1d": init_device_mesh("cpu", (W,))}
if W == 4:
    MESHES["2d"] = init_device_mesh("cpu", (2, 2), mesh_dim_names=("dp", "tp"))
if W == 6:
    MESHES["2d"] = init_device_mesh("cpu", (2, 3), mesh_dim_names=("dp", "tp"))

LEGFILE = os.path.join(args.legdir, f"rank{args.rank}.leg") if args.legdir else ""


class Abort(Exception):
    pass


CUR = {"leg": "idle", "timeout": args.pg_timeout}
GROUPS = [None, SYNC_PG] + [m.get_group(d) for m in MESHES.values() for d in range(m.ndim)]
EAGER_TIMEOUT = 30.0


def set_timeout(sec):
    # eager legs fail fast when one rank raises and the others wait in a collective; compiled legs need room for
    # ranks that compile at different speeds
    if CUR["timeout"] == sec:
        return
    for g in GROUPS:
        try:
            dist.set_timeout(datetime.timedelta(seconds=sec), g)
        except Exception:
            pass
    CUR["timeout"] = sec


def set_leg(name):
    CUR["leg"] = name
    set_timeout(args.pg_timeout if ("cdist" in name or "cplain" in name or name == "again") else EAGER_TIMEOUT)
    if LEGFILE:
        with open(LEGFILE, "w") as f:
            f.write(name)


def norm_msg(e):
    s = (str(e).strip().splitlines() or [""])[0][:200]
    s = re.sub(r"0x[0-9a-f]+", "ADDR", s)
    s = re.sub(r"\d+", "N", s)
    return f"{type(e).__name__}: {s}"


_SKIP_LINE = re.compile(r"^(backend=.*raised:|Set TORCHDYNAMO_VERBOSE|$|\s*Explanation:|\s*Hint:|\s*Developer debug)")


def root_exc(e):
    seen = 0
    while seen < 8:
        nxt = getattr(e, "inner_exception", None) or e.__cause__
        if nxt is None or nxt is e:
            break
        e, seen = nxt, seen + 1
    return e


def cmsg(e):
    """Compile errors wrap the real error several times; name the outer type and the innermost error's first line."""
    if not isinstance(e, BaseException):
        return str(e)
    inner = root_exc(e)
    lines = [ln for ln in str(inner).splitlines() if not _SKIP_LINE.match(ln)]
    s = (lines[0] if lines else "").strip()[:160]
    s = re.sub(r"0x[0-9a-f]+", "ADDR", s)
    s = re.sub(r"\d+", "N", s)
    outer = type(e).__name__
    return f"{outer}<-{type(inner).__name__}: {s}" if inner is not e else f"{outer}: {s}"


PROG: dict = {"lines": []}
_PROG_LINE = re.compile(r'File "<distfuzz-compile-prog>", line (\d+)')
_MOD_LINE = re.compile(r'File "[^"]*distfuzz/compile/module\.py", line \d+, in forward\n\s+(.+)')


def _chain(e):
    out = []
    while isinstance(e, BaseException) and e not in out and len(out) < 8:
        out.append(e)
        e = getattr(e, "inner_exception", None) or e.__cause__ or e.__context__
    return out


def err_op(e):
    """The op an error comes from: the program line named in the traceback or in dynamo's "from user code" block,
    else an aten op or FX node named in the message."""
    if not isinstance(e, BaseException):
        return "?"
    from distfuzz.dtensor.fuzz import opname

    chain = _chain(e)
    text = "\n".join(str(x) for x in chain)
    tb = "\n".join("".join(traceback.format_exception(x)) for x in chain)
    nums = [int(n) for n in _PROG_LINE.findall(text + "\n" + tb)]
    lines = PROG["lines"]
    if nums and 0 < nums[-1] <= len(lines):
        ln = lines[nums[-1] - 1]
        if " = " in ln:
            return opname(re.sub(r"\bv\d+\b", "v0", ln.split(" = ", 1)[1]))
    m = _MOD_LINE.findall(text + "\n" + tb)
    if m:
        return re.sub(r"\s+", "", m[-1])[:40]
    for pat in (
        r"aten\.(\w+)",
        r"call_function <(?:function|built-in function|built-in method|method '?)(\w+)",
        r"call_method (\w+)",
    ):
        m = re.search(pat, text)
        if m:
            return m.group(1)
    return "?"


def is_unsupported(e):
    import torch._dynamo.exc as dexc

    return isinstance(e, dexc.Unsupported)


def sync_flag(flag):
    t = torch.tensor([1 if flag else 0], dtype=torch.int64)
    out = [torch.zeros(1, dtype=torch.int64) for _ in range(W)]
    dist.all_gather(out, t, group=SYNC_PG)
    return [bool(o.item()) for o in out]


def log_err(leg, e):
    print(f"[{leg}] {type(e).__name__}: {str(e)[:300]}", file=sys.stderr, flush=True)
    if os.environ.get("DTF_TB"):
        traceback.print_exc()


def thash(t):
    t = torch.empty(t.numel(), dtype=t.dtype).copy_(t.detach().reshape(-1))
    if t.dtype == torch.bool:
        t = t.to(torch.uint8)
    return hashlib.md5(t.view(torch.uint8).numpy().tobytes()).hexdigest()[:12]


def R_ref(x, pls):
    return x.clone()


def R_dist(x, pls):
    return x.redistribute(x.device_mesh, pls).clone()


def SETITEM(x, idx, y):
    x[idx] = y
    return x


def _d(x):
    return x.detach() if isinstance(x, torch.Tensor) else x


BASE_G = {
    "_d": _d,
    "torch": torch,
    "F": F,
    "math": math,
    "Shard": Shard,
    "Replicate": Replicate,
    "Partial": Partial,
    "SETITEM": SETITEM,
}


def mkfn(src, distributed):
    g = dict(BASE_G, R=R_dist if distributed else R_ref)
    exec(compile(src, "<distfuzz-compile-prog>", "exec"), g)
    return g["prog_fn"]


def full_all(vals, findings, leg, exprs):
    """full_tensor() of every DTensor in vals; a rank-asymmetric failure is a finding and aborts the program."""
    out = []
    for i, v in enumerate(vals):
        if not isinstance(v, torch.Tensor):
            out.append(v)
            continue
        f, err = None, None
        try:
            f = v.full_tensor() if isinstance(v, DTensor) else v
        except Exception as e:
            err = e
        flags = sync_flag(err is not None)
        if any(flags) and not all(flags):
            findings.append(
                dict(
                    kind="RANK_DIVERGENT",
                    leg=leg,
                    expr=exprs[i] + " [full_tensor]",
                    msg=norm_msg(err) if err else "other rank failed",
                )
            )
            raise Abort()
        out.append(err if err is not None else f)
    return out


def grads_of(ins):
    return [x.grad if isinstance(x, torch.Tensor) and x.requires_grad else None for x in ins]


def dyn_arg(cfg):
    return {"static": False, "dynamic": True, "mark": None}[cfg["dynamic"]]


def mark(cfg, ins, specs_of):
    if cfg["dynamic"] != "mark":
        return
    for x, spec in zip(ins, specs_of):
        if not isinstance(x, torch.Tensor):
            continue
        for p in spec["pl"]:
            if p.startswith("S"):
                d = int(p[1:])
                if d < x.dim():
                    torch._dynamo.maybe_mark_dynamic(x, d)


def graphs():
    return int(counters["stats"]["unique_graphs"])


def run_op(prog, findings):
    cfg = prog["cfg"]
    mesh = MESHES[prog["mesh"]]
    src, fargs, outs, bws = build_source(prog)
    PROG["lines"] = src.splitlines()
    exprs = {st["out"]: st["expr"] for st in prog["steps"]}
    out_exprs = [exprs[n] for n in outs]
    oi = {n: i for i, n in enumerate(outs)}
    hashes: list = []
    info = dict(graphs=0, graph_breaks=0, recompiles=0, variant_graphs=None, nondist=0)
    specs = prog["inputs"]
    arg_specs = [specs[j] for _, j in fargs]

    def inputs(distributed, sp):
        ctx = Ctx("dist" if distributed else "ref", mesh if distributed else None, sp)
        return [make_input(ctx, j) for _, j in fargs]

    def call(fn, ins):
        o = fn(*ins)
        for t in bws:
            if t in oi and isinstance(o[oi[t]], torch.Tensor) and o[oi[t]].requires_grad:
                o[oi[t]].sum().backward()
        return list(o)

    def add(kind, leg, i, msg, **kw):
        findings.append(dict(kind=kind, leg=leg, expr=out_exprs[i] if i is not None else "", msg=msg, **kw))

    torch.manual_seed(0)
    # leg 1: single-process reference (identical on every rank, so no sync needed to skip)
    set_leg("ref")
    try:
        rin = inputs(False, specs)
        rout = call(mkfn(src, False), rin)
    except Exception as e:
        info["skip"] = "ref:" + norm_msg(e)
        return hashes, info
    rgr = grads_of(rin)

    # leg 2: eager DTensor
    set_leg("eager")
    derr = None
    try:
        din = inputs(True, specs)
        dout = call(mkfn(src, True), din)
    except Exception as e:
        derr = e
        log_err("eager", e)
    flags = sync_flag(derr is not None)
    if any(flags) and not all(flags):
        add("EAGER_RANK_DIVERGENT", "eager", None, norm_msg(derr) if derr else "other rank failed")
        raise Abort()
    if derr is not None:
        info["skip"] = "eager:" + norm_msg(derr)
        return hashes, info
    efull = full_all(dout, findings, "eager", out_exprs)
    eager_ok = [
        isinstance(r, torch.Tensor) and isinstance(e, torch.Tensor) and compare(r, e) is None
        for r, e in zip(rout, efull)
    ]
    egr = full_all(grads_of(din), findings, "eager", [f"grad{k}" for k in range(len(din))])
    eager_gok = [
        isinstance(r, torch.Tensor) and isinstance(e, torch.Tensor) and compare(r, e) is None for r, e in zip(rgr, egr)
    ]
    info["eager_mismatch"] = sum(
        1 for ok, r in zip(eager_ok, rout) if not ok and isinstance(r, torch.Tensor) and r.numel()
    )

    kw = dict(backend=cfg["backend"], fullgraph=cfg["fullgraph"], dynamic=dyn_arg(cfg))
    # leg 3: compiled plain tensors (control: a difference here is not about DTensor)
    set_leg("cplain")
    torch._dynamo.reset()
    counters.clear()
    cperr, cpout, cp = None, None, None
    try:
        cp = torch.compile(mkfn(src, False), **kw)
        cpin = inputs(False, specs)
        mark(cfg, cpin, arg_specs)
        cpout = call(cp, cpin)
    except Exception as e:
        cperr = e
    cpgr = grads_of(cpin) if cperr is None else None

    # leg 4: compiled DTensor
    set_leg("cdist")
    counters.clear()
    cderr, stage = None, "fwd"
    cd = torch.compile(mkfn(src, True), **kw)
    try:
        cdin = inputs(True, specs)
        mark(cfg, cdin, arg_specs)
        o = cd(*cdin)
        stage = "bw"
        for t in bws:
            if t in oi and isinstance(o[oi[t]], torch.Tensor) and o[oi[t]].requires_grad:
                o[oi[t]].sum().backward()
        cdout = list(o)
    except Exception as e:
        cderr = e
        log_err("cdist", e)
    info["graphs"] = graphs()
    info["graph_breaks"] = gbs = int(sum(counters["graph_break"].values()))
    flags = sync_flag(cderr is not None)
    if any(flags) and not all(flags):
        add("RANK_DIVERGENT", "cdist", None, cmsg(cderr) if cderr else "other rank failed", op=err_op(cderr or ""))
        raise Abort()
    if cderr is not None:
        if cperr is not None and cmsg(cperr) == cmsg(cderr):
            info["nondist"] += 1
            info["skip"] = "nondist_err:" + cmsg(cderr)
            return hashes, info
        kind = "FULLGRAPH_BREAK" if is_unsupported(cderr) else ("BW_ERR" if stage == "bw" else "ERR")
        add(kind, "cdist", None, cmsg(cderr), op=err_op(cderr))
        return hashes, info
    cfull = full_all(cdout, findings, "cdist", out_exprs)
    mtag = "GB_" if gbs else ""
    for i, (r, c) in enumerate(zip(rout, cfull)):
        if not isinstance(r, torch.Tensor) or not eager_ok[i]:
            continue
        if not isinstance(c, torch.Tensor):
            add("FULL_TENSOR_ERR", "cdist", i, norm_msg(c) if isinstance(c, Exception) else f"got {type(c).__name__}")
            continue
        dv, ev = cdout[i], dout[i]
        if isinstance(ev, DTensor) and not isinstance(dv, DTensor):
            add("NOT_DTENSOR", "cdist", i, f"got {type(dv).__name__}")
        if isinstance(ev, DTensor) and isinstance(dv, DTensor):
            if tuple(ev.placements) != tuple(dv.placements):
                add("PLACEMENT", "cdist", i, f"eager={ev.placements} compiled={dv.placements}")
            try:
                exp_local, _ = compute_local_shape_and_global_offset(dv.shape, dv.device_mesh, dv.placements)
                if tuple(exp_local) != tuple(dv.to_local().shape):
                    add(
                        "LOCAL_SHAPE",
                        "cdist",
                        i,
                        f"placements={dv.placements} global={tuple(dv.shape)} expected={tuple(exp_local)} "
                        f"actual={tuple(dv.to_local().shape)}",
                    )
            except Exception as e:
                add("LOCAL_SHAPE_UNCHECKED", "cdist", i, norm_msg(e))
        d = compare(r, c)
        if d is None:
            hashes.append(thash(c))
            continue
        if cpout is not None and isinstance(cpout[i], torch.Tensor) and compare(cpout[i], c) is None:
            info["nondist"] += 1  # compiled DTensor agrees with compiled plain tensors: an inductor numerics effect
            continue
        add(mtag + "MISMATCH", "cdist", i, d, placements=str(getattr(dv, "placements", "")))
    cgr = full_all(grads_of(cdin), findings, "cdist", [f"grad{k}" for k in range(len(cdin))])
    for k, (r, c) in enumerate(zip(rgr, cgr)):
        if not isinstance(r, torch.Tensor) or not eager_gok[k]:
            continue
        expr = f"grad of {fargs[k][0]}"
        if not isinstance(c, torch.Tensor):
            findings.append(dict(kind="GRAD_MISSING", leg="cdist", expr=expr, msg=str(type(c).__name__)))
            continue
        d = compare(r, c)
        if d is None:
            continue
        if cpgr is not None and isinstance(cpgr[k], torch.Tensor) and compare(cpgr[k], c) is None:
            info["nondist"] += 1
            continue
        findings.append(dict(kind=mtag + "GRAD_MISMATCH", leg="cdist", expr=expr, msg=d))

    # leg 5: same compiled function, fresh inputs of identical shape: must not recompile, must give the same values
    set_leg("again")
    g0 = graphs()
    aerr = None
    try:
        ain = inputs(True, specs)
        mark(cfg, ain, arg_specs)
        aout = call(cd, ain)
    except Exception as e:
        aerr = e
    flags = sync_flag(aerr is not None)
    if any(flags) and not all(flags):
        add("RANK_DIVERGENT", "again", None, cmsg(aerr) if aerr else "other rank failed", op=err_op(aerr or ""))
        raise Abort()
    if aerr is not None:
        add("ERR_SECOND_CALL", "again", None, cmsg(aerr), op=err_op(aerr))
    else:
        info["recompiles"] = graphs() - g0
        if info["recompiles"] > 0:
            add("RECOMPILE", "again", None, f"{info['recompiles']} new graphs on identical inputs")
        for i, (a, c) in enumerate(zip(aout, cdout)):
            if isinstance(a, DTensor) and isinstance(c, DTensor):
                if compare(c.to_local(), a.to_local()) is not None:
                    add("NONDETERMINISTIC", "again", i, "second call differs locally")

    var = cfg.get("variant")
    if var:
        arg_specs2 = [var["inputs"][j] for _, j in fargs]
        run_variant(
            var, src, cd, cp if cperr is None else None, inputs, call, cfg, arg_specs2, out_exprs, findings, info
        )
    torch._dynamo.reset()
    return hashes, info


def run_variant(var, src, cd, cp, inputs, call, cfg, arg_specs2, out_exprs, findings, info):
    """Call the already-compiled function on inputs with one size changed (dynamic configs only)."""
    specs2 = var["inputs"]
    set_leg("variant_ref")
    try:
        rout = call(mkfn(src, False), inputs(False, specs2))
    except Exception:
        info["variant"] = "ref_err"
        return
    set_leg("variant_eager")
    derr = None
    try:
        dout = call(mkfn(src, True), inputs(True, specs2))
    except Exception as e:
        derr = e
    flags = sync_flag(derr is not None)
    if any(flags) and not all(flags):
        findings.append(
            dict(kind="EAGER_RANK_DIVERGENT", leg="variant_eager", expr="", msg=norm_msg(derr) if derr else "other")
        )
        raise Abort()
    if derr is not None:
        info["variant"] = "eager_err"
        return
    efull = full_all(dout, findings, "variant_eager", out_exprs)
    eager_ok = [
        isinstance(r, torch.Tensor) and isinstance(e, torch.Tensor) and compare(r, e) is None
        for r, e in zip(rout, efull)
    ]
    # control: the plain-tensor function compiled on the first inputs, called on the variant like the DTensor one
    set_leg("variant_cplain")
    cpout, cperr = None, None
    try:
        if cp is not None:
            cpin = inputs(False, specs2)
            mark(cfg, cpin, arg_specs2)
            cpout = call(cp, cpin)
    except Exception as e:
        cperr = e
    set_leg("variant_cdist")
    g0 = graphs()
    cerr = None
    try:
        vin = inputs(True, specs2)
        mark(cfg, vin, arg_specs2)
        cout = call(cd, vin)
    except Exception as e:
        cerr = e
    info["variant_graphs"] = graphs() - g0
    flags = sync_flag(cerr is not None)
    if any(flags) and not all(flags):
        findings.append(
            dict(
                kind="VARIANT_RANK_DIVERGENT",
                leg="variant_cdist",
                expr="",
                msg=cmsg(cerr) if cerr else "other rank failed",
                op=err_op(cerr or ""),
            )
        )
        raise Abort()
    tag = f"{var['old']}->{var['new']}"
    if cerr is not None:
        if cperr is not None and cmsg(cperr) == cmsg(cerr):
            info["nondist"] += 1
            return
        findings.append(
            dict(kind="VARIANT_ERR", leg="variant_cdist", expr="", msg=cmsg(cerr), op=err_op(cerr), size=tag)
        )
        return
    if info["variant_graphs"] >= 4:
        findings.append(
            dict(kind="RECOMPILE_STORM", leg="variant_cdist", expr="", msg=f"{info['variant_graphs']} graphs", size=tag)
        )
    cfull = full_all(cout, findings, "variant_cdist", out_exprs)
    for i, (r, c) in enumerate(zip(rout, cfull)):
        if not isinstance(r, torch.Tensor) or not eager_ok[i]:
            continue
        if not isinstance(c, torch.Tensor):
            findings.append(dict(kind="VARIANT_FULL_TENSOR_ERR", leg="variant_cdist", expr=out_exprs[i], msg="err"))
            continue
        d = compare(r, c)
        if d is None:
            continue
        if cpout is not None and isinstance(cpout[i], torch.Tensor) and compare(cpout[i], c) is None:
            info["nondist"] += 1
            continue
        findings.append(dict(kind="VARIANT_MISMATCH", leg="variant_cdist", expr=out_exprs[i], msg=d, size=tag))


def main():
    proto.write(json.dumps({"ready": True, "rank": args.rank}) + "\n")
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        prog = json.loads(line)
        if prog.get("cmd") == "exit":
            break
        t0 = time.time()
        info: dict = {}
        findings: list = []
        hashes: list = []
        aborted, fatal = False, None
        try:
            if prog.get("track") == "module":
                from distfuzz.compile.module import run_module

                hashes, info = run_module(prog, MESHES, findings, sync_flag, set_leg, cmsg, err_op, Abort)
            else:
                hashes, info = run_op(prog, findings)
        except Abort:
            aborted = True
        except Exception:
            aborted = True
            fatal = traceback.format_exc()[-2000:]
        fatal_leg = CUR["leg"]
        set_leg("idle")
        if not aborted:
            try:
                dist.barrier(group=SYNC_PG)
            except Exception:
                aborted = True
        proto.write(
            json.dumps(
                dict(
                    id=prog.get("id"),
                    rank=args.rank,
                    findings=findings,
                    hashes=hashes,
                    aborted=aborted,
                    fatal=fatal,
                    fatal_leg=fatal_leg if fatal else "",
                    info=info,
                    dt=time.time() - t0,
                )
            )
            + "\n"
        )
        if aborted:
            break
    try:
        dist.destroy_process_group()
    except Exception:
        pass
    os._exit(0)


main()
