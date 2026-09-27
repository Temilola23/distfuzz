import argparse
import datetime
import hashlib
import inspect
import json
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
ap.add_argument("--cov", type=int, default=1)
args = ap.parse_args()

proto = os.fdopen(os.dup(1), "w", buffering=1)
os.dup2(2, 1)

# imported only after stdout is redirected, so library output cannot corrupt the protocol stream
import torch  # noqa: E402
import torch._dynamo  # noqa: E402
import torch.distributed as dist  # noqa: E402
from torch.distributed.device_mesh import init_device_mesh  # noqa: E402
from torch.distributed.tensor import DTensor  # noqa: E402
from torch.distributed.tensor._utils import compute_local_shape_and_global_offset  # noqa: E402

from distfuzz.dtensor.gen import MUT_RE, NAME_RE  # noqa: E402
from distfuzz.dtensor.runtime import Ctx, compare, make_env  # noqa: E402

torch.set_num_threads(1)
torch._dynamo.config.cache_size_limit = 64
torch._dynamo.config.accumulated_cache_size_limit = 4096

COV_NEW: set[str] = set()
TARGET = os.sep + os.path.join("torch", "distributed", "tensor") + os.sep
if args.cov:
    mon = sys.monitoring  # type: ignore[attr-defined]  # coverage needs Python 3.12+
    TOOL = 4
    mon.use_tool_id(TOOL, "dtfuzz")
    DISABLE = mon.DISABLE

    def _line(code, line):
        fn = code.co_filename
        i = fn.find(TARGET)
        if i >= 0 and code.co_flags & inspect.CO_OPTIMIZED:  # skip import-time module/class bodies
            COV_NEW.add(fn[i + 1 :] + ":" + str(line))
        return DISABLE

    mon.register_callback(TOOL, mon.events.LINE, _line)
    mon.set_events(TOOL, mon.events.LINE)

dist.init_process_group(
    "gloo",
    init_method=f"tcp://127.0.0.1:{args.port}",
    rank=args.rank,
    world_size=args.world,
    timeout=datetime.timedelta(seconds=20),
)
W = args.world
# the 1-d mesh reuses the default PG; control-plane collectives on it can pair with a lagging DTensor collective
SYNC_PG = dist.new_group(backend="gloo", timeout=datetime.timedelta(seconds=20))
MESHES = {"1d": init_device_mesh("cpu", (W,))}
if W == 4:
    MESHES["2d"] = init_device_mesh("cpu", (2, 2))
if W == 6:
    MESHES["2d"] = init_device_mesh("cpu", (2, 3))


def norm_msg(e):
    s = (str(e).strip().splitlines() or [""])[0][:200]
    s = re.sub(r"0x[0-9a-f]+", "ADDR", s)
    s = re.sub(r"\d+", "N", s)
    return f"{type(e).__name__}: {s}"


def sync_flag(flag):
    t = torch.tensor([1 if flag else 0], dtype=torch.int64)
    out = [torch.zeros(1, dtype=torch.int64) for _ in range(W)]
    dist.all_gather(out, t, group=SYNC_PG)
    return [bool(o.item()) for o in out]


def thash(t):
    t = torch.empty(t.numel(), dtype=t.dtype).copy_(t.detach().reshape(-1))
    if t.dtype == torch.bool:
        t = t.to(torch.uint8)
    return hashlib.md5(t.view(torch.uint8).numpy().tobytes()).hexdigest()[:12]


class Abort(Exception):
    pass


def check_value(tag, si, expr, rv, dv, findings, hashes):
    n0 = len(findings)
    if not isinstance(rv, torch.Tensor):
        return False
    if not isinstance(dv, torch.Tensor):
        findings.append(dict(kind=tag + "TYPE", step=si, expr=expr, msg=f"got {type(dv).__name__}"))
        return True
    is_dt = isinstance(dv, DTensor)
    if not is_dt:
        findings.append(dict(kind=tag + "NOT_DTENSOR", step=si, expr=expr, msg="op returned plain Tensor"))
    else:
        if tuple(dv.shape) != tuple(rv.shape) or dv.dtype != rv.dtype:
            findings.append(
                dict(
                    kind=tag + "META",
                    step=si,
                    expr=expr,
                    msg=f"shape/dtype ref={tuple(rv.shape)},{rv.dtype} dt={tuple(dv.shape)},{dv.dtype}",
                )
            )
        try:
            exp_local, _ = compute_local_shape_and_global_offset(dv.shape, dv.device_mesh, dv.placements)
            if tuple(exp_local) != tuple(dv.to_local().shape):
                findings.append(
                    dict(
                        kind=tag + "LOCAL_SHAPE",
                        step=si,
                        expr=expr,
                        msg=f"placements={dv.placements} global={tuple(dv.shape)} "
                        f"expected local={tuple(exp_local)} actual={tuple(dv.to_local().shape)}",
                    )
                )
        except Exception as e:
            findings.append(
                dict(
                    kind=tag + "LOCAL_SHAPE_UNCHECKED",
                    step=si,
                    expr=expr,
                    msg=f"placements={dv.placements} {norm_msg(e)}",
                )
            )
    if dv.requires_grad != rv.requires_grad:
        findings.append(
            dict(kind=tag + "REQUIRES_GRAD", step=si, expr=expr, msg=f"ref={rv.requires_grad} dt={dv.requires_grad}")
        )
    full, err = None, None
    try:
        full = dv.full_tensor() if is_dt else dv
    except Exception as e:
        err = e
    flags = sync_flag(err is not None)
    if any(flags) and not all(flags):
        findings.append(
            dict(
                kind="RANK_DIVERGENT_ERROR",
                step=si,
                expr=expr + " [full_tensor]",
                msg=norm_msg(err) if err else "other rank failed",
            )
        )
        raise Abort()
    if err is not None:
        kind = "FUZZER_INPUT_ERR" if expr.startswith("mk(") else tag + "FULL_TENSOR_ERR"
        findings.append(
            dict(kind=kind, step=si, expr=expr, msg=norm_msg(err), placements=str(dv.placements) if is_dt else "")
        )
        return True
    d = compare(rv, full)
    if d is not None:
        findings.append(
            dict(kind=tag + "MISMATCH", step=si, expr=expr, msg=d, placements=str(dv.placements) if is_dt else "")
        )
    hashes.append(thash(full))
    return len(findings) > n0


def run_program(prog):
    mesh = MESHES[prog["mesh"]]
    rctx = Ctx("ref", None, prog["inputs"])
    dctx = Ctx("dist", mesh, prog["inputs"])
    renv, denv = make_env(rctx), make_env(dctx)
    findings, hashes = [], []
    live = []
    tainted = set()
    torch.manual_seed(0)
    try:
        for si, st in enumerate(prog["steps"]):
            expr, out = st["expr"], st["out"]
            refs = set(NAME_RE.findall(expr))
            cascade = bool(refs & tainted)
            rerr = derr = None
            try:
                rv = eval(expr, renv)
            except Exception as e:
                rerr, rv = e, None
            try:
                dv = eval(expr, denv)
            except Exception as e:
                derr, dv = e, None
                if os.environ.get("DTF_TB"):
                    traceback.print_exc()
            flags = sync_flag(derr is not None)
            if any(flags) and not all(flags):
                findings.append(
                    dict(
                        kind="RANK_DIVERGENT_ERROR",
                        step=si,
                        expr=expr,
                        msg=norm_msg(derr) if derr else "other rank failed",
                    )
                )
                raise Abort()
            mut = MUT_RE.search(expr) is not None
            if derr is not None and rerr is not None:
                if mut:
                    return findings, hashes, False
                continue
            if derr is not None:
                if expr.startswith("mk("):
                    findings.append(dict(kind="FUZZER_INPUT_ERR", step=si, expr=expr, msg=norm_msg(derr)))
                elif not cascade:
                    findings.append(dict(kind="DIST_ERR", step=si, expr=expr, msg=norm_msg(derr)))
                if mut:
                    return findings, hashes, False
                continue
            if rerr is not None:
                if not cascade:
                    findings.append(dict(kind="REF_ERR_DIST_OK", step=si, expr=expr, msg=norm_msg(rerr)))
                if mut:
                    return findings, hashes, False
                continue
            renv[out], denv[out] = rv, dv
            if isinstance(rv, torch.Tensor):
                live.append((si, out))
            if cascade:
                tainted.add(out)
                if mut:
                    tainted.update(refs)
            if st.get("check", True):
                scratch = []
                bad = check_value("", si, expr, rv, dv, scratch, hashes)
                if not cascade:
                    findings.extend(scratch)
                if bad:
                    tainted.add(out)
                    if mut:
                        tainted.update(refs)
        for si, out in live:
            scratch = []
            check_value("FINAL_", si, prog["steps"][si]["expr"], renv[out], denv[out], scratch, hashes)
            if out not in tainted:
                findings.extend(scratch)
        if tainted:
            return findings, hashes, False
        dmade = {}
        for j, dt in dctx.made:
            dmade.setdefault(j, dt)
        for i, rt in rctx.made:
            if not rt.requires_grad or i not in dmade:
                continue
            rg, dg = rt.grad, dmade[i].grad
            if (rg is None) != (dg is None):
                findings.append(
                    dict(
                        kind="GRAD_PRESENCE",
                        step=-1,
                        expr=f"mk({i}).grad",
                        msg=f"ref grad None={rg is None} dt grad None={dg is None}",
                    )
                )
                continue
            if rg is None:
                continue
            check_value("GRAD_", -1, f"mk({i}).grad", rg, dg, findings, hashes)
    except Abort:
        return findings, hashes, True
    return findings, hashes, False


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
        try:
            findings, hashes, aborted = run_program(prog)
            fatal = None
        except Exception:
            findings, hashes, aborted = [], [], True
            fatal = traceback.format_exc()[-2000:]
        cov = sorted(COV_NEW)
        COV_NEW.clear()
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
                    cov=cov,
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
