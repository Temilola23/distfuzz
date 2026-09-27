from __future__ import annotations

import importlib
import re

DTYPE_EXPR = {
    "f32": "torch.float32",
    "f64": "torch.float64",
    "bf16": "torch.bfloat16",
    "f16": "torch.float16",
    "i64": "torch.int64",
    "i32": "torch.int32",
    "bool": "torch.bool",
}


def _fuzzer_runtime():
    """Load distfuzz.dtensor.runtime at *generation* time only (single process, no ranks)."""
    return importlib.import_module("distfuzz.dtensor.runtime")


def tensor_literal(t):
    import torch

    dt = {
        torch.float32: "torch.float32",
        torch.float64: "torch.float64",
        torch.bfloat16: "torch.bfloat16",
        torch.float16: "torch.float16",
        torch.int64: "torch.int64",
        torch.int32: "torch.int32",
        torch.bool: "torch.bool",
    }[t.dtype]
    vals = t.flatten().tolist()
    if t.dtype.is_floating_point:
        vals = [float(v) for v in vals]
        body = "[" + ", ".join(repr(v) if v != int(v) else f"{int(v)}." for v in vals) + "]"
    elif t.dtype == torch.bool:
        body = "[" + ", ".join("True" if v else "False" for v in vals) + "]"
    else:
        body = repr(vals)
    shape = tuple(t.shape)
    return f"torch.tensor({body}, dtype={dt}).reshape({shape!r})"


PL_EXPR = {"R": "Replicate()"}


def pl_expr(p):
    if p == "R":
        return "Replicate()"
    if p.startswith("S"):
        return f"Shard({int(p[1:])})"
    if p.startswith("P:"):
        return f"Partial({p[2:]!r})"
    raise ValueError(p)


COMPAT = """try:  # public DTensor API (torch >= 2.5); fall back to the private module on older wheels
    from torch.distributed.tensor import DTensor, Partial, Replicate, Shard, distribute_tensor
except ImportError:
    from torch.distributed._tensor import DTensor, Replicate, Shard, distribute_tensor
    try:
        from torch.distributed._tensor import Partial
    except ImportError:
        from torch.distributed._tensor.placement_types import _Partial as Partial"""


def generate(rec, name="repro"):
    """rec: {"sig", "prog", "world"} as written by distfuzz.dtensor minimize (or a findings.jsonl row)."""
    rt = _fuzzer_runtime()
    prog, world = rec["prog"], rec.get("world", 4)
    mesh_shape = (world,) if prog["mesh"] == "1d" else ((2, 2) if world == 4 else (2, 3))
    used_helpers = {
        h for st in prog["steps"] for h in ("R", "SETITEM", "BW", "CMP") if re.search(rf"\b{h}\(", st["expr"])
    }
    inp_code = []
    for i, spec in enumerate(prog["inputs"]):
        T = rt.gen_full(spec)
        pls = spec["pl"]
        pdims = [d for d, p in enumerate(pls) if p.startswith("P:")]
        inp_code.append(
            f"# input {i}: global shape {tuple(spec['shape'])}, {spec['dtype']}, placements {pls}"
            + (", requires_grad" if spec.get("rg") else "")
        )
        inp_code.append(f"X{i} = {tensor_literal(T)}")
        if pdims:
            n = 1
            for d in pdims:
                n *= mesh_shape[d]
            op = pls[pdims[0]][2:]
            pieces = rt.partial_pieces(T, n, op, spec["seed"] + 7919)
            inp_code.append(f"# per-rank local pieces whose {op}-reduction over the Partial mesh dims equals X{i}")
            inp_code.append(f"X{i}_PIECES = [\n" + "".join(f"    {tensor_literal(p)},\n" for p in pieces) + "]")
        inp_code.append(f"PL{i} = [{', '.join(pl_expr(p) for p in pls)}]")

    mk_lines = [
        "def mk(i):",
        '    """Input i: a plain tensor (reference) or a DTensor with the fuzzed placements."""',
        "    X, PL = INPUTS[i]",
        "    if MODE == 'ref':",
        "        t = X.clone()",
        "        return t.requires_grad_() if RG[i] else t",
        "    pdims = [d for d, p in enumerate(PL) if p.is_partial()]",
        "    if pdims:",
        "        coord, idx = MESH.get_coordinate(), 0",
        "        for d in pdims:",
        "            idx = idx * MESH.size(d) + coord[d]",
        "        local = PIECES[i][idx]  # this rank's piece; must not be broadcast from rank 0",
        "        if not all(p.is_partial() or p.is_replicate() for p in PL):",
        "            repl = [Replicate() if p.is_partial() else p for p in PL]",
        "            local = distribute_tensor(local, MESH, repl, src_data_rank=None).to_local()",
        "        t = DTensor.from_local(local, MESH, PL, run_check=False, shape=X.shape, stride=X.stride())",
        "    else:",
        "        t = distribute_tensor(X, MESH, PL)",
        "    return t.detach().requires_grad_() if RG[i] else t",
    ]
    helpers = []
    if "R" in used_helpers:
        helpers += [
            "",
            "def R(x, pls):",
            '    """redistribute (identity on the reference path)."""',
            "    if MODE == 'ref':",
            "        return x.clone()",
            "    y = x.redistribute(x.device_mesh, [parse_pl(p) for p in pls])",
            "    return y.clone() if y is x else y",
            "",
            "def parse_pl(s):",
            "    return Replicate() if s == 'R' else Shard(int(s[1:])) if s[0] == 'S' else Partial(s[2:])",
        ]
    if "SETITEM" in used_helpers:
        helpers += ["", "def SETITEM(x, idx, y):", "    x[idx] = y", "    return x"]
    if "BW" in used_helpers:
        helpers += ["", "def BW(x):", "    x.sum().backward()"]
    if "CMP" in used_helpers:
        helpers += [
            "",
            "_COMPILED = {}",
            "def CMP(src, *args):",
            "    fn = eval(src)",
            "    if MODE == 'ref':",
            "        return fn(*args)",
            "    if src not in _COMPILED:",
            "        _COMPILED[src] = torch.compile(fn, backend='aot_eager', dynamic=False)",
            "    return _COMPILED[src](*args)",
        ]

    steps = [(st["out"], st["expr"]) for st in prog["steps"]]
    body = "\n".join(f"    {o} = {e}\n    check({o!r}, {o})" for o, e in steps)
    exprs = dict(steps)
    ninp = len(prog["inputs"])
    has_pieces = [any(p.startswith("P:") for p in s["pl"]) for s in prog["inputs"]]
    src = f'''#!/usr/bin/env python3
"""distfuzz.repro standalone repro: generated from a distfuzz.dtensor program (plain PyTorch, no fuzzer imports).

Fuzzer signature: {rec.get("sig", "?")}
Run:  python {name}.py        # spawns {world} CPU/Gloo ranks on a {mesh_shape} DeviceMesh

Every step runs twice on each rank: on plain tensors (the reference) and on DTensors.
For each value, DTensor.full_tensor() must equal the reference. Disagreements print
'<step>: MISMATCH' followed by the expected (reference) and actual (DTensor) values.
"""
import datetime
import faulthandler
import math
import os
import socket
import sys
import time
import traceback

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.distributed.device_mesh import init_device_mesh
{COMPAT}

WORLD = {world}
MESH_SHAPE = {mesh_shape!r}
inf = math.inf

{chr(10).join(inp_code)}

INPUTS = [{", ".join(f"(X{i}, PL{i})" for i in range(ninp))}]
PIECES = [{", ".join(f"X{i}_PIECES" if has_pieces[i] else "None" for i in range(ninp))}]
RG = {[bool(s.get("rg")) for s in prog["inputs"]]!r}
EXPRS = {exprs!r}
MODE, MESH = "ref", None


{chr(10).join(mk_lines)}
{chr(10).join(helpers)}


def program():
{body}


REF, FIRST_BAD, RANK = {{}}, [None], [0]


def check(name, x):
    """Reference pass: snapshot each value. DTensor pass: compare full_tensor() with the snapshot
    right after the step that produced it (so later in-place ops cannot blur the attribution)."""
    if MODE == "ref":
        if isinstance(x, torch.Tensor):
            REF[name] = x.detach().clone()
        return
    if name not in REF or FIRST_BAD[0] is not None or not isinstance(x, torch.Tensor):
        return
    r = REF[name]
    full = x.full_tensor() if isinstance(x, DTensor) else x
    diff = close(r, full)
    if diff is None:
        return
    FIRST_BAD[0] = name
    if RANK[0] == 0:
        print(f"{{EXPRS[name]}}: MISMATCH ({{diff}}) at {{name}}  placements={{getattr(x, 'placements', None)}}\\n"
              f"   expected {{tuple(r.shape)}} {{r.dtype}}: {{r.detach().flatten()[:12].tolist()}}\\n"
              f"   actual   {{tuple(full.shape)}} {{full.dtype}}: {{full.detach().flatten()[:12].tolist()}}", flush=True)


def run(mode):
    global MODE
    MODE = mode
    torch.manual_seed(0)
    program()


def close(ref, got):
    if tuple(ref.shape) != tuple(got.shape):
        return f"shape expected {{tuple(ref.shape)}} got {{tuple(got.shape)}}"
    if ref.dtype != got.dtype:
        return f"dtype expected {{ref.dtype}} got {{got.dtype}}"
    if ref.numel() == 0:
        return None
    r, g = ref.detach(), got.detach()
    if r.dtype.is_floating_point:
        tol = 2e-2 if r.dtype in (torch.bfloat16, torch.float16) else 1e-4
        ok = torch.isclose(g.double(), r.double(), rtol=tol, atol=tol, equal_nan=True)
        return None if bool(ok.all()) else "values"
    return None if torch.equal(r, g) else "values"


def main(rank, port):
    global MESH
    faulthandler.dump_traceback_later(60, exit=True)   # hang watchdog: dumps every thread's stack
    dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{{port}}", rank=rank, world_size=WORLD,
                            timeout=datetime.timedelta(seconds=30))
    MESH = init_device_mesh("cpu", MESH_SHAPE)
    RANK[0] = rank
    run("ref")
    try:
        run("dist")
    except Exception as e:
        print(f"[rank {{rank}}] raised {{type(e).__name__}}: {{str(e).splitlines()[0][:300] if str(e) else ''}}", flush=True)
        traceback.print_exc()
        sys.stdout.flush(); sys.stderr.flush()
        time.sleep(3)  # let other ranks that also raise report before mp.spawn tears the job down
        os._exit(1)
    dist.barrier()
    if rank == 0:
        print(f"[distfuzz-repro] DONE torch {{torch.__version__}}", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    s = socket.socket(); s.bind(("127.0.0.1", 0)); port = s.getsockname()[1]; s.close()
    mp.spawn(main, args=(port,), nprocs=WORLD, join=True)
'''
    return src


def units(rec):
    """Minimization units = program steps."""
    return list(range(len(rec["prog"]["steps"])))


def with_units(rec, keep):
    import copy

    r = copy.deepcopy(rec)
    r["prog"]["steps"] = [s for i, s in enumerate(rec["prog"]["steps"]) if i in keep]
    return r


def is_valid(rec):
    """Every vN used by a step must be defined by an earlier step."""
    defined = set()
    for st in rec["prog"]["steps"]:
        for v in re.findall(r"\bv\d+\b", st["expr"]):
            if v not in defined:
                return False
        defined.add(st["out"])
    return bool(rec["prog"]["steps"])
