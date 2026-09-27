from __future__ import annotations

import ast
import json
import os
import re
from dataclasses import dataclass, field
from typing import Any

from . import gen_distfuzz, gen_dtensor

COMPAT_IMPORT = gen_dtensor.COMPAT


@dataclass
class Finding:
    source: str  # what the user passed
    fuzzer: str  # dtensor | collectives | manual
    fuzzer_sig: str | None  # the fuzzer's own signature string (may be None)
    script: str  # standalone repro source
    launcher: list  # argv prefix inside the container, e.g. ["python"]
    args: list = field(default_factory=list)
    world: int = 4
    rec: dict | None = None  # program JSON, when minimizable
    note: str = ""


def detect_json(rec):
    p = rec.get("prog", {})
    if "steps" in p and "inputs" in p:
        return "dtensor"
    if "calls" in p:
        return "collectives"
    raise ValueError("unknown program JSON format")


def launcher_for(src: str, world: int):
    if "mp.spawn" in src or "multiprocessing.spawn" in src or "start_processes" in src:
        return ["python"]
    if "init_process_group" in src:
        # torchrun-style (env:// rendezvous). --network none => pin the master to loopback.
        return [
            "torchrun",
            "--nnodes",
            "1",
            f"--nproc-per-node={world}",
            "--master-addr",
            "127.0.0.1",
            "--master-port",
            "29511",
        ]
    return ["python"]


def _compat_imports(tree_src: str) -> str:
    """Replace `from torch.distributed.tensor import ...` by the version-compat block."""
    lines = tree_src.splitlines()
    out = []
    for ln in lines:
        if re.match(r"^from torch\.distributed\.tensor import .*(DTensor|distribute_tensor)", ln):
            out.append(COMPAT_IMPORT)
        else:
            out.append(ln)
    return "\n".join(out)


def extract_case(path: str, case: str, world: int = 4) -> str:
    """AST-extract one @case function from a multi-case repro module into a standalone script."""
    src = open(path).read()
    tree = ast.parse(src)
    keep: list[Any] = []
    case_names = set()
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and (
            any(isinstance(d, ast.Name) and d.id == "case" for d in node.decorator_list)
            or [a.arg for a in node.args.args] == ["rank", "mesh"]
        ):
            case_names.add(node.name)
    if case not in case_names:
        raise KeyError(f"{case} not in {sorted(case_names)}")
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in case_names and node.name != case:
            continue  # other cases
        if isinstance(node, ast.If) and "__main__" in ast.unparse(node.test):
            continue  # CLI dispatch replaced below
        keep.append(node)
    parts = []
    for prev, n in zip([None] + keep[:-1], keep):
        start = n.decorator_list[0].lineno if getattr(n, "decorator_list", None) else n.lineno
        seg = "\n".join(src.splitlines()[start - 1 : n.end_lineno])
        imp = (ast.Import, ast.ImportFrom)
        sep = "" if prev is None else ("\n" if isinstance(prev, imp) and isinstance(n, imp) else "\n\n\n")
        parts.append(sep + seg)
    body = "".join(parts)
    body = _compat_imports(body)
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == case)
    doc = ast.get_docstring(fn) or ""
    head = (
        f'"""distfuzz.repro standalone repro, extracted from {os.path.basename(path)}::{case}.\n\n'
        f'{doc}\nRun:  python repro.py        # spawns {world} CPU/Gloo ranks\n"""\n'
    )
    tail = f"""

if __name__ == "__main__":
    s = socket.socket(); s.bind(("127.0.0.1", 0)); port = s.getsockname()[1]; s.close()
    mp.spawn(main, args=({world}, port, {case!r}), nprocs={world}, join=True)
"""
    # drop the original module docstring (first Expr) since we wrote our own header
    if body.lstrip().startswith(('"""', "'''")):
        body = body.split('"""', 2)[2] if body.lstrip().startswith('"""') else body.split("'''", 2)[2]
    return head + body.lstrip("\n") + tail


def load(source: str, world: int | None = None) -> Finding:
    if "::" in source:
        path, case = source.split("::", 1)
        w = world or 4
        script = extract_case(path, case, w)
        return Finding(
            source=source,
            fuzzer="manual",
            fuzzer_sig=None,
            script=script,
            launcher=["python"],
            world=w,
            note=f"case `{case}` extracted from {os.path.basename(path)}",
        )
    if source.endswith(".json"):
        rec = json.load(open(source))
        kind = detect_json(rec)
        if kind == "dtensor":
            w = world or rec.get("world", 4)
            rec["world"] = w
            return Finding(
                source=source,
                fuzzer=kind,
                fuzzer_sig=rec.get("sig"),
                rec=rec,
                script=gen_dtensor.generate(rec),
                launcher=["python"],
                world=w,
            )
        rec = gen_distfuzz.normalize(rec)
        w = rec["prog"]["world"]
        return Finding(
            source=source,
            fuzzer=kind,
            fuzzer_sig=rec.get("sig"),
            rec=rec,
            script=gen_distfuzz.generate(rec),
            launcher=["python"],
            world=w,
        )
    if source.endswith(".py"):
        src = open(source).read()
        w = world or 4
        m = re.search(r"--nproc[-_]per[-_]node[ =](\d+)", src)
        if m and not world:
            w = int(m.group(1))
        return Finding(
            source=source,
            fuzzer="manual",
            fuzzer_sig=None,
            script=src,
            launcher=launcher_for(src, w),
            world=w,
            note="standalone script, used verbatim",
        )
    raise ValueError(f"unsupported input {source}")


def regenerate(f: Finding, rec: dict) -> Finding:
    """Same finding, different program (used by the minimizer)."""
    gen = gen_dtensor if f.fuzzer == "dtensor" else gen_distfuzz
    return Finding(
        source=f.source,
        fuzzer=f.fuzzer,
        fuzzer_sig=f.fuzzer_sig,
        rec=rec,
        script=gen.generate(rec),
        launcher=f.launcher,
        world=f.world,
        note=f.note,
    )
