"""Minimize a compile-track finding while keeping its signature.

    python -m distfuzz.compile minimize runs/compile/findings.jsonl --index 3

Op programs: ddmin over the steps, then each input placement to Replicate, requires_grad off, and ``ret`` to
"leaves". Module programs: one block, no norm, no bias, SGD, fewer and equal batches, smaller dims, each kept only if
the signature survives. The compile config (backend, dynamic) is part of the signature and is never changed.
Writes ``<out>/min_<index>.json`` and a standalone script ``<out>/min_<index>_repro.py`` (no distfuzz imports).
"""

import argparse
import copy
import json
import os

from distfuzz.compile.fuzz import collect, signature
from distfuzz.compile.module import legalize
from distfuzz.compile.prog import MK_STEP_RE
from distfuzz.compile.world import World
from distfuzz.dtensor.gen import NAME_RE
from distfuzz.dtensor.minimize import ddmin

FLAKY = ("HANG", "CRASH", "FATAL", "RANK_DIVERGENT")


class Tester:
    def __init__(self, world_size, logdir, timeout):
        self.world = World(world_size, logdir, pg_timeout=min(120, timeout * 0.6))
        self.timeout = timeout
        self.runs = 0

    def sigs(self, prog):
        self.runs += 1
        status, res = self.world.run(prog, self.timeout)
        findings, _ = collect(status, res, self.world, prog)
        return {signature(f, prog) for f in findings}

    def close(self):
        self.world.close()


def well_formed(steps):
    defined = set()
    for st in steps:
        if set(NAME_RE.findall(st["expr"])) - defined:
            return False
        defined.add(st["out"])
    return any(not MK_STEP_RE.match(st["expr"]) for st in steps)


def minimize_op(prog, target, test):
    def with_steps(steps):
        p = copy.deepcopy(prog)
        p["steps"] = copy.deepcopy(steps)
        return p

    prog["steps"] = ddmin(prog["steps"], lambda st: well_formed(st) and test(with_steps(st)))
    for spec in prog["inputs"]:
        for d, old in enumerate(spec["pl"]):
            if old != "R":
                spec["pl"][d] = "R"
                if not test(prog):
                    spec["pl"][d] = old
        if spec.get("rg"):
            spec["rg"] = False
            if not test(prog):
                spec["rg"] = True
    if prog["cfg"].get("ret") == "all":
        prog["cfg"]["ret"] = "leaves"
        if not test(prog):
            prog["cfg"]["ret"] = "all"
    return prog


def minimize_module(prog, target, test, rng_seed=0):
    import random

    rng = random.Random(rng_seed)
    edits = [
        ("model", "blocks", 1),
        ("model", "norm", "none"),
        ("model", "bias", False),
        ("model", "kind", "mlp"),
        ("cfg", "opt", "sgd"),
        ("cfg", "raf", True),
        ("cfg", "sp", False),
    ]
    for sec, k, v in edits:
        if prog[sec].get(k) == v:
            continue
        old = prog[sec][k]
        prog[sec][k] = v
        if not test(prog):
            prog[sec][k] = old
    while len(prog["cfg"]["batches"]) > 1:
        old = prog["cfg"]["batches"]
        prog["cfg"]["batches"] = old[:-1]
        if not test(prog):
            prog["cfg"]["batches"] = old
            break
    for d in ("I", "D", "H", "O", "V", "S"):
        for v in (1, 2, 4):
            if v >= prog["model"][d]:
                break
            old = prog["model"][d]
            prog["model"][d] = v
            cand = legalize(copy.deepcopy(prog), rng, 4)
            if cand["model"][d] == v and test(cand):
                break
            prog["model"][d] = old
    return prog


def main(argv=None):
    ap = argparse.ArgumentParser(prog="distfuzz.compile minimize")
    ap.add_argument("findings", help="findings.jsonl or failing.jsonl")
    ap.add_argument("--index", type=int, default=0)
    ap.add_argument("--world", type=int, default=4)
    ap.add_argument("--timeout", type=float, default=200)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    recs = [json.loads(line) for line in open(a.findings)]
    rec = recs[a.index]
    target = rec["sig"]
    out = a.out or os.path.dirname(os.path.abspath(a.findings))
    os.makedirs(out, exist_ok=True)
    tester = Tester(a.world, os.path.join(out, "minlogs"), a.timeout)
    flaky = bool(set(target.split("|")) & set(FLAKY)) or "RANK_DIVERGENT" in target

    def test(p):
        for _ in range(2 if flaky else 1):
            if target not in tester.sigs(p):
                return False
        return True

    prog = copy.deepcopy(rec["prog"])
    prog.pop("mut", None)
    if not test(prog):
        print(json.dumps(dict(index=a.index, sig=target, result="not reproducible", runs=tester.runs)))
        tester.close()
        return
    n0 = len(prog.get("steps", []))
    if prog.get("track") == "module":
        prog = minimize_module(prog, target, test)
    else:
        prog = minimize_op(prog, target, test)
    tester.close()
    path = os.path.join(out, f"min_{a.index}.json")
    with open(path, "w") as f:
        json.dump(dict(sig=target, prog=prog, runs=tester.runs), f, indent=1)
    from distfuzz.compile.repro import write_repro

    rp = write_repro(prog, os.path.join(out, f"min_{a.index}_repro.py"))
    print(
        json.dumps(
            dict(index=a.index, sig=target, steps=f"{n0}->{len(prog.get('steps', []))}", runs=tester.runs, repro=rp)
        )
    )


if __name__ == "__main__":
    main()
