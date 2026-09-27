import argparse
import json
import os
import random
import re
import time
from collections import Counter

from distfuzz.dtensor.gen import Gen
from distfuzz.dtensor.world import World

BUG_KINDS = (
    "MISMATCH",
    "META",
    "LOCAL_SHAPE",
    "REQUIRES_GRAD",
    "GRAD_PRESENCE",
    "NOT_DTENSOR",
    "TYPE",
    "REF_ERR_DIST_OK",
    "RANK_DIVERGENT_ERROR",
    "HANG",
    "CRASH",
    "FATAL",
    "RANK_HASH_DIVERGENT",
    "FULL_TENSOR_ERR",
)


def opname(expr):
    e = expr.strip()
    if e.startswith("R("):
        return "redistribute"
    if e.startswith("CMP("):
        m = re.search(r"lambda [^:]*: (.*?)['\"],", e)
        return "compile:" + (opname(m.group(1)) if m else "?")
    m = re.match(r"^(torch\.(?:nn\.functional\.|linalg\.)?|F\.)(\w+)\(", e)
    if m:
        return m.group(2)
    m = re.match(r"^v\d+((?:\.\w+(?:\([^()]*\))?)*?)\.(\w+)\(", e)
    if m:
        return m.group(2)
    if re.match(r"^v\d+\[", e):
        return "getitem"
    m = re.search(r"\s([-+*/@%<>=&|]{1,2})\s", e)
    if m:
        return m.group(1)
    if e.startswith("-"):
        return "neg"
    if e.startswith("~"):
        return "invert"
    m = re.match(r"^v\d+\.(\w+)$", e)
    if m:
        return m.group(1)
    return e[:20]


def signature(f):
    kind = f["kind"]
    msg = f.get("msg", "")
    expr = re.sub(r"\bv\d+\b", "v0", f.get("expr", ""))  # v0, not v: opname() matches ^v\d+
    expr = re.sub(r"\bmk\((\d+)\)", "mk(N)", expr)
    if "MISMATCH" in kind:
        msg = msg.split(" ")[0]
    elif kind in ("META", "LOCAL_SHAPE", "REQUIRES_GRAD", "GRAD_PRESENCE") or kind.endswith(("LOCAL_SHAPE", "META")):
        msg = ""
    msg = re.sub(r"\d+", "N", msg)[:120]
    op = opname(expr)
    if op.startswith(("SETITEM(", "BW(")):
        op = op.split("(")[0]
    return f"{kind}|{op}|{msg}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--world", type=int, default=4)
    ap.add_argument("--mode", choices=["guided", "random"], default="guided")
    ap.add_argument("--minutes", type=float, default=30)
    ap.add_argument("--out", default=os.path.join("runs", "dtensor"))
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--timeout", type=float, default=30)
    ap.add_argument("--no-compile", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rng = random.Random(a.seed)
    gen = Gen(a.world, rng, allow_compile=not a.no_compile)
    world = World(a.world, os.path.join(a.out, "logs"))
    cov = set()
    corpus = []
    sigs = Counter()
    kinds_count = Counter()
    fsig = open(os.path.join(a.out, "findings.jsonl"), "a")
    fstat = open(os.path.join(a.out, "stats.jsonl"), "a")
    t0 = time.time()
    end = t0 + a.minutes * 60
    execs = steps_total = 0
    last_stat = 0
    pid = 0
    while time.time() < end:
        if a.mode == "guided" and corpus and rng.random() < 0.7:
            parent = rng.choice(corpus[-200:] if rng.random() < 0.5 else corpus)
            prog = gen.mutate(parent)
        else:
            prog = gen.generate()
        pid += 1
        prog["id"] = pid
        if not prog["steps"]:
            continue
        has_cmp = any("CMP(" in s["expr"] for s in prog["steps"])
        status, res = world.run(prog, a.timeout * (6 if has_cmp else 1))
        execs += 1
        steps_total += len(prog["steps"])
        findings = []
        new_cov = set()
        if status == "timeout":
            findings.append(dict(kind="HANG", step=-1, expr="", msg="program timeout"))
        elif status == "crash":
            findings.append(dict(kind="CRASH", step=-1, expr="", msg=str(res)))
        if res and status in ("ok", "aborted"):
            seen = {}
            for r in res:
                if r.get("fatal"):
                    findings.append(dict(kind="FATAL", step=-1, expr="", msg=r["fatal"].strip().splitlines()[-1][:200]))
                for f in r.get("findings", []):
                    key = (f["kind"], f["step"], f["msg"])
                    if key in seen:
                        seen[key]["ranks"].append(r["rank"])
                    else:
                        seen[key] = dict(f, ranks=[r["rank"]])
                        findings.append(seen[key])
                new_cov.update(r.get("cov", []))
            hs = [tuple(r.get("hashes", [])) for r in res]
            if status == "ok" and len(set(hs)) > 1:
                findings.append(
                    dict(kind="RANK_HASH_DIVERGENT", step=-1, expr="", msg="full_tensor differs across ranks")
                )
        added = new_cov - cov
        if added:
            cov |= added
            if a.mode == "guided" and status == "ok":
                corpus.append(prog)
        for f in findings:
            s = signature(f)
            kinds_count[f["kind"]] += 1
            if s not in sigs:
                fsig.write(json.dumps(dict(sig=s, finding=f, prog=prog, t=time.time() - t0, world=a.world)) + "\n")
                fsig.flush()
                if any(k in f["kind"] for k in BUG_KINDS):
                    print(f"[{time.time() - t0:7.1f}s] NEW {s}", flush=True)
            sigs[s] += 1
        if time.time() - last_stat > 30 or time.time() >= end:
            last_stat = time.time()
            el = last_stat - t0
            st = dict(
                t=round(el, 1),
                execs=execs,
                eps=round(execs / el, 2),
                steps=steps_total,
                cov=len(cov),
                corpus=len(corpus),
                sigs=len(sigs),
                restarts=world.restarts,
                kinds=kinds_count,
            )
            fstat.write(json.dumps(st) + "\n")
            fstat.flush()
            print(json.dumps(st), flush=True)
    world.close()
    json.dump(dict(sigs=sigs, cov=sorted(cov)), open(os.path.join(a.out, "summary.json"), "w"))


if __name__ == "__main__":
    main()
