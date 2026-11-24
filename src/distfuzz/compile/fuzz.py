"""Compile-path fuzzer: each DTensor program (or FSDP2/TP module program) runs as a single-process reference, as
eager DTensor, compiled on plain tensors (control) and compiled on DTensor. Findings are differences the compiled
DTensor leg introduces over eager DTensor that the compiled plain-tensor control does not show.

    python -m distfuzz.compile fuzz --minutes 45 --seed 1 --out runs/compile
"""

import argparse
import json
import os
import random
import re
import time
from collections import Counter

from distfuzz.compile.prog import BACKENDS, CompileGen
from distfuzz.compile.world import World
from distfuzz.dtensor.fuzz import opname

INFO_KINDS = ("LOCAL_SHAPE_UNCHECKED",)


def _msg(kind, msg):
    if "MISMATCH" in kind:
        return msg.split(" ")[0]
    if kind in ("RECOMPILE", "RECOMPILE_STORM", "NONDETERMINISTIC", "LOCAL_SHAPE", "PLACEMENT"):
        return ""
    return re.sub(r"\d+", "N", msg)[:120]


def signature(f, prog):
    """Names the track, the compile configuration, the leg's finding kind, the op and a normalized message, e.g.
    ``COMPILE|inductor|dynamic|MISMATCH|sum|values``."""
    kind = f["kind"]
    if kind in ("HANG", "CRASH", "FATAL"):
        c = prog.get("cfg", {})
        m = re.sub(r"\d+", "N", f.get("msg", ""))[:80] if kind == "FATAL" else ""
        return f"{kind}|{prog.get('track', 'op')}|{f.get('leg', '')}|{c.get('backend', '')}|{c.get('dynamic', '')}|{m}"
    c = prog["cfg"]
    msg = _msg(kind, f.get("msg", ""))
    if prog.get("track") == "module":
        par = c["par"] + ("+sp" if c.get("sp") else "")
        where = re.sub(r"\d+", "N", f.get("expr", "")) if "MISMATCH" in kind and "step" not in f.get("expr", "") else ""
        if kind == "EAGER_MISMATCH":
            msg = ""
        if kind in ("LOSS_MISMATCH",):
            msg = ""
        head = "EAGER" if kind.startswith("EAGER") else "MODULE"
        return f"{head}|{par}|{c['backend']}|{c['dynamic']}|{c['wrap']}|{kind}|{where}|{msg}"
    expr = re.sub(r"\bv\d+\b", "v0", f.get("expr", ""))
    op = f.get("op") or (opname(expr) if expr else "?")
    if op.startswith(("SETITEM(", "BW(")):
        op = op.split("(")[0]
    head = "EAGER" if kind.startswith("EAGER") else "COMPILE"
    return f"{head}|{c['backend']}|{c['dynamic']}|{kind}|{op}|{msg}"


def collect(status, res, world, prog):
    findings = []
    if status == "timeout":
        findings.append(dict(kind="HANG", leg=",".join(sorted(set(world.legs()))), expr="", msg="program timeout"))
    elif status == "crash":
        findings.append(dict(kind="CRASH", leg=",".join(sorted(set(world.legs()))), expr="", msg=str(res)))
    infos = []
    if res and status in ("ok", "aborted"):
        seen: dict = {}
        for r in res:
            if r.get("fatal"):
                last = r["fatal"].strip().splitlines()[-1][:200]
                if "sync_flag" in r["fatal"] or "Timed out" in last or "Connection closed" in last:
                    # a rank waited in a control-plane collective while another was stuck or had failed elsewhere
                    last = "collective timeout or peer gone"
                findings.append(
                    dict(kind="FATAL", leg=r.get("fatal_leg", ""), expr="", msg=last, tb=r["fatal"][-1500:])
                )
            for f in r.get("findings", []):
                key = (f["kind"], f.get("leg"), f.get("expr"), f["msg"])
                if key in seen:
                    seen[key]["ranks"].append(r["rank"])
                else:
                    seen[key] = dict(f, ranks=[r["rank"]])
                    findings.append(seen[key])
            if r.get("info"):
                infos.append(r["info"])
        hs = [tuple(r.get("hashes", [])) for r in res]
        if status == "ok" and len(set(hs)) > 1:
            findings.append(dict(kind="RANK_HASH_DIVERGENT", leg="cdist", expr="", msg="full_tensor differs"))
    return findings, infos


def main(argv=None):
    ap = argparse.ArgumentParser(prog="distfuzz.compile fuzz")
    ap.add_argument("--world", type=int, default=4)
    ap.add_argument("--minutes", type=float, default=30)
    ap.add_argument("--out", default=os.path.join("runs", "compile"))
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--timeout", type=float, default=240, help="seconds per program before it counts as a hang")
    ap.add_argument("--backends", default=",".join(BACKENDS), help="comma list of aot_eager, inductor")
    ap.add_argument("--module-rate", type=float, default=0.25, help="share of FSDP2/TP module programs")
    ap.add_argument("--mutate-rate", type=float, default=0.3, help="share of programs mutated from compiled ones")
    ap.add_argument("--confirm-hangs", type=int, default=1, help="re-run the first program of each hang signature")
    ap.add_argument("--shared-cache", action="store_true", help="ranks share one inductor cache dir (as torchrun does)")
    ap.add_argument("--save-all", action="store_true", help="write every program in full to programs.jsonl (replay)")
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    rng = random.Random(a.seed)
    gen = CompileGen(a.world, rng, tuple(a.backends.split(",")), a.module_rate)
    world = World(
        a.world, os.path.join(a.out, "logs"), pg_timeout=min(120, a.timeout * 0.5), shared_cache=a.shared_cache
    )
    sigs: Counter = Counter()
    kinds: Counter = Counter()
    skips: Counter = Counter()
    tracks: Counter = Counter()
    agg = Counter()
    pool: list = []
    fsig = open(os.path.join(a.out, "findings.jsonl"), "a")
    fall = open(os.path.join(a.out, "failing.jsonl"), "a")
    fstat = open(os.path.join(a.out, "stats.jsonl"), "a")
    fprog = open(os.path.join(a.out, "programs.jsonl"), "a")
    t0 = time.time()
    end = t0 + a.minutes * 60
    execs = 0
    last_stat = 0.0
    pid = 0
    while time.time() < end:
        if pool and rng.random() < a.mutate_rate:
            prog = gen.mutate(rng.choice(pool[-100:]))
        else:
            prog = gen.generate()
        pid += 1
        prog["id"] = pid
        if prog.get("track") != "module" and not prog["steps"]:
            continue
        ts = time.time()
        status, res = world.run(prog, a.timeout)
        dt = time.time() - ts
        execs += 1
        tracks[prog.get("track", "op")] += 1
        findings, infos = collect(status, res, world, prog)
        if status == "timeout" and a.confirm_hangs:
            s = signature(findings[0], prog)
            if s not in sigs:
                st2, _ = world.run(prog, a.timeout)
                findings[0]["confirmed"] = st2 == "timeout"
        info = infos[0] if infos else {}
        if info.get("skip"):
            skips[info["skip"].split(":")[0]] += 1
        for k in ("graphs", "graph_breaks", "recompiles", "nondist"):
            agg[k] += sum(int(i.get(k) or 0) for i in infos[:1])
        if info.get("variant_graphs") is not None:
            agg["variant_calls"] += 1
            agg["variant_recompiled"] += int(info["variant_graphs"] > 0)
        if info.get("eager_mismatch"):
            agg["eager_mismatch_progs"] += 1
        rec = dict(id=pid, status=status, dt=round(dt, 2), info=info, track=prog.get("track"), n=len(findings))
        rec["cfg"] = {k: v for k, v in prog.get("cfg", {}).items() if k != "variant"}
        if a.save_all:
            rec["prog"] = prog
        fprog.write(json.dumps(rec) + "\n")
        if status == "ok" and not info.get("skip") and info.get("graphs"):
            pool.append(prog)
        for f in findings:
            s = signature(f, prog)
            kinds[f["kind"]] += 1
            rec = dict(sig=s, finding=f, prog=prog, t=round(time.time() - t0, 1), world=a.world)
            fall.write(json.dumps(rec) + "\n")
            if s not in sigs:
                fsig.write(json.dumps(rec) + "\n")
                fsig.flush()
                if f["kind"] not in INFO_KINDS:
                    print(f"[{time.time() - t0:7.1f}s] NEW {s}", flush=True)
            sigs[s] += 1
        fall.flush()
        if time.time() - last_stat > 30 or time.time() >= end:
            last_stat = time.time()
            el = last_stat - t0
            st = dict(
                t=round(el, 1),
                execs=execs,
                eps=round(execs / el, 3),
                sigs=len(sigs),
                restarts=world.restarts,
                tracks=tracks,
                kinds=kinds,
                skips=skips,
                agg=agg,
            )
            fstat.write(json.dumps(st) + "\n")
            fstat.flush()
            print(json.dumps(st), flush=True)
    world.close()
    with open(os.path.join(a.out, "summary.json"), "w") as f:
        json.dump(dict(sigs=sigs, kinds=kinds, skips=skips, tracks=tracks, agg=agg, execs=execs), f, indent=1)


if __name__ == "__main__":
    main()


def replay_main(argv=None):
    """Run saved programs in order in one world (``fuzz --save-all`` output), to reproduce findings that depend on
    what ran before them in the same rank processes."""
    ap = argparse.ArgumentParser(prog="distfuzz.compile replay")
    ap.add_argument("programs", help="programs.jsonl written with --save-all")
    ap.add_argument("--from-id", type=int, default=0)
    ap.add_argument("--upto", type=int, default=10**9)
    ap.add_argument("--only", default="", help="comma list of program ids to run, in file order")
    ap.add_argument("--world", type=int, default=4)
    ap.add_argument("--timeout", type=float, default=240)
    ap.add_argument("--logs", default="runs/replay_logs")
    a = ap.parse_args(argv)
    only = {int(x) for x in a.only.split(",") if x}
    world = World(a.world, a.logs, pg_timeout=min(120, a.timeout * 0.5))
    for line in open(a.programs):
        rec = json.loads(line)
        prog = rec.get("prog")
        if prog is None or not (a.from_id <= rec["id"] <= a.upto) or (only and rec["id"] not in only):
            continue
        status, res = world.run(prog, a.timeout)
        findings, _ = collect(status, res, world, prog)
        print(json.dumps(dict(id=rec["id"], status=status, sigs=sorted({signature(f, prog) for f in findings}))))
    world.close()
