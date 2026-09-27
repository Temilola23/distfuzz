import argparse
import json
import os
import random
import re
import shutil
import tempfile
import time

from distfuzz.dcp.gen import Gen
from distfuzz.dcp.world import World

# Checkpoints are written here; point it at a tmpfs (the Docker recipe uses /ckpt).
CK_ROOT = os.environ.get("CK_ROOT") or tempfile.gettempdir()


class Pools:
    def __init__(self, logdir, cov=True, max_w=4):
        self.logdir, self.cov = logdir, cov
        self.pools = {}
        self.max_w = max_w
        self.budget = int(os.environ.get("PROC_BUDGET", 7))
        self.evictions = 0

    def get(self, w):
        if w in self.pools:
            self.pools[w] = self.pools.pop(w)
            return self.pools[w]
        while self.pools and sum(self.pools) + w > self.budget:
            old = next(iter(self.pools))
            self.pools.pop(old).close()
            self.evictions += 1
        self.pools[w] = World(w, os.path.join(self.logdir, f"w{w}"), cov=self.cov)
        return self.pools[w]

    @property
    def restarts(self):
        return sum(p.restarts for p in self.pools.values()) + self.evictions

    def close(self):
        for p in self.pools.values():
            p.close()


def run_scenario(pools, scn, workdir, timeout=120):
    shutil.rmtree(workdir, ignore_errors=True)
    os.makedirs(workdir, exist_ok=True)
    findings, cov = [], set()
    for si, seg in enumerate(scn["segs"]):
        world = pools.get(seg["w"])
        status, res = world.run({"cmd": "seg", "scn": scn, "si": si, "dir": workdir}, timeout)
        base = dict(seg=si, par=seg["par"], w=seg["w"])
        if status == "timeout":
            findings.append(dict(kind="HANG", phase="seg", msg="segment timeout", **base))
            break
        if status == "crash":
            findings.append(dict(kind="CRASH", phase="seg", msg=str(res)[:200], **base))
            break
        seen = {}
        for r in res:
            cov.update(r.get("cov", []))
            if r.get("fatal"):
                f = dict(
                    kind="FATAL", phase="executor", msg=r["fatal"].strip().splitlines()[-1][:200], tb=r["fatal"], **base
                )
                seen.setdefault(("FATAL", f["msg"]), f)
            for f in r.get("findings", []):
                key = (f["kind"], f.get("phase"), f["msg"])
                if key in seen:
                    seen[key].setdefault("ranks", []).append(r["rank"])
                else:
                    seen[key] = dict(f, ranks=[r["rank"]], par=seg["par"], w=seg["w"])
        findings.extend(seen.values())
        hs = {tuple(r.get("hashes", [])) for r in res}
        if status == "ok" and not findings and len(hs) > 1:
            findings.append(
                dict(kind="RANK_HASH_DIVERGENT", phase="gather", msg="ranks disagree on full state", **base)
            )
        if findings:
            break
    return findings, cov


def signature(f):
    k = f["kind"]
    msg = re.sub(r"\d+(\.\d+)?(e-?\d+)?", "N", f.get("msg", ""))[:140]
    if k in ("LOAD_MISMATCH", "STATE_MISMATCH", "ROUNDTRIP_MISMATCH"):
        return f"{k}|{f.get('pair')}|{','.join(f.get('cats', []))}"
    if k == "LOSS_MISMATCH":
        return f"{k}|{f.get('pair')}"
    if k in ("HANG", "CRASH", "RANK_HASH_DIVERGENT"):
        return f"{k}|{f.get('par')}"
    return f"{k}|{f.get('phase')}|{msg}"


def main(argv):
    ap = argparse.ArgumentParser(prog="python -m distfuzz.dcp fuzz")
    ap.add_argument("--mode", choices=["guided", "random"], default="guided")
    ap.add_argument("--minutes", type=float, default=60)
    ap.add_argument("--out", default=os.path.join("runs", "dcp"))
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--timeout", type=float, default=120)
    ap.add_argument("--max-w", type=int, default=4)
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    rng = random.Random(a.seed)
    gen = Gen(rng, max_w=a.max_w)
    pools = Pools(os.path.join(a.out, "logs"), max_w=a.max_w)
    cov, corpus, sigs, kinds = set(), [], {}, {}
    fsig = open(os.path.join(a.out, "findings.jsonl"), "a")
    fstat = open(os.path.join(a.out, "stats.jsonl"), "a")
    t0 = time.time()
    end = t0 + a.minutes * 60
    execs = segs_total = steps_total = 0
    last = 0
    pid = 0
    workdir = os.path.join(CK_ROOT, f"fz{a.seed}")
    while time.time() < end:
        if a.mode == "guided" and corpus and rng.random() < 0.7:
            parent = rng.choice(corpus[-100:] if rng.random() < 0.5 else corpus)
            scn = gen.mutate(parent)
        else:
            scn = gen.generate()
        pid += 1
        scn["id"] = pid
        findings, ncov = run_scenario(pools, scn, workdir, a.timeout)
        execs += 1
        segs_total += len(scn["segs"])
        steps_total += sum(s["steps"] for s in scn["segs"])
        added = ncov - cov
        if added:
            cov |= added
            if a.mode == "guided" and not findings:
                corpus.append(scn)
        for f in findings:
            s = signature(f)
            kinds[f["kind"]] = kinds.get(f["kind"], 0) + 1
            if s not in sigs:
                sigs[s] = 0
                fsig.write(json.dumps(dict(sig=s, finding=f, scn=scn, t=round(time.time() - t0, 1))) + "\n")
                fsig.flush()
                print(f"[{time.time() - t0:7.1f}s] NEW {s}", flush=True)
            sigs[s] += 1
        if time.time() - last > 30 or time.time() >= end:
            last = time.time()
            el = last - t0
            st = dict(
                t=round(el, 1),
                execs=execs,
                eps=round(execs / el, 3),
                segs=segs_total,
                steps=steps_total,
                cov=len(cov),
                corpus=len(corpus),
                sigs=len(sigs),
                restarts=pools.restarts,
                kinds=kinds,
            )
            fstat.write(json.dumps(st) + "\n")
            fstat.flush()
            print(json.dumps(st), flush=True)
    pools.close()
    json.dump(dict(sigs=sigs, cov=sorted(cov)), open(os.path.join(a.out, "summary.json"), "w"))
