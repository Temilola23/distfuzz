import argparse
import json
import os
import random
import re
import shutil
import tempfile
import time

from distfuzz.dcp.gen import Gen
from distfuzz.dcp.train_gen import TrainGen
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


def static_empty(scn, seg):
    """Whether FSDP/EP sharding leaves some rank an empty dim-0 shard (torch.chunk semantics), from shapes alone."""
    from distfuzz.dcp.blocks import chunk_sizes
    from distfuzz.dcp.runtime import build_model

    par, w = seg["par"], seg["w"]
    deg = {"fsdp": w, "ep": w, "hsdp": seg.get("mesh2d", [1, w])[1], "fsdp_tp": seg.get("mesh2d", [w, 1])[0]}.get(par)
    if not deg or deg == 1:
        return False
    model = build_model(scn, 0)
    return any(p.dim() and 0 in chunk_sizes(p.shape[0], deg) for p in model.parameters())


def hang_finding(rep, base):
    """HANG (no rank raised) or PARTIAL_ERR (some ranks raised, the others are stuck in a collective)."""
    if not rep:
        return dict(kind="HANG", phase="seg", msg="segment timeout", **base)
    errs = [r for r in rep if r.get("err")]
    stuck = [r for r in rep if not r.get("returned")]
    locs = sorted({r["loc"] for r in stuck if r.get("loc")})
    detail = "; ".join(
        f"rank{r['rank']}:{'returned' if r['returned'] else str(r.get('phase')) + '@' + str(r.get('loc'))}" for r in rep
    )
    if errs:
        e = errs[0]
        return dict(
            kind="PARTIAL_ERR",
            phase=e.get("err_phase") or "seg",
            msg=e["err"],
            err_ranks=[r["rank"] for r in errs],
            locs=locs,
            detail=detail,
            **base,
        )
    # the earliest phase any stuck rank is in (ranks that got further wait for it)
    order = ["build", "load", "gather_after_load", "resave", "train", "gather_end", "presave", "save"]
    phs = [r["phase"] for r in stuck if r.get("phase")]
    phase = min(phs, key=lambda p: order.index(p.split(":")[0]) if p.split(":")[0] in order else 99) if phs else "seg"
    return dict(kind="HANG", phase=phase, msg="segment timeout", locs=locs, detail=detail, **base)


def run_scenario(pools, scn, workdir, timeout=120, tags=None):
    shutil.rmtree(workdir, ignore_errors=True)
    os.makedirs(workdir, exist_ok=True)
    findings, cov = [], set()
    tags = {} if tags is None else tags
    for si, seg in enumerate(scn["segs"]):
        tags.clear()  # tags describe the segment that produced the finding
        world = pools.get(seg["w"])
        status, res = world.run({"cmd": "seg", "scn": scn, "si": si, "dir": workdir}, timeout, err_grace=10)
        base = dict(seg=si, par=seg["par"], w=seg["w"])
        if status == "timeout":
            findings.append(hang_finding(res, base))
            try:
                tags.setdefault("empty_shard", static_empty(scn, seg))
            except Exception:  # noqa
                pass
            break
        if status == "crash":
            findings.append(dict(kind="CRASH", phase="seg", msg=str(res)[:200], **base))
            break
        seen = {}
        for r in res:
            cov.update(r.get("cov", []))
            for k, v in (r.get("tags") or {}).items():
                tags[k] = tags.get(k, False) or bool(v)
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


def signature(f, scn=None):
    if scn is not None and scn.get("profile") == "train":
        return train_signature(f, scn)
    k = f["kind"]
    msg = re.sub(r"\d+(\.\d+)?(e-?\d+)?", "N", f.get("msg", ""))[:140]
    if k in ("LOAD_MISMATCH", "STATE_MISMATCH", "ROUNDTRIP_MISMATCH"):
        return f"{k}|{f.get('pair')}|{','.join(f.get('cats', []))}"
    if k == "LOSS_MISMATCH":
        return f"{k}|{f.get('pair')}"
    if k in ("HANG", "CRASH", "RANK_HASH_DIVERGENT"):
        return f"{k}|{f.get('par')}"
    return f"{k}|{f.get('phase')}|{msg}"


def axis_tags(f, scn, tags=None):
    """Config axes that matter for the phase a finding came from, e.g. clip:inf and empty-shard for a clip error."""
    tags = tags or f.get("tags") or {}
    seg = scn["segs"][f.get("seg", 0)] if scn.get("segs") else {}
    ph = f.get("phase") or ""
    out = []
    cc = scn.get("clip")
    if isinstance(cc, (int, float)) and cc:
        cc = {"max": cc}
    if ph.startswith("train:clip") or f["kind"] == "GNORM_MISMATCH":
        if cc:
            out += [f"clip:{cc.get('norm', 2.0)}", f"clip-impl:{cc.get('impl', 'torch')}"]
            if cc.get("foreach") is not None:
                out.append(f"foreach:{cc['foreach']}")
    if ph.startswith("train:opt"):
        o = scn["optim"]
        out.append("opt:" + o["name"] + (":fused" if o.get("fused") else f":foreach={o.get('foreach')}"))
    if ph.startswith(("train:fwd", "train:bwd")) or f["kind"] in ("LOSS_MISMATCH", "STATE_MISMATCH"):
        kinds = sorted({b["t"] for b in scn["model"]["blocks"] if b["t"] in ("moe", "lora", "attn")})
        out += kinds
        if scn.get("ac", "none") != "none":
            out.append("ac:" + scn["ac"])
        if seg.get("mp"):
            out.append("mp")
        if seg.get("tp", {}).get("head") == "col_lp":
            out.append("loss-parallel")
        if scn.get("accum", 1) > 1:
            out.append("accum" + (":no_sync" if scn.get("no_sync") else ""))
        if seg.get("par") == "ep":
            out.append("a2a:" + seg.get("a2a", "functional"))
            out += [t.replace("_", "-") for t in ("zero_send", "zero_recv") if tags.get(t)]
    if ph in (
        "load",
        "save",
        "resave",
        "async_wait",
        "presave",
        "post_save_barrier",
        "train_during_async",
        "gather_after_load",
    ) or f["kind"] in (
        "LOAD_MISMATCH",
        "ROUNDTRIP_MISMATCH",
        "LOAD_EXTRA",
    ):
        sv = seg.get("save", {})
        ld = seg.get("load", {})
        if ph in ("save", "async_wait", "presave", "post_save_barrier", "train_during_async"):
            out.append(f"save:{sv.get('api')}:{sv.get('mode')}:{sv.get('async')}")
        else:
            out.append(f"load:{ld.get('api')}:{ld.get('mode')}")
    if tags.get("empty_shard"):
        out.append("empty-shard")
    return out


def train_signature(f, scn):
    k = f["kind"]
    seg = scn["segs"][f.get("seg", 0)] if scn.get("segs") else {}
    par = f.get("pair") or seg.get("par")
    par = {"fsdp": "fsdp2"}.get(par, par)
    msg = re.sub(r"\d+(\.\d+)?(e-?\d+)?", "N", f.get("msg", ""))[:90]
    parts = ["TRAIN", k, f.get("phase") or "-", par] + axis_tags(f, scn)
    if k in ("ERR", "RANK_DIVERGENT_ERR", "PARTIAL_ERR", "FATAL"):
        parts.append(msg)
    elif k == "HANG":
        parts.append(",".join(f.get("locs", [])) or "timeout")
    elif k in ("LOAD_MISMATCH", "STATE_MISMATCH", "ROUNDTRIP_MISMATCH"):
        parts.append(",".join(f.get("cats", [])))
    return "|".join(str(p) for p in parts)


def main(argv):
    ap = argparse.ArgumentParser(prog="python -m distfuzz.dcp fuzz")
    ap.add_argument("--mode", choices=["guided", "random"], default="guided")
    ap.add_argument("--minutes", type=float, default=60)
    ap.add_argument("--out", default=os.path.join("runs", "dcp"))
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--timeout", type=float, default=120)
    ap.add_argument("--max-w", type=int, default=4)
    ap.add_argument("--profile", choices=["classic", "train"], default="classic")
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    rng = random.Random(a.seed)
    gen = (TrainGen if a.profile == "train" else Gen)(rng, max_w=a.max_w)
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
        tags = {}
        findings, ncov = run_scenario(pools, scn, workdir, a.timeout, tags)
        execs += 1
        segs_total += len(scn["segs"])
        steps_total += sum(s["steps"] for s in scn["segs"])
        added = ncov - cov
        if added:
            cov |= added
            if a.mode == "guided" and not findings:
                corpus.append(scn)
        for f in findings:
            f["tags"] = dict(tags)
            s = signature(f, scn)
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
