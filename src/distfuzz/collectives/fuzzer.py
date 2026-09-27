from __future__ import annotations

import hashlib
import json
import os
import random
import time
from collections import deque

from .executor import TIMEOUT, Session
from .oracle import classify
from .prog import Generator, Mutator, dumps, has_divergence


class Fuzzer:
    def __init__(
        self,
        outdir,
        mode="guided",
        world=4,
        timeout=TIMEOUT,
        detail=False,
        seed=0,
        gen_prob=0.1,
        diverge_weight=6,
        fault=False,
    ):
        self.outdir = outdir
        self.mode = mode
        os.makedirs(os.path.join(outdir, "findings"), exist_ok=True)
        os.makedirs(os.path.join(outdir, "corpus"), exist_ok=True)
        os.makedirs(os.path.join(outdir, "logs"), exist_ok=True)
        self.rng = random.Random(seed)
        self.world = world
        self.fault = fault
        self.gen = Generator(world, self.rng, fault=fault)
        self.mut = Mutator(world, self.rng, diverge_weight=diverge_weight, fault=fault)
        self.sess = Session(world, timeout=timeout, detail=detail, coverage=True, logdir=os.path.join(outdir, "logs"))
        # fault programs mutate poorly (a crashing prefix keeps crashing), so generate more often
        self.gen_prob = max(gen_prob, 0.35) if fault else gen_prob
        self.cover = set()
        self.corpus = []
        self.findings = {}
        self.stats = dict(
            execs=0,
            crash=0,
            hang=0,
            ok=0,
            valid=0,
            invalid=0,
            uncertain=0,
            racy=0,
            exc_progs=0,
            timeouts=0,
            detail_mismatch=0,
            divergent=0,
            restarts=0,
            checked=0,
            oom=0,
            wall_exec=0.0,
        )
        self.t0 = None
        self.history = deque(maxlen=30)
        self.statf = open(os.path.join(outdir, "stats.jsonl"), "a")

    def pick(self):
        if self.mode == "random" or not self.corpus or self.rng.random() < self.gen_prob:
            return self.gen.generate()
        return self.mut.mutate(self.rng.choice(self.corpus), self.corpus)

    def record_finding(self, f, prog, res, info):
        sig = f["sig"]
        rec = self.findings.get(sig)
        if rec:
            rec["count"] += 1
            return
        h = hashlib.sha1(sig.encode()).hexdigest()[:10]
        rec = dict(
            sig=sig,
            kind=f["kind"],
            count=1,
            first_t=time.time() - self.t0,
            first_exec=self.stats["execs"],
            id=h,
            detail=f["detail"],
            info=info,
        )
        if f["kind"] in ("CRASH", "HANG", "KILL_SURVIVOR_CRASH", "KILL_SURVIVOR_HANG"):
            rec["history"] = list(self.history)[:-1]
        self.findings[sig] = rec
        with open(os.path.join(self.outdir, "findings", f"{h}.json"), "w") as fh:
            json.dump(dict(rec, prog=prog, kind_exec=res["kind"]), fh, indent=1, default=str)

    def new_cover(self, res):
        new = set()
        for r in res["results"]:
            if r and r.get("cov"):
                new |= r["cov"] - self.cover
        return new

    def step(self):
        prog = self.pick()
        t = time.time()
        self.history.append(prog)
        res = self.sess.run(prog)
        self.stats["wall_exec"] += time.time() - t
        try:
            findings, info = classify(prog, res)
        except Exception as e:  # noqa: BLE001 - a reference-model bug is a finding about the fuzzer
            findings, info = (
                [dict(kind="REFERENCE_BUG", sig=f"REFERENCE_BUG|{type(e).__name__}|{str(e)[:80]}", detail=repr(e))],
                {"ref": "error"},
            )
        self.stats["execs"] += 1
        self.stats["oom" if info.get("oom") else res["kind"]] += 1
        if has_divergence(prog):
            self.stats["divergent"] += 1
        new = self.new_cover(res) if res["kind"] == "ok" else set()
        self.cover |= new
        ref = info.get("ref")
        if ref in ("valid", "invalid", "uncertain"):
            self.stats[ref] += 1
        for k in ("racy", "checked"):
            self.stats[k] += bool(info.get(k))
        self.stats["exc_progs"] += bool(info.get("exc"))
        self.stats["timeouts"] += info.get("timeouts", 0)
        self.stats["detail_mismatch"] += info.get("detail_mismatch", 0)
        for f in findings:
            self.record_finding(f, prog, res, info)
        if res["kind"] != "ok":
            self.history.clear()
        if self.mode == "guided" and new:
            self.corpus.append(prog)
            with open(os.path.join(self.outdir, "corpus", f"{len(self.corpus):05d}.json"), "w") as fh:
                fh.write(dumps(prog))

    def snapshot(self):
        s = dict(
            self.stats,
            t=round(time.time() - self.t0, 1),
            cover=len(self.cover),
            corpus=len(self.corpus),
            findings=len(self.findings),
            restarts=self.sess.restarts,
        )
        self.statf.write(json.dumps(s) + "\n")
        self.statf.flush()
        return s

    def run(self, seconds, report_every=30):
        self.t0 = time.time()
        self.sess.start()
        last = self.t0
        try:
            while time.time() - self.t0 < seconds:
                self.step()
                if time.time() - last >= report_every:
                    last = time.time()
                    s = self.snapshot()
                    print(
                        f"[{self.mode}] t={s['t']}s execs={s['execs']} exec/s={s['execs'] / max(s['t'], 1e-9):.2f} "
                        f"cover={s['cover']} corpus={s['corpus']} findings={s['findings']} "
                        f"hang={s['hang']} crash={s['crash']} oom={s['oom']} checked={s['checked']}",
                        flush=True,
                    )
        finally:
            final = self.snapshot()
            self.sess.stop(hard=True)
            with open(os.path.join(self.outdir, "summary.json"), "w") as fh:
                json.dump(
                    dict(
                        final=final,
                        mode=self.mode,
                        findings=[{k: v for k, v in r.items() if k != "info"} for r in self.findings.values()],
                    ),
                    fh,
                    indent=1,
                    default=str,
                )
        return final
