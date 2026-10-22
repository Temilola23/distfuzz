from __future__ import annotations

import hashlib
import json
import os
import random
import time
from collections import deque

from .executor import TIMEOUT, Session
from .oracle import classify
from .prog import Generator, Mutator, dumps, has_divergence, sanitize

# a program with no collective: what it covers is the harness, not the program
NOOP_CALLS = {
    "calls": [
        {
            "op": "tensor",
            "args": {"spec": {"dtype": "float32", "shape": [4], "layout": "contig", "seed": 1, "kind": "int"}},
            "rets": {"out": "t1"},
            "div": {},
        }
    ],
}


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
        signal="py",
        triage_runs=2,
        smash=0,
        gen_diverge=0.0,
    ):
        self.outdir = outdir
        self.mode = mode
        self.signal = signal
        # re-running for stability only means something for C++ counters (see triage_new)
        self.triage = mode == "guided" and "cxx" in signal
        self.triage_runs = triage_runs
        os.makedirs(os.path.join(outdir, "findings"), exist_ok=True)
        os.makedirs(os.path.join(outdir, "corpus"), exist_ok=True)
        os.makedirs(os.path.join(outdir, "logs"), exist_ok=True)
        self.rng = random.Random(seed)
        self.world = world
        self.fault = fault
        self.gen = Generator(world, self.rng, fault=fault)
        self.mut = Mutator(world, self.rng, diverge_weight=diverge_weight, fault=fault)
        self.sess = Session(world, timeout=timeout, detail=detail, coverage=signal, logdir=os.path.join(outdir, "logs"))
        # fault programs mutate poorly (a crashing prefix keeps crashing), so generate more often
        self.gen_prob = max(gen_prob, 0.35) if fault else gen_prob
        # syzkaller's smash (SyzVegas, USENIX Security '21, sec. 2.1): a program that just raised coverage gets
        # `smash` mutations in a row before ordinary picks resume; programs admitted meanwhile wait their turn
        self.smash = smash
        # chance that a generated (not mutated) program is made rank-divergent; the generator never does it itself
        self.gen_diverge = gen_diverge
        self.smashing = deque()
        self.cover = set()
        self.ignore = set()
        self.corpus = []
        self.saved = 0
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
            noise=0,
            unstable=0,
            triage_removed=0,
            smashed=0,
            wall_exec=0.0,
        )
        self.t0 = None
        self.history = deque(maxlen=30)
        self.statf = open(os.path.join(outdir, "stats.jsonl"), "a")

    def pick(self):
        if self.smashing:
            prog, left = self.smashing.popleft()
            if left > 1:
                self.smashing.appendleft((prog, left - 1))
            self.stats["smashed"] += 1
            return self.mut.mutate(prog, self.corpus)
        if self.mode == "random" or not self.corpus or self.rng.random() < self.gen_prob:
            p = self.gen.generate()
            return self.mut.diverge(p) if self.rng.random() < self.gen_diverge else p
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
        return new - self.ignore

    def calibrate(self, settle=5, limit=60):
        """Ignore the C++ features the harness produces on its own (background threads, per-program group setup).

        Runs a program with no collective until its feature union stops growing for `settle` runs in a row.
        """
        noise, quiet = set(), 0
        for _ in range(limit):
            res = self.sess.run({**NOOP_CALLS, "world": self.world})
            if res["kind"] == "ok":
                got = set().union(*(r.get("cov", set()) for r in res["results"] if r))
                quiet = quiet + 1 if got <= noise else 0
                noise |= got
            if quiet >= settle:
                break
        self.ignore |= {f for f in noise if isinstance(f, int)}
        self.stats["noise"] = len(self.ignore)

    def execute(self, prog):
        """Run one program, classify it, record findings and stats; return the result and its new coverage."""
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
        return res, (self.new_cover(res) if res["kind"] == "ok" else set())

    def step(self):
        prog = self.pick()
        _, new = self.execute(prog)
        if not new:
            return
        if self.triage:
            prog, new = self.triage_new(prog, new)
            if not new:
                return
        self.cover |= new
        # every coverage-raising program is saved, so coverage can be re-measured the same way for every mode;
        # only guided mode mutates them
        self.saved += 1
        with open(os.path.join(self.outdir, "corpus", f"{self.saved:05d}.json"), "w") as fh:
            fh.write(dumps(prog))
        if self.mode == "guided":
            self.corpus.append(prog)
            if self.smash:
                self.smashing.append((prog, self.smash))

    def covered(self, prog):
        res, _ = self.execute(prog)
        if res["kind"] != "ok":
            return set()
        return set().union(*(r.get("cov", set()) for r in res["results"] if r))

    def triage_new(self, prog, new):
        """Keep the C++ features of `new` that every re-run reproduces, then drop calls they do not need.

        Python line coverage fires once per rank process, so it cannot be re-checked and is kept as is.
        """
        py = {f for f in new if not isinstance(f, int)}
        stable = new - py
        for _ in range(self.triage_runs):
            if not stable:
                break
            stable &= self.covered(prog)
        flaky = new - py - stable
        self.ignore |= flaky  # don't pay to re-check them next time
        self.stats["unstable"] += len(flaky)
        if stable:
            i = len(prog["calls"]) - 1
            while i >= 0 and len(prog["calls"]) > 1:
                cand = sanitize({**prog, "calls": prog["calls"][:i] + prog["calls"][i + 1 :]})
                if cand["calls"] and stable <= self.covered(cand):
                    self.stats["triage_removed"] += len(prog["calls"]) - len(cand["calls"])
                    prog = cand
                i = min(i - 1, len(prog["calls"]) - 1)
        return prog, stable | py

    def snapshot(self):
        s = dict(
            self.stats,
            t=round(time.time() - self.t0, 1),
            cover=len(self.cover),
            cover_cxx=sum(isinstance(f, int) for f in self.cover),
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
        if "cxx" in self.signal:
            self.calibrate()
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
                        signal=self.signal,
                        findings=[{k: v for k, v in r.items() if k != "info"} for r in self.findings.values()],
                    ),
                    fh,
                    indent=1,
                    default=str,
                )
        return final
