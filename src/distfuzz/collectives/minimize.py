from __future__ import annotations

import copy

from .executor import TIMEOUT, Session
from .oracle import classify
from .prog import sanitize


def sigs_of(sess, prog, tries=3):
    out = set()
    for _ in range(tries):
        res = sess.run(prog)
        fs, _ = classify(prog, res)
        out |= {f["sig"] for f in fs}
        if out:
            break
    return out


def minimize(prog, sig, world=4, timeout=TIMEOUT, tries=3, logdir=None):
    sess = Session(world, timeout=timeout, coverage=False, logdir=logdir)
    sess.start()
    try:
        cur = copy.deepcopy(prog)
        n = 2
        while len(cur["calls"]) >= 2:
            chunk = max(1, len(cur["calls"]) // n)
            reduced = False
            for s in range(0, len(cur["calls"]), chunk):
                cand = copy.deepcopy(cur)
                cand["calls"] = cur["calls"][:s] + cur["calls"][s + chunk :]
                sanitize(cand)
                if cand["calls"] and sig in sigs_of(sess, cand, tries):
                    cur, reduced = cand, True
                    n = max(n - 1, 2)
                    break
            if not reduced:
                if chunk == 1:
                    break
                n = min(n * 2, len(cur["calls"]))
        for i, c in enumerate(cur["calls"]):
            for r in list(c.get("div", {})):
                cand = copy.deepcopy(cur)
                del cand["calls"][i]["div"][r]
                if sig in sigs_of(sess, cand, tries):
                    cur = cand
        return cur
    finally:
        sess.stop(hard=True)


REPRO = '''"""Standalone repro emitted by distfuzz. Finding: {sig}

Run (with distfuzz installed): python3 {name}
"""
import json
from distfuzz.collectives.executor import run_once
from distfuzz.collectives.oracle import classify

PROG = json.loads({prog!r})

if __name__ == "__main__":
    res = run_once(PROG, world=PROG["world"], timeout={timeout})
    findings, info = classify(PROG, res)
    print("exec kind:", res["kind"])
    print("reference:", info.get("ref"), info.get("ref_why"))
    for f in findings:
        print(f["sig"])
        print(f["detail"][:2000])
'''
