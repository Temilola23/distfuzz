from __future__ import annotations

import tempfile

from .executor import Session


def sequence(rec, only_prog=False):
    if only_prog:
        return [rec["prog"]]
    return [*(rec.get("history") or []), rec["prog"]]


def replay_once(seq, world, timeout):
    s = Session(world, timeout=timeout, coverage=False, logdir=tempfile.mkdtemp(prefix="distfuzz-replay-"))
    s.start()
    try:
        for i, p in enumerate(seq):
            res = s.run(p)
            if res["kind"] != "ok":
                return res["kind"], i, {"exitcodes": res.get("exitcodes"), "log": res.get("crash_log") or ""}
        return "ok", None, {}
    finally:
        s.stop(hard=True)


def reproduce(seq, world, timeout, reps, match=None, log=print):
    hits, last = 0, None
    for r in range(reps):
        kind, at, detail = replay_once(seq, world, timeout)
        log(f"  rep {r}: {kind} at {at} exit={detail.get('exitcodes')}")
        if kind in ("crash", "hang") and (match is None or match in detail.get("log", "")):
            hits += 1
            last = (kind, at, detail)
    return hits, last


def trim_history(seq, world, timeout, reps, match=None, log=print):
    # syzkaller-style crash-log bisection, simplified: drop leading programs while it still reproduces
    cur = list(seq)
    while len(cur) > 1 and reproduce(cur[1:], world, timeout, reps, match, log)[0]:
        cur = cur[1:]
    return cur
