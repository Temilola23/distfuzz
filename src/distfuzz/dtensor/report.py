import argparse
import json
from collections import Counter


def at(stats, t):
    best = stats[0]
    for s in stats:
        if s["t"] <= t:
            best = s
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--at", type=float)
    a = ap.parse_args()
    for d in a.dirs:
        stats = [json.loads(line) for line in open(f"{d}/stats.jsonl")]
        finds = [json.loads(line) for line in open(f"{d}/findings.jsonl")]
        last = stats[-1]
        print(f"== {d}")
        print(
            f"  time={last['t']:.0f}s execs={last['execs']} execs/s={last['eps']} steps={last['steps']} "
            f"restarts={last['restarts']} cov_lines={last['cov']} corpus={last['corpus']} unique_sigs={last['sigs']}"
        )
        for t in (60, 300, 600, 900, 1200, 1500, 1800, 2100):
            if t <= last["t"] + 30:
                s = at(stats, t)
                print(f"    t<={t:5d}s  execs={s['execs']:6d}  cov={s['cov']:5d}  sigs={s['sigs']}")
        if a.at:
            s = at(stats, a.at)
            nb = sum(1 for f in finds if f["t"] <= a.at and not f["sig"].startswith("DIST_ERR"))
            print(f"  @{a.at:.0f}s: cov={s['cov']} sigs={s['sigs']} non-DIST_ERR sigs={nb}")
        c = Counter(f["sig"].split("|")[0] for f in finds)
        print("  unique signatures by kind:", dict(c.most_common()))


if __name__ == "__main__":
    main()
