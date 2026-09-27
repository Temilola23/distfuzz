import argparse
import json
import os
import time

from .executor import TIMEOUT


def main():
    ap = argparse.ArgumentParser(prog="python -m distfuzz.collectives")
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fuzz")
    f.add_argument("--mode", choices=["guided", "random"], default="guided")
    f.add_argument("--minutes", type=float, default=30)
    f.add_argument("--world", type=int, default=4)
    f.add_argument("--timeout", type=float, default=TIMEOUT)
    f.add_argument("--seed", type=int, default=0)
    f.add_argument("--detail", action="store_true")
    f.add_argument("--out", required=True)
    b = sub.add_parser("bench")
    b.add_argument("--n", type=int, default=40)
    b.add_argument("--world", type=int, default=4)
    b.add_argument("--timeout", type=float, default=TIMEOUT)
    b.add_argument("--out", required=True)
    m = sub.add_parser("minimize")
    m.add_argument("finding")
    m.add_argument("--timeout", type=float, default=TIMEOUT)
    a = ap.parse_args()

    if a.cmd == "fuzz":
        from .fuzzer import Fuzzer

        fz = Fuzzer(a.out, mode=a.mode, world=a.world, timeout=a.timeout, detail=a.detail, seed=a.seed)
        print(json.dumps(fz.run(a.minutes * 60)))
    elif a.cmd == "bench":
        import random

        from .executor import Session, run_once
        from .prog import Generator

        rng = random.Random(1)
        g = Generator(a.world, rng)
        progs = [g.generate() for _ in range(a.n)]
        s = Session(a.world, timeout=a.timeout, coverage=False)
        t = time.time()
        s.start()
        startup = time.time() - t
        t = time.time()
        kinds = [s.run(p)["kind"] for p in progs]
        persistent = time.time() - t
        s.stop(hard=True)
        m = min(10, a.n)
        t = time.time()
        for p in progs[:m]:
            run_once(p, world=a.world, timeout=a.timeout)
        fresh = time.time() - t
        out = dict(
            n=a.n,
            world=a.world,
            startup_s=round(startup, 2),
            persistent_exec_per_s=round(a.n / persistent, 2),
            fresh_n=m,
            fresh_exec_per_s=round(m / fresh, 3),
            kinds={k: kinds.count(k) for k in set(kinds)},
        )
        os.makedirs(a.out, exist_ok=True)
        json.dump(out, open(os.path.join(a.out, "bench.json"), "w"), indent=1)
        print(json.dumps(out))
    elif a.cmd == "minimize":
        from .minimize import REPRO, minimize

        rec = json.load(open(a.finding))
        small = minimize(rec["prog"], rec["sig"], world=rec["prog"]["world"], timeout=a.timeout)
        base = os.path.splitext(a.finding)[0]
        json.dump(dict(rec, prog_min=small), open(base + ".min.json", "w"), indent=1, default=str)
        name = base + "_repro.py"
        open(name, "w").write(
            REPRO.format(sig=rec["sig"], name=os.path.basename(name), prog=json.dumps(small), timeout=a.timeout)
        )
        print(f"{len(rec['prog']['calls'])} -> {len(small['calls'])} calls; repro: {name}")


if __name__ == "__main__":
    main()
