import json
import os
import random
import re
import time

from distfuzz.dcp.gen import Gen
from distfuzz.dcp.standalone import run_standalone

HERE = os.path.dirname(os.path.abspath(__file__))


def key(f):
    k = f["kind"]
    if k in ("ERR", "RANK_DIVERGENT_ERR", "FATAL"):
        return (k, f.get("phase"), re.sub(r"\d+", "N", f.get("msg", "")).splitlines()[0][:80])
    return (k,)


def matches(target, fs):
    return any(key(f) == target for f in fs)


def minimize(scn, target, budget_s=900):
    g = Gen(random.Random(0))
    t0, runs = time.time(), 0
    changed = True
    while changed and time.time() - t0 < budget_s:
        changed = False
        for cand in g.simplifications(scn):
            if time.time() - t0 > budget_s:
                break
            runs += 1
            if matches(target, run_standalone(cand, timeout=90, verbose=False)):
                scn, changed = cand, True
                break
    return scn, runs


def emit(scn, target, path):
    src = open(os.path.join(HERE, "runtime.py")).read()
    sa = open(os.path.join(HERE, "standalone.py")).read()
    sa = sa.split("if __name__ ==")[0]
    body = (
        f"# repro for {target}\n# run: python {os.path.basename(path)}  (exit 1 = reproduced)\n"
        + src
        + "\n"
        + sa.replace('if "run_segment" not in globals():', "if False:")
        + f"\nSCN = {json.dumps(scn, indent=1)}\n\n"
        + "if __name__ == '__main__':\n"
        + "    sys.exit(1 if run_standalone(SCN) else 0)\n"
    )
    body = body.replace("SCN = {", "null, true, false = None, True, False\nSCN = {", 1)
    open(path, "w").write(body)


def main(argv):
    src, out, *want = argv
    os.makedirs(out, exist_ok=True)
    for i, line in enumerate(open(src)):
        rec = json.loads(line)
        if want and not any(w in rec["sig"] for w in want):
            continue
        target = key(rec["finding"])
        scn = rec["scn"]
        scn.pop("id", None)
        fs = run_standalone(scn, timeout=90, verbose=False)
        if not matches(target, fs):
            print(f"[{i}] {rec['sig']}: NOT REPRODUCED standalone ({[key(f) for f in fs]})", flush=True)
            continue
        m, runs = minimize(scn, target)
        name = re.sub(r"[^A-Za-z0-9]+", "_", rec["sig"])[:70] + f"_{i}.py"
        emit(m, target, os.path.join(out, name))
        print(f"[{i}] {rec['sig']}: minimized in {runs} runs -> {name}", flush=True)
