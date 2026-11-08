import json
import os
import random
import re
import time

from distfuzz.dcp.gen import Gen
from distfuzz.dcp.standalone import run_standalone
from distfuzz.dcp.train_gen import TrainGen

HERE = os.path.dirname(os.path.abspath(__file__))


def key(f):
    k = f["kind"]
    if k in ("ERR", "RANK_DIVERGENT_ERR", "FATAL", "PARTIAL_ERR"):
        return (k, f.get("phase"), re.sub(r"\d+", "N", f.get("msg", "")).splitlines()[0][:80])
    return (k,)


def matches(target, fs):
    return any(key(f) == target for f in fs)


def runner_for(scn, logdir=None):
    """Classic scenarios replay in fresh processes; training-profile scenarios replay on persistent worlds, as in
    the campaign, so PARTIAL_ERR and hang locations come out the same way."""
    if scn.get("profile") != "train":
        return lambda cand: run_standalone(cand, timeout=90, verbose=False), lambda: None
    import tempfile

    from distfuzz.dcp.fuzz import CK_ROOT, Pools, run_scenario

    pools = Pools(logdir or tempfile.mkdtemp(prefix="dcpmin_"), cov=False)
    workdir = os.path.join(CK_ROOT, f"min{os.getpid()}")
    return lambda cand: run_scenario(pools, cand, workdir, timeout=60)[0], pools.close


def minimize(scn, target, budget_s=900, run=None):
    """Greedy: take the first simplification that keeps key(finding) (kind, phase, normalized message), repeat."""
    g = (TrainGen if scn.get("profile") == "train" else Gen)(random.Random(0))
    close = None
    if run is None:
        run, close = runner_for(scn)
    t0, runs = time.time(), 0
    changed = True
    try:
        while changed and time.time() - t0 < budget_s:
            changed = False
            for cand in g.simplifications(scn):
                if time.time() - t0 > budget_s:
                    break
                runs += 1
                if matches(target, run(cand)):
                    scn, changed = cand, True
                    break
    finally:
        if close:
            close()
    return scn, runs


def emit(scn, target, path):
    src = open(os.path.join(HERE, "blocks.py")).read() + "\n" + open(os.path.join(HERE, "runtime.py")).read()
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
        run, close = runner_for(scn, os.path.join(out, "logs"))
        try:
            fs = run(scn)
            if not matches(target, fs):
                print(f"[{i}] {rec['sig']}: NOT REPRODUCED ({[key(f) for f in fs]})", flush=True)
                continue
            m, runs = minimize(scn, target, run=run)
        finally:
            close()
        if m.get("profile") == "train":
            with open(os.path.join(out, "minimized.jsonl"), "a") as fh:
                fh.write(json.dumps(dict(sig=rec["sig"], key=list(target), runs=runs, scn=m)) + "\n")
        name = re.sub(r"[^A-Za-z0-9]+", "_", rec["sig"])[:70] + f"_{i}.py"
        emit(m, target, os.path.join(out, name))
        print(f"[{i}] {rec['sig']}: minimized in {runs} runs -> {name}", flush=True)
