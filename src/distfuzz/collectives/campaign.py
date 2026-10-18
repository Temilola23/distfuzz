"""An unattended campaign: fuzzer rounds with successive seeds, alternating random and guided, merged by signature.

Each round is an ordinary `fuzz` run in its own directory, so its findings feed `minimize`, `replay` and the repro
pipeline unchanged. campaign.json records the rounds that finished; running the same command again after an
interruption skips them and starts the unfinished round over, and a larger --rounds extends a finished campaign.
"""

from __future__ import annotations

import json
import os
import shutil

from .executor import TIMEOUT
from .fuzzer import Fuzzer

MODES = ("random", "guided")


def round_dir(out, i):
    return os.path.join(out, f"round_{i:02d}")


def load_known(path):
    """Signatures from a JSON list, or from an earlier campaign's index.json (a dict keyed by signature)."""
    with open(path) as fh:
        return set(json.load(fh))


def write_json(path, data):
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(data, fh, indent=1)
    os.replace(tmp, path)


def merge(out, rounds, known):
    """One record per signature across the finished rounds, pointing at the first finding file that has it."""
    index = {}
    for i in rounds:
        with open(os.path.join(round_dir(out, i), "summary.json")) as fh:
            findings = json.load(fh)["findings"]
        for f in findings:
            rec = index.setdefault(
                f["sig"],
                dict(
                    kind=f["kind"],
                    new=f["sig"] not in known,
                    count=0,
                    rounds=[],
                    first=os.path.join(os.path.basename(round_dir(out, i)), "findings", f"{f['id']}.json"),
                ),
            )
            rec["count"] += f["count"]
            rec["rounds"].append(i)
    return index


def report(state, index):
    rounds = state["rounds"].values()
    lines = [
        "# Campaign",
        "",
        f"{len(rounds)} rounds, {sum(r['execs'] for r in rounds)} programs, {len(index)} signatures, "
        f"{sum(r['new'] for r in index.values())} not in the known list.",
        "",
        "| new | signature | rounds | count | first finding |",
        "|---|---|---|---|---|",
    ]
    for sig, r in sorted(index.items(), key=lambda kv: (not kv[1]["new"], kv[1]["rounds"][0], kv[0])):
        cell = sig.replace("|", "\\|")
        lines.append(f"| {'yes' if r['new'] else ''} | `{cell}` | {len(r['rounds'])} | {r['count']} | {r['first']} |")
    return "\n".join(lines) + "\n"


def run(out, rounds=6, minutes=10, world=4, timeout=TIMEOUT, seed=0, signal="py", known=frozenset()):
    os.makedirs(out, exist_ok=True)
    statef = os.path.join(out, "campaign.json")
    config = dict(minutes=minutes, world=world, timeout=timeout, seed=seed, signal=signal)
    state = {"config": config, "rounds": {}}
    if os.path.exists(statef):
        with open(statef) as fh:
            state = json.load(fh)
        if state["config"] != config:
            raise ValueError(f"{out} holds a campaign run with {state['config']}, not {config}")
    for i in range(rounds):
        if str(i) in state["rounds"]:
            continue
        d = round_dir(out, i)
        shutil.rmtree(d, ignore_errors=True)  # left by an interrupted attempt
        mode = MODES[i % len(MODES)]
        print(f"[campaign] round {i + 1}/{rounds}: {mode}, seed {seed + i}", flush=True)
        final = Fuzzer(d, mode=mode, world=world, timeout=timeout, seed=seed + i, signal=signal).run(minutes * 60)
        state["rounds"][str(i)] = dict(mode=mode, seed=seed + i, execs=final["execs"], t=final["t"])
        write_json(statef, state)
    index = merge(out, sorted(map(int, state["rounds"])), known)
    write_json(os.path.join(out, "index.json"), index)
    with open(os.path.join(out, "SUMMARY.md"), "w") as fh:
        fh.write(report(state, index))
    return dict(
        rounds=len(state["rounds"]),
        execs=sum(r["execs"] for r in state["rounds"].values()),
        signatures=len(index),
        new=sorted(sig for sig, r in index.items() if r["new"]),
    )
