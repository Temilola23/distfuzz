"""A new finding that involved a collective timeout is re-run at the confirm timeout and kept only if it recurs."""

import json
import time

import pytest

from distfuzz.collectives import fuzzer as fuzzer_mod
from distfuzz.collectives.executor import Session
from distfuzz.collectives.fuzzer import Fuzzer

A, B = "VALID_ERROR|all_reduce|RuntimeError|Timed out", "WRONG_RESULT|broadcast|float32"
PROG = {"world": 4, "calls": []}


class Script:
    """Each run returns the next (signatures, timeouts) pair; records the timeout every run asked for."""

    restarts = 0

    def __init__(self, *runs):
        self.runs, self.timeouts = list(runs), []

    def run(self, prog, timeout=None):
        self.timeouts.append(timeout)
        sigs, n = self.runs.pop(0)
        return {"kind": "ok", "results": [{"sigs": sigs, "timeouts": n}]}


def classify(prog, res):
    r = res["results"][0]
    findings = [dict(kind=s.split("|")[0], sig=s, detail="") for s in r["sigs"]]
    return findings, {"ref": "valid", "timeouts": r["timeouts"]}


@pytest.fixture
def fuzz(tmp_path, monkeypatch):
    monkeypatch.setattr(fuzzer_mod, "classify", classify)

    def make(*runs, **kw):
        fz = Fuzzer(str(tmp_path), mode="random", **kw)
        fz.sess, fz.t0 = Script(*runs), 0.0
        return fz

    return make


def test_new_timeout_finding_that_does_not_recur_is_dropped(fuzz, tmp_path):
    fz = fuzz(([A], 1), ([], 0))
    fz.execute(PROG)
    assert fz.sess.timeouts == [None, 4.0]
    assert A not in fz.findings and not list((tmp_path / "findings").iterdir())
    assert fz.stats["confirm_runs"] == 1 and fz.stats["unconfirmed"] == 1
    assert json.loads((tmp_path / "unconfirmed.jsonl").read_text())["sig"] == A


def test_new_timeout_finding_that_recurs_is_kept(fuzz, tmp_path):
    fz = fuzz(([A], 1), ([A], 1))
    fz.execute(PROG)
    assert fz.findings[A]["count"] == 1 and fz.stats["unconfirmed"] == 0
    assert not (tmp_path / "unconfirmed.jsonl").exists()


def test_finding_without_a_timeout_is_not_rerun(fuzz):
    fz = fuzz(([B], 0))
    fz.execute(PROG)
    assert fz.sess.timeouts == [None] and B in fz.findings


def test_known_finding_is_counted_without_a_rerun(fuzz):
    fz = fuzz(([A], 1), ([A], 1), ([A], 1))
    fz.execute(PROG)
    fz.execute(PROG)
    assert fz.sess.timeouts == [None, 4.0, None] and fz.findings[A]["count"] == 2


def test_only_the_new_findings_of_a_program_need_to_recur(fuzz):
    fz = fuzz(([B], 0), ([A, B], 1), ([], 0))
    fz.execute(PROG)
    fz.execute(PROG)
    assert fz.findings[B]["count"] == 2 and A not in fz.findings


def test_confirm_timeout_zero_turns_it_off(fuzz):
    fz = fuzz(([A], 1), confirm_timeout=0)
    fz.execute(PROG)
    assert fz.sess.timeouts == [None] and A in fz.findings


@pytest.mark.multirank
@pytest.mark.parametrize("group", ["world", "g"])
def test_per_program_timeout_reaches_the_ranks_and_extends_the_hang_deadline(group):
    """Rank 3 skips the all_reduce, so the other ranks wait out the timeout, on the world group or on a group the
    program creates. 14 s is past the session's 13 s hang deadline (6 * 0.5 + 10), so the program is only `ok` if
    the deadline follows the per-program timeout."""
    T = {"dtype": "float32", "shape": [2], "layout": "contig", "seed": 1, "kind": "int"}
    prog = {
        "world": 4,
        "calls": [
            {"op": "new_group", "args": {"ranks": [0, 1, 2, 3]}, "rets": {"g": "g"}, "div": {}},
            {"op": "tensor", "args": {"spec": T}, "rets": {"out": "t1"}, "div": {}},
            {
                "op": "all_reduce",
                "args": {"t": "t1", "op": "SUM", "group": group, "async_op": False},
                "rets": {},
                "div": {"3": {"__skip__": True}},
            },
        ],
    }
    s = Session(4, timeout=0.5, coverage=False)
    s.start()
    try:
        t = time.time()
        assert s.run(prog)["kind"] == "ok" and time.time() - t < 5
        t = time.time()
        res = s.run(prog, timeout=14.0)
        assert res["kind"] == "ok" and time.time() - t >= 14
    finally:
        s.stop()
