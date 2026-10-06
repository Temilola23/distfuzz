"""Triage of new C++ coverage, driven through a fake executor whose coverage is scripted per call."""

import pytest

from distfuzz.collectives.fuzzer import Fuzzer


def T(dtype="float32", shape=(2,)):
    return {"dtype": dtype, "shape": list(shape), "layout": "contig", "seed": 1, "kind": "int"}


def C(op, args, rets=None):
    return {"op": op, "args": args, "rets": rets or {}, "div": {}}


TENSOR = C("tensor", {"spec": T()}, {"out": "t1"})
ALL_REDUCE = C("all_reduce", {"t": "t1", "op": "SUM", "group": "world", "async_op": False})
BARRIER = C("barrier", {"group": "world", "async_op": False})
BROADCAST = C("broadcast", {"t": "t1", "root": 0, "root_mode": "global", "group": "world", "async_op": False})
PROG = {"world": 4, "calls": [TENSOR, ALL_REDUCE, BARRIER, BROADCAST]}

EDGES = {"all_reduce": {100}, "broadcast": {200}, "barrier": {300}}


class FakeSession:
    """Rank 0 reports the scripted C++ features of each call, plus `extra` on the first run only."""

    def __init__(self, extra=frozenset(), py=frozenset()):
        self.extra, self.py, self.runs, self.restarts = set(extra), set(py), [], 0

    def run(self, prog):
        cov = set().union(*(EDGES.get(c["op"], set()) for c in prog["calls"]))
        if not self.runs:
            cov |= self.extra | self.py
        self.runs.append([c["op"] for c in prog["calls"]])
        rank = {"outputs": {}, "lists": {}, "exc": None, "wait_exc": []}
        return {"kind": "ok", "results": [dict(rank, cov=cov)] + [dict(rank) for _ in range(3)]}


def fuzzer(tmp_path, sess, mode="guided", signal="cxx"):
    fz = Fuzzer(str(tmp_path), mode=mode, signal=signal)
    fz.sess, fz.t0 = sess, 0.0
    fz.pick = lambda: PROG
    return fz


def test_features_missing_from_a_rerun_are_dropped(tmp_path):
    fz = fuzzer(tmp_path, FakeSession(extra={999}))
    fz.step()
    assert fz.cover == {100, 200, 300}
    assert fz.stats["unstable"] == 1


def test_all_unstable_admits_nothing(tmp_path):
    fz = fuzzer(tmp_path, FakeSession(extra={999}))
    fz.cover = {100, 200, 300}
    fz.step()
    assert fz.cover == {100, 200, 300} and fz.corpus == []


def test_corpus_program_keeps_only_calls_the_new_coverage_needs(tmp_path):
    fz = fuzzer(tmp_path, FakeSession())
    fz.cover = {100, 300}  # only broadcast's edge is new
    fz.step()
    assert fz.cover == {100, 200, 300}
    assert [c["op"] for c in fz.corpus[0]["calls"]] == ["tensor", "broadcast"]
    assert fz.stats["triage_removed"] == 2


def test_every_triage_run_is_charged_to_the_budget(tmp_path):
    sess = FakeSession()
    fz = fuzzer(tmp_path, sess)
    fz.cover = {100, 300}
    fz.step()
    assert fz.stats["execs"] == len(sess.runs) == 1 + 2 + 3  # first run, two re-runs, three removal attempts


def test_python_coverage_is_kept_without_rerun(tmp_path):
    fz = fuzzer(tmp_path, FakeSession(py={"torch/distributed/x.py:1"}), signal="py+cxx")
    fz.cover = {100, 200, 300}
    fz.step()
    assert fz.cover == {100, 200, 300, "torch/distributed/x.py:1"}
    assert len(fz.corpus) == 1


@pytest.mark.parametrize("mode,signal", [("random", "cxx"), ("guided", "py")])
def test_no_triage_without_guided_cxx(tmp_path, mode, signal):
    sess = FakeSession(extra={999})
    fz = fuzzer(tmp_path, sess, mode=mode, signal=signal)
    fz.step()
    assert len(sess.runs) == 1 and 999 in fz.cover


def test_calls_the_new_coverage_needs_are_never_removed(tmp_path):
    fz = fuzzer(tmp_path, FakeSession())
    fz.cover = {300}  # all_reduce and broadcast both contribute
    fz.step()
    assert [c["op"] for c in fz.corpus[0]["calls"]] == ["tensor", "all_reduce", "broadcast"]


class NoisySession(FakeSession):
    """Every program also reports a varying slice of harness noise, as background threads do."""

    NOISE = [5000, 5001, 5002, 5003]

    def run(self, prog):
        res = super().run(prog)
        res["results"][0]["cov"] |= set(self.NOISE[: len(self.runs) % 4 + 1])
        return res


def test_calibration_learns_the_harness_noise(tmp_path):
    fz = fuzzer(tmp_path, NoisySession())
    fz.calibrate()
    assert fz.ignore == set(NoisySession.NOISE)
    fz.step()
    assert fz.cover == {100, 200, 300}


def test_without_calibration_noise_would_enter_the_corpus(tmp_path):
    """Control for the test above: the noise really is new coverage unless it is ignored."""
    fz = fuzzer(tmp_path, NoisySession())
    fz.step()
    assert fz.cover & set(NoisySession.NOISE)


def test_features_that_failed_triage_are_not_triaged_again(tmp_path):
    sess = FakeSession(extra={999})
    fz = fuzzer(tmp_path, sess)
    fz.cover = {100, 200, 300}
    fz.step()
    assert 999 in fz.ignore
    sess.runs = []  # the next run reports 999 again
    fz.step()
    assert len(sess.runs) == 1
