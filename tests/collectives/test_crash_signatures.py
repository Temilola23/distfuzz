"""CRASH signatures come from the crashing program's own log output and do not depend on which rank died."""

import pytest

from distfuzz.collectives import oracle
from distfuzz.collectives.executor import MARK, Session

EMPTY = {"world": 4, "calls": []}


def crash(exitcodes, log=""):
    return {"kind": "crash", "crash_log": log, "exitcodes": exitcodes, "results": [None] * 4}


def sig(res):
    return oracle.classify(EMPTY, res)[0][0]["sig"]


def test_same_signal_on_different_ranks_is_one_signature():
    assert sig(crash([-11, None, None, None])) == sig(crash([None, None, -11, -11])) == "CRASH|valid|SIGSEGV"


def test_different_signals_are_different_signatures():
    assert sig(crash([-6, None, None, None])) != sig(crash([-11, None, None, None]))
    assert sig(crash([-6, -11, None, None])) == "CRASH|valid|SIGABRT+SIGSEGV"


def test_gloo_check_failure_names_the_signature():
    log = "[rank 1] [F1018 ProcessGroupGloo.cpp:2262] Check failed: outputs[0].sizes().vec() == x (2 3 vs. 2). | y"
    assert sig(crash([None, -6, None, None], log)).startswith("CRASH|valid|Check failed: outputs[#].sizes()")


def test_log_tail_keeps_only_the_last_program(tmp_path):
    (tmp_path / "rank0.log").write_text(
        f"{MARK} 1\nterminate called after throwing an instance of 'gloo::EnforceNotMet'\n"
        f"{MARK} 2\nsome output of program 2\n"
    )
    tail = Session(1, logdir=str(tmp_path))._log_tails([0])
    assert "program 2" in tail and "EnforceNotMet" not in tail and MARK not in tail


def test_log_without_a_marker_is_read_whole(tmp_path):
    """A rank that died during start-up never wrote a marker; its whole log is the evidence."""
    (tmp_path / "rank0.log").write_text("Fatal Python error: Segmentation fault\n")
    assert "Fatal Python error" in Session(1, logdir=str(tmp_path))._log_tails([0])


@pytest.mark.multirank
def test_ranks_mark_each_program_in_their_log(tmp_path):
    s = Session(2, timeout=2.0, coverage=False, logdir=str(tmp_path))
    s.start()
    try:
        for _ in range(3):
            assert s.run({"world": 2, "calls": []})["kind"] == "ok"
    finally:
        s.stop(hard=True)
    for r in range(2):
        log = (tmp_path / f"rank{r}.log").read_text()
        assert [ln.split()[-1] for ln in log.splitlines() if ln.startswith(MARK)] == ["1", "2", "3"]
