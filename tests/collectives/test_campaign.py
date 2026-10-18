"""Campaign rounds, merging and resume, run through the real Fuzzer with a fake executor and scripted findings."""

import json

import pytest

from distfuzz.collectives import campaign
from distfuzz.collectives import fuzzer as fuzzer_mod

A, B, C = "WRONG_RESULT|all_reduce|float32", "HANG|all_gather", "CRASH|broadcast|SIGSEGV"


class Script:
    """Round i (the i-th Session started) reports the signatures in sigs[i] on every program, and
    `interrupt_round`, if set, raises KeyboardInterrupt on its first program."""

    def __init__(self, sigs, interrupt_round=None):
        self.sigs, self.interrupt_round, self.started = sigs, interrupt_round, []

    def session(self, *args, **kwargs):
        script, i = self, len(self.started)
        self.started.append(kwargs.get("logdir"))

        class Session:
            restarts = 0

            def start(self):
                pass

            def stop(self, hard=False):
                pass

            def run(self, prog):
                if i == script.interrupt_round:
                    raise KeyboardInterrupt
                return {"kind": "ok", "results": [{"sigs": script.sigs[i]}]}

        return Session()


def classify(prog, res):
    sigs = res["results"][0]["sigs"]
    return [dict(kind=s.split("|")[0], sig=s, detail="") for s in sigs], {"ref": "valid"}


@pytest.fixture
def script(monkeypatch):
    def install(sigs, interrupt_round=None):
        s = Script(sigs, interrupt_round)
        monkeypatch.setattr(fuzzer_mod, "Session", s.session)
        monkeypatch.setattr(fuzzer_mod, "classify", classify)
        return s

    return install


def go(out, rounds, known=frozenset(), **kw):
    return campaign.run(str(out), rounds=rounds, minutes=0.002, known=known, **kw)


def test_rounds_alternate_modes_and_seeds(tmp_path, script):
    script([[A], [A], [A]])
    go(tmp_path, 3, seed=7)
    state = json.loads((tmp_path / "campaign.json").read_text())
    assert [(r["mode"], r["seed"]) for r in state["rounds"].values()] == [
        ("random", 7),
        ("guided", 8),
        ("random", 9),
    ]


def test_findings_merge_by_signature_across_rounds(tmp_path, script):
    script([[A], [A, B], []])
    res = go(tmp_path, 3)
    index = json.loads((tmp_path / "index.json").read_text())
    assert set(index) == {A, B}
    assert index[A]["rounds"] == [0, 1] and index[B]["rounds"] == [1]
    for sig in (A, B):
        rounds = index[sig]["rounds"]
        per_round = [json.loads((tmp_path / f"round_{i:02d}" / "summary.json").read_text()) for i in rounds]
        assert index[sig]["count"] == sum(f["count"] for s in per_round for f in s["findings"] if f["sig"] == sig)
        # `first` is the finding file the fuzzer wrote in the earliest round, usable by minimize/replay
        rec = json.loads((tmp_path / index[sig]["first"]).read_text())
        assert rec["sig"] == sig and index[sig]["first"].startswith(f"round_{rounds[0]:02d}/")
    assert res["signatures"] == 2


def test_known_signatures_are_not_new(tmp_path, script):
    script([[A, C]])
    known = tmp_path / "known.json"
    known.write_text(json.dumps([A]))
    res = go(tmp_path / "c", 1, known=campaign.load_known(str(known)))
    assert res["new"] == [C]
    md = (tmp_path / "c" / "SUMMARY.md").read_text()
    assert "CRASH\\|broadcast\\|SIGSEGV" in md  # pipes in signatures must not split the table row


def test_an_earlier_index_is_a_known_list(tmp_path, script):
    script([[A], [A], [A, B]])  # one round for the first campaign, two for the second
    go(tmp_path / "first", 1)
    res = go(tmp_path / "second", 2, known=campaign.load_known(str(tmp_path / "first" / "index.json")))
    assert res["new"] == [B]


def test_interrupted_round_reruns_and_finished_rounds_are_not_counted_twice(tmp_path, script):
    script([[A], [B]], interrupt_round=1)
    with pytest.raises(KeyboardInterrupt):
        go(tmp_path, 2)
    # the interrupted round wrote a summary.json of its own on the way out; it must not count as finished
    assert list(json.loads((tmp_path / "campaign.json").read_text())["rounds"]) == ["0"]
    count_a = json.loads((tmp_path / "round_00" / "summary.json").read_text())["findings"][0]["count"]

    s = script([[B]])  # a fresh process: only round 1 runs
    res = go(tmp_path, 2)
    assert len(s.started) == 1 and s.started[0].endswith("round_01/logs")
    index = json.loads((tmp_path / "index.json").read_text())
    assert index[A]["count"] == count_a and index[A]["rounds"] == [0]
    assert index[B]["rounds"] == [1] and res["rounds"] == 2


def test_more_rounds_extend_a_finished_campaign(tmp_path, script):
    script([[A]])
    go(tmp_path, 1)
    s = script([[B]])
    res = go(tmp_path, 2)
    assert len(s.started) == 1 and res["rounds"] == 2 and res["signatures"] == 2


def test_resume_with_different_settings_is_refused(tmp_path, script):
    script([[A], [A]])
    go(tmp_path, 1)
    with pytest.raises(ValueError, match="holds a campaign"):
        go(tmp_path, 2, seed=1)


@pytest.mark.multirank
def test_real_ranks_end_to_end(tmp_path):
    """The real Fuzzer and executor, so the campaign reads what the fuzzer actually writes."""
    res = campaign.run(str(tmp_path), rounds=2, minutes=0.25, world=2)
    state = json.loads((tmp_path / "campaign.json").read_text())
    assert res["rounds"] == 2 and all(r["execs"] > 0 for r in state["rounds"].values())
    index = json.loads((tmp_path / "index.json").read_text())
    assert res["signatures"] == len(index)
    for rec in index.values():
        assert (tmp_path / rec["first"]).exists()
    assert (tmp_path / "SUMMARY.md").read_text().startswith("# Campaign")
