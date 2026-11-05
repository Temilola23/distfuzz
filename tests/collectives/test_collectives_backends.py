"""--backend/--device options and the cross-backend oracle, exercised on CPU without starting ranks."""

import random
import sys

import pytest
import torch

from distfuzz.collectives import __main__ as cli
from distfuzz.collectives import tensors as T
from distfuzz.collectives.desc import NCCL_EXCLUDED, NCCL_EXCLUDED_REDOPS
from distfuzz.collectives.executor import NCCL_CONFIRM_TIMEOUT, NCCL_TIMEOUT, Session, check_backend
from distfuzz.collectives.fuzzer import Fuzzer
from distfuzz.collectives.oracle import classify, cross_compare, outcome
from distfuzz.collectives.prog import Generator, Mutator
from distfuzz.collectives.reference import run_reference


def C(op, args, rets=None, div=None):
    return {"op": op, "args": args, "rets": rets or {}, "div": div or {}}


SPEC = {"dtype": "int64", "shape": [3], "layout": "contig", "seed": 4, "kind": "int"}
VALID = {
    "world": 2,
    "calls": [
        C("tensor", {"spec": SPEC}, {"out": "t1"}),
        C("all_reduce", {"t": "t1", "op": "SUM", "group": "world", "async_op": False}),
    ],
}
# rank 1 broadcasts from a different root: the reference calls it invalid (a cross-rank mismatch)
INVALID = {
    "world": 2,
    "calls": [
        C("tensor", {"spec": SPEC}, {"out": "t1"}),
        C(
            "broadcast",
            {"t": "t1", "root": 0, "root_mode": "global", "group": "world", "async_op": False},
            div={"1": {"root": 1}},
        ),
    ],
}


def ok_run(prog, tweak=None):
    """A run whose outputs are exactly what the reference expects; `tweak(rank, outputs)` may change them."""
    ref = run_reference(prog)
    results = []
    for r in range(prog["world"]):
        outs = {v: val.v.clone() for v, val in ref.t[r].items() if val.v is not None}
        if tweak:
            tweak(r, outs)
        results.append({"exc": None, "wait_exc": [], "outputs": outs, "lists": {}, "guard_bad": 0})
    return {"kind": "ok", "results": results}


def raised_run(prog, msg="boom", idx=1):
    res = ok_run(prog)
    res["results"][0]["exc"] = {"idx": idx, "op": prog["calls"][idx]["op"], "type": "RuntimeError", "msg": msg}
    return res


def test_nccl_generator_and_mutator_never_emit_excluded_ops_or_redops():
    rng = random.Random(7)
    gen, mut = Generator(2, rng, backend="nccl"), Mutator(2, rng, backend="nccl")
    seen_ops, seen_redops = set(), set()
    p = gen.generate()
    for _ in range(600):
        p = mut.mutate(p) if rng.random() < 0.5 else gen.generate()
        for c in p["calls"]:
            seen_ops.add(c["op"])
            for src in [c["args"], *c.get("div", {}).values()]:
                if src.get("op") in ("SUM", "PRODUCT", "MIN", "MAX", "BAND", "BOR", "BXOR", "AVG"):
                    seen_redops.add(src["op"])
    assert not seen_ops & set(NCCL_EXCLUDED)
    assert not seen_redops & set(NCCL_EXCLUDED_REDOPS)
    assert {"all_reduce", "broadcast", "p2p"} <= seen_ops and "SUM" in seen_redops


def test_gloo_generator_is_unchanged_by_the_backend_option():
    a = [Generator(4, random.Random(3)).generate() for _ in range(1)]
    rng = random.Random(3)
    g = Generator(4, rng, backend="gloo")
    assert [g.generate()] == a
    assert "monitored_barrier" in g.ops and ("BAND", 5) in g.redops


def test_fault_ops_are_refused_on_nccl():
    with pytest.raises(ValueError, match="nccl"):
        Generator(2, random.Random(0), fault=True, backend="nccl")


@pytest.mark.parametrize("backend,device", [("gloo", "cuda"), ("nccl", "cpu"), ("mpi", "cpu")])
def test_unsupported_backend_device_pairs_are_refused(backend, device):
    with pytest.raises(ValueError):
        check_backend(backend, device)
    with pytest.raises(ValueError):
        Session(2, backend=backend, device=device)


def test_nccl_fuzzer_defaults_to_the_longer_timeouts(tmp_path):
    fz = Fuzzer(str(tmp_path), mode="random", backend="nccl", world=2)
    assert fz.sess.device == "cuda" and fz.sess.cfg["timeout"] == NCCL_TIMEOUT
    assert fz.confirm_timeout == NCCL_CONFIRM_TIMEOUT
    assert Fuzzer(str(tmp_path / "g"), mode="random").sess.cfg["timeout"] == 0.5


@pytest.mark.parametrize(
    "argv,err",
    [
        (["--backend", "nccl", "--device", "cpu"], "backend/device"),
        (["--device", "cuda"], "backend/device"),
        (["--backend", "nccl", "--fault"], "--fault"),
        (["--cross-backend", "nccl", "--fault"], "--fault"),
    ],
)
def test_cli_rejects_bad_combinations(monkeypatch, capsys, tmp_path, argv, err):
    monkeypatch.setattr(sys, "argv", ["x", "fuzz", "--out", str(tmp_path), *argv])
    with pytest.raises(SystemExit):
        cli.main()
    assert err in capsys.readouterr().err


def test_make_on_cpu_device_matches_default():
    for layout in T.LAYOUTS:
        spec = dict(SPEC, shape=[2, 3], layout=layout)
        a, b = T.make(spec, 1, 5), T.make(spec, 1, 5, device="cpu")
        assert torch.equal(a.base, b.base) and a.view.stride() == b.view.stride() and b.guards_ok()


def test_nccl_signatures_name_the_backend():
    res = ok_run(VALID, lambda r, o: o["t1"].add_(1))
    gloo, _ = classify(VALID, res)
    nccl, _ = classify(VALID, res, backend="nccl")
    assert [f["sig"] for f in nccl] == ["nccl:" + f["sig"] for f in gloo] and gloo


def test_nccl_silent_accept_on_a_cross_rank_mismatch_is_counted_not_reported():
    res = ok_run(INVALID)
    assert run_reference(INVALID).status == "invalid"
    assert [f["kind"] for f in classify(INVALID, res)[0]] == ["SILENT_ACCEPT"]
    fs, info = classify(INVALID, res, backend="nccl")
    assert fs == [] and info["nccl_unchecked"] == 1


def test_nccl_watchdog_death_is_a_hang():
    log = "[rank 0] [PG 1 Rank 0] Watchdog caught collective operation timeout: WorkNCCL(SeqNum=3) ran for 5001 ms"
    res = {"kind": "crash", "results": [{"died": True}] * 2, "exitcodes": [-6, -6], "crash_log": log}
    assert outcome(res, "nccl")[0] == "hang" and outcome(res, "gloo")[0] == "crash"
    assert [f["sig"] for f in classify(VALID, res, backend="nccl")[0]] == ["nccl:HANG|valid"]
    fs, info = classify(INVALID, res, backend="nccl")
    assert fs == [] and info["nccl_unchecked"] == 1
    assert classify(VALID, res)[0][0]["kind"] == "CRASH"


def test_nccl_timeout_wording_counts_as_a_timeout():
    res = raised_run(VALID, "WorkNCCL(SeqNum=2, OpType=ALLREDUCE) ran for 5000 milliseconds before timing out.")
    assert classify(VALID, res, backend="nccl")[1]["timeouts"] == 1
    assert classify(VALID, res)[1]["timeouts"] == 0


def xc(prog, a, b):
    return cross_compare(prog, run_reference(prog), [("gloo", a), ("nccl", b)])


def test_cross_compare_agreeing_runs_have_no_finding():
    assert xc(VALID, ok_run(VALID), ok_run(VALID)) == []


def test_cross_compare_reports_a_value_difference_with_both_backends_in_the_signature():
    bad = ok_run(VALID, lambda r, o: o["t1"].add_(r))  # rank 1 only
    (f,) = xc(VALID, ok_run(VALID), bad)
    assert f["sig"] == "XBACKEND_VALUE|all_reduce|torch.int64|gloo!=nccl" and "(1, 't1')" in f["detail"]


def test_cross_compare_is_exact_on_integers_and_tolerant_on_floats():
    fprog = {"world": 2, "calls": [C("tensor", {"spec": dict(SPEC, dtype="float32", kind="randn")}, {"out": "t1"})]}
    fprog["calls"].append(VALID["calls"][1])
    near = ok_run(fprog, lambda r, o: o["t1"].mul_(1 + 1e-6))
    assert xc(fprog, ok_run(fprog), near) == []
    off_by_one = ok_run(VALID, lambda r, o: o["t1"].add_(1))
    assert len(xc(VALID, ok_run(VALID), off_by_one)) == 1


def test_cross_compare_reports_an_outcome_difference():
    (f,) = xc(VALID, ok_run(VALID), raised_run(VALID))
    assert f["sig"] == "XBACKEND_OUTCOME|all_reduce|gloo=ok|nccl=raise"
    timeout = raised_run(VALID, "Timed out waiting 500ms")
    assert xc(VALID, timeout, ok_run(VALID))[0]["sig"] == "XBACKEND_OUTCOME|all_reduce|gloo=timeout|nccl=ok"


def test_cross_compare_skips_programs_the_backends_may_legally_disagree_on():
    assert xc(INVALID, raised_run(INVALID), ok_run(INVALID)) == []


def test_cross_compare_leaves_crashes_and_hangs_to_the_per_backend_oracle():
    hang = {"kind": "hang", "results": [None, None], "stuck_ranks": [0]}
    assert xc(VALID, ok_run(VALID), hang) == [] and xc(VALID, hang, raised_run(VALID)) == []


class FakeSession:
    restarts = 0

    def __init__(self, res):
        self.res, self.calls = res, 0

    def run(self, prog, timeout=None):
        self.calls += 1
        return self.res


def test_fuzzer_cross_mode_tags_every_signature_and_records_the_disagreement(tmp_path):
    fz = Fuzzer(str(tmp_path), mode="random", world=2, cross="gloo", confirm_timeout=0)
    bad = ok_run(VALID, lambda r, o: o["t1"].add_(1))
    fz.sess, fz.sess2, fz.t0 = FakeSession(bad), FakeSession(ok_run(VALID)), 0.0
    fz.execute(VALID)
    assert fz.sess.calls == fz.sess2.calls == 1
    assert set(fz.findings) == {
        "gloo:WRONG_RESULT|all_reduce|torch.int64",
        "XBACKEND_VALUE|all_reduce|torch.int64|gloo!=gloo",
    }
    assert fz.stats["xbackend_checked"] == 1
    fz.pool.shutdown()


@pytest.mark.multirank
def test_cross_mode_end_to_end_two_gloo_sessions_agree(tmp_path):
    """The cross plumbing with real ranks: gloo against gloo must agree on a valid program and on a generated batch."""
    fz = Fuzzer(str(tmp_path), mode="random", world=2, cross="gloo", timeout=3.0, cross_timeout=3.0, seed=11)
    fz.t0 = 0.0
    fz.sess.start()
    fz.sess2.start()
    try:
        res, findings, info = fz.run_classify(VALID)
        assert res["kind"] == "ok" and findings == [] and info["xbackend_checked"]
        xsigs = []
        for _ in range(15):
            p = fz.gen.generate()
            xsigs += [f["sig"] for f in fz.run_classify(p)[1] if f["kind"] == "XBACKEND_VALUE"]
        assert xsigs == []
    finally:
        fz.sess.stop(hard=True)
        fz.sess2.stop(hard=True)
        fz.pool.shutdown()
