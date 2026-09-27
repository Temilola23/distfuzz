import json
import random
import time

import pytest
import torch

from distfuzz.collectives import oracle
from distfuzz.collectives import prog as P
from distfuzz.collectives.desc import FAULT_OPS
from distfuzz.collectives.reference import run_reference


def T(dtype="float32", shape=(2,), layout="contig", seed=1, kind="int"):
    return {"dtype": dtype, "shape": list(shape), "layout": layout, "seed": seed, "kind": kind}


def call(op, args, rets=None, div=None):
    return {"op": op, "args": args, "rets": rets or {}, "div": div or {}}


def _ops(progs):
    return {c["op"] for p in progs for c in p["calls"]}


def _well_formed(p):
    defined = set()
    for c in p["calls"]:
        if P.refs(c) - defined:
            return False
        defined |= set(c.get("rets", {}).values())
    return True


def test_fault_ops_only_generated_in_fault_mode():
    rng = random.Random(0)
    plain = [P.Generator(4, rng).generate() for _ in range(300)]
    assert not _ops(plain) & set(FAULT_OPS)
    rng = random.Random(0)
    faulty = [P.Generator(4, rng, fault=True).generate() for _ in range(300)]
    assert set(FAULT_OPS) <= _ops(faulty)


@pytest.mark.parametrize("seed", range(20))
def test_fault_mode_generate_and_mutate_keep_programs_well_formed(seed):
    rng = random.Random(seed)
    g, m = P.Generator(4, rng, fault=True), P.Mutator(4, rng, fault=True)
    corpus = [g.generate() for _ in range(3)]
    assert all(_well_formed(p) for p in corpus)
    for _ in range(20):
        q = m.mutate(rng.choice(corpus), corpus)
        assert _well_formed(q), json.dumps(q)[:300]
        names = [v for c in q["calls"] for v in c["rets"].values()]
        assert len(names) == len(set(names))
        json.dumps(q)


def test_fault_work_ops_consume_or_create_async_work():
    rng = random.Random(3)
    g = P.Generator(4, rng, fault=True)
    st = P.State(4)
    calls = g.gen_fault_call(st, "drop_work")
    producer = calls[-2]
    assert producer["op"] == "all_reduce" and producer["args"]["async_op"]
    assert calls[-1]["args"]["w"] == producer["rets"]["w"]
    assert st.works == []


@pytest.mark.parametrize("op", FAULT_OPS)
def test_fault_programs_are_uncertain_and_racy(op):
    rng = random.Random(1)
    st = P.State(4)
    p = {"world": 4, "calls": P.Generator(4, rng, fault=True).gen_fault_call(st, op)}
    ref = run_reference(p)
    assert ref.status == "uncertain" and ref.racy
    assert p["calls"][ref.at]["op"] == op


def test_fault_program_values_are_not_checked():
    p = {
        "world": 4,
        "calls": [
            call("tensor", {"spec": T()}, {"out": "t1"}),
            call("all_reduce", {"t": "t1", "op": "SUM", "group": "world", "async_op": False}),
            call("async_free", {"t": "t1", "op": "SUM", "group": "world"}),
        ],
    }
    rs = [{"outputs": {"t1": torch.zeros(0)}, "lists": {}, "exc": None, "wait_exc": []}] * 4
    findings, info = oracle.classify(p, {"kind": "ok", "results": rs})
    assert findings == [] and not info["checked"]


def _crash_prog(victim=2):
    return {"world": 4, "calls": [call("crash_rank", {"victim": victim, "group": "world", "when": "during_async"})]}


def _crash(exitcodes):
    return {"kind": "crash", "crash_log": "", "exitcodes": exitcodes, "results": [None] * 4}


def test_victim_sigkill_is_not_a_finding():
    findings, _ = oracle.classify(_crash_prog(), _crash([0, 0, -9, 0]))
    assert findings == []


def test_survivor_segv_after_peer_kill_is_a_finding():
    findings, _ = oracle.classify(_crash_prog(), _crash([0, -11, -9, 0]))
    assert [f["sig"] for f in findings] == ["KILL_SURVIVOR_CRASH|SIGSEGV"]


def test_survivor_hang_after_peer_kill_is_its_own_kind():
    findings, _ = oracle.classify(_crash_prog(), {"kind": "hang", "stuck_ranks": [1], "results": [None] * 4})
    assert findings[0]["kind"] == "KILL_SURVIVOR_HANG"


def test_crash_without_crash_rank_is_still_a_crash():
    findings, _ = oracle.classify({"world": 4, "calls": []}, _crash([-6, 0, 0, 0]))
    assert findings[0]["kind"] == "CRASH"


@pytest.mark.multirank
def test_inflight_set_aborts_the_process():
    from distfuzz.collectives.executor import Session

    # One in-flight set_ aborts a rank only now and then, so each program repeats it 40 times.
    calls = []
    for i in range(40):
        calls += [
            call("tensor", {"spec": T("int32", (18,), seed=i)}, {"out": f"t{i}"}),
            call("async_resize", {"t": f"t{i}", "op": "SUM", "group": "world", "how": "set_"}),
        ]
    p = {"world": 2, "calls": calls}
    s = Session(2, timeout=2.0, coverage=False)
    s.start()
    try:
        for _ in range(5):
            res = s.run(p)
            if res["kind"] == "ok":  # the abort can land after the rank already reported
                time.sleep(0.5)
                res = s.run({"world": 2, "calls": []})
            findings, _ = oracle.classify(p, res)
            if any(f["kind"] == "CRASH" for f in findings):
                return
    finally:
        s.stop(hard=True)
    pytest.fail("set_ on tensors with an in-flight all_reduce never aborted in 5 tries")
