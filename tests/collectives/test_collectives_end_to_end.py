import pytest
import torch

from distfuzz.collectives.executor import run_once
from distfuzz.collectives.oracle import classify
from distfuzz.collectives.semantics import salt
from distfuzz.collectives.tensors import expected_initial

pytestmark = pytest.mark.multirank


def T(dtype="float32", shape=(2,), layout="contig", seed=1, kind="int"):
    return {"dtype": dtype, "shape": list(shape), "layout": layout, "seed": seed, "kind": kind}


def C(op, args, rets=None, div=None):
    return {"op": op, "args": args, "rets": rets or {}, "div": div or {}}


def run(calls, timeout=3.0):
    p = {"world": 4, "calls": calls}
    res = run_once(p, world=4, timeout=timeout)
    fs, info = classify(p, res)
    return res, fs, info


def kinds(fs):
    return sorted({f["kind"] for f in fs})


def test_TP_valid_program_has_no_findings():
    """Sanity: a sizeable valid program (subgroups, group roots, uneven a2a, reduce_scatter, async+wait)."""
    calls = [
        C("tensor", {"spec": T("int64", (4, 2))}, {"out": "t1"}),
        C("all_reduce", {"t": "t1", "op": "SUM", "group": "world", "async_op": True}, {"w": "w2"}),
        C("wait", {"w": "w2"}),
        C("new_subgroups", {"size": 2}, {"g": "g3"}),
        C("broadcast", {"t": "t1", "root": 1, "root_mode": "group", "group": "g3", "async_op": False}),
        C(
            "all_to_all_single",
            {
                "inspec": T("float32", (0, 2)),
                "matrix": [[2, 0, 1, 3], [1, 1, 1, 1], [0, 2, 0, 2], [3, 1, 2, 0]],
                "even": False,
                "group": "world",
                "async_op": False,
            },
            {"out": "t4", "in": "t5"},
        ),
        C(
            "reduce_scatter",
            {
                "out": T("int32", (3,)),
                "ins": T("int32", (3,), seed=9),
                "n_delta": 0,
                "op": "MAX",
                "group": "world",
                "async_op": False,
            },
            {"out": "t6"},
        ),
        C(
            "all_gather",
            {"out": T("int64", (4, 2)), "n_delta": 0, "t": "t1", "group": "g3", "async_op": False},
            {"outs": "l7"},
        ),
        C("ring", {"t": "t1", "shift": 1}, {"out": "t8"}),
        C("p2p", {"t": "t8", "src": 0, "dst": 3, "tag": 1, "async_op": False}),
        C(
            "gather",
            {
                "t": "t6",
                "out": T("int32", (3,)),
                "n_delta": 0,
                "root": 2,
                "root_mode": "global",
                "list_everywhere": False,
                "group": "world",
                "async_op": False,
            },
            {"outs": "l9"},
        ),
        C("reduce", {"t": "t1", "root": 0, "root_mode": "group", "op": "MIN", "group": "g3", "async_op": False}),
    ]
    res, fs, info = run(calls)
    assert res["kind"] == "ok" and info["ref"] == "valid" and info["exc"] == 0, info
    assert fs == [], fs


def test_FP_ring_after_async_op_reports_wrong_result():
    """BUG R1: reference does not mark ring as racy -> WRONG_RESULT for a program whose only 'bug' is the
    (fuzzer-generated) missing wait().  Real torch is not at fault."""
    calls = [
        C("tensor", {"spec": T("int64", (8,))}, {"out": "t1"}),
        C("all_reduce", {"t": "t1", "op": "SUM", "group": "world", "async_op": True}, {"w": "w2"}),
        C("ring", {"t": "t1", "shift": 1}, {"out": "t3"}),
    ]
    res, fs, info = run(calls)
    assert res["kind"] == "ok"
    assert info["racy"], "ring must be tracked as touching t1 while w2 is in flight"


def test_FP_all_to_all_single_nonmember_then_use():
    """BUG R2: SILENT_ACCEPT for a program that is perfectly valid on real torch."""
    calls = [
        C("new_group", {"ranks": [2], "local_sync": False}, {"g": "g1"}),
        C(
            "all_to_all_single",
            {"group": "g1", "inspec": T("float32", (2, 2)), "matrix": [[3]], "even": True, "async_op": False},
            {"out": "t2", "in": "t3"},
        ),
        C("all_reduce", {"group": "world", "t": "t2", "op": "SUM", "async_op": False}),
    ]
    res, fs, info = run(calls)
    assert res["kind"] == "ok" and info["exc"] == 0
    assert "SILENT_ACCEPT" not in kinds(fs), fs


def test_FP_complex_product_is_documented_rejection():
    """BUG R3: torch's supports_complex() deny-list; the reference calls it valid -> VALID_ERROR."""
    calls = [
        C("tensor", {"spec": T("complex64", (3,))}, {"out": "t1"}),
        C("all_reduce", {"t": "t1", "op": "PRODUCT", "group": "world", "async_op": False}),
    ]
    res, fs, info = run(calls)
    assert info["exc"] == 4
    assert "VALID_ERROR" not in kinds(fs), fs


def test_FP_reduce_after_reduce_is_reference_bug():
    """BUG R4: REFERENCE_BUG (AttributeError) instead of checking the program."""
    calls = [
        C("tensor", {"spec": T("int32", (3,))}, {"out": "t1"}),
        C("reduce", {"t": "t1", "root": 1, "root_mode": "global", "op": "SUM", "group": "world", "async_op": False}),
        C("all_reduce", {"t": "t1", "op": "MAX", "group": "world", "async_op": False}),
    ]
    p = {"world": 4, "calls": calls}
    res = run_once(p, world=4, timeout=3.0)
    try:
        classify(p, res)
    except AttributeError as e:
        pytest.fail(f"reference crashed: {e}")


REDUCE_TO_2 = C(
    "reduce", {"t": "t1", "root": 2, "root_mode": "global", "op": "SUM", "group": "world", "async_op": False}
)


def test_TP_reduce_nonroot_buffer_is_unspecified():
    """Ground truth for O5: Gloo leaves partial sums in non-root buffers, so the reference cannot predict them."""
    spec = T("float32", (6,))
    p = {"world": 4, "calls": [C("tensor", {"spec": spec}, {"out": "t1"}), REDUCE_TO_2]}
    res = run_once(p, world=4, timeout=3.0)
    assert res["kind"] == "ok", res
    changed = [
        r
        for r in (0, 1, 3)
        if not torch.equal(res["results"][r]["outputs"]["t1"], expected_initial(spec, r, salt(0, "out")))
    ]
    assert changed, "Gloo left every non-root buffer untouched; treating it as unspecified would be too lenient"


@pytest.mark.parametrize("shape", [(6,), (0, 3, 1)])
def test_FP_all_gather_of_nonroot_reduce_buffer(shape):
    """BUG O5: WRONG_RESULT|all_gather|list for every all_gather of a buffer a previous reduce left unspecified."""
    calls = [
        C("tensor", {"spec": T("float32", shape)}, {"out": "t1"}),
        REDUCE_TO_2,
        C(
            "all_gather",
            {"group": "world", "out": T("float32", shape, seed=7), "n_delta": 0, "t": "t1", "async_op": False},
            {"outs": "l3"},
        ),
    ]
    res, fs, info = run(calls)
    assert fs == [] and info["checked"], (fs, info)


def test_FP_same_membership_distinct_groups():
    """BUG R5: VALID_ERROR|..|Timed out for a program that is invalid (rank 1 uses another group)."""
    calls = [
        C("new_subgroups", {"size": 4}, {"g": "g1"}),
        C("tensor", {"spec": T()}, {"out": "t2"}),
        C(
            "broadcast",
            {"group": "world", "t": "t2", "root": 0, "root_mode": "global", "async_op": False},
            div={"1": {"group": "g1"}},
        ),
    ]
    res, fs, info = run(calls, timeout=2.0)
    assert info["ref"] == "invalid", info


def test_TP_strided_all_reduce_is_reported():
    """Real Gloo bug (strides ignored) must still be found after any fix."""
    calls = [
        C("tensor", {"spec": T("int64", (4,), layout="noncontig")}, {"out": "t1"}),
        C("all_reduce", {"t": "t1", "op": "SUM", "group": "world", "async_op": False}),
    ]
    res, fs, info = run(calls)
    assert "WRONG_RESULT" in kinds(fs), (fs, info)


def test_TP_agit_stack_form_rejected_by_gloo_contradicts_docs():
    """Reproducible on 2.11 and 2.14: docs promise the stacked output form, Gloo raises -> VALID_ERROR is a
    true positive with respect to the documented contract."""
    calls = [
        C("tensor", {"spec": T("float32", (2, 3))}, {"out": "t1"}),
        C(
            "all_gather_into_tensor",
            {"out": T("float32", (4, 2, 3)), "t": "t1", "group": "world", "async_op": False},
            {"out": "t2"},
        ),
    ]
    res, fs, info = run(calls)
    assert "VALID_ERROR" in kinds(fs) and "invalid tensor size" in fs[0]["sig"]


def test_ISO_session_survives_timeout_desync():
    """Executor isolation: a program in which two ranks time out must not poison the next valid program."""
    from distfuzz.collectives.executor import Session

    s = Session(4, timeout=1.0, coverage=False)
    s.start()
    try:
        bad = {
            "world": 4,
            "calls": [
                C("tensor", {"spec": T()}, {"out": "t1"}),
                C(
                    "all_reduce",
                    {"t": "t1", "op": "SUM", "group": "world", "async_op": False},
                    div={"3": {"__skip__": True}},
                ),
            ],
        }
        good = {
            "world": 4,
            "calls": [
                C("tensor", {"spec": T("int64", (5,))}, {"out": "t1"}),
                C("all_reduce", {"t": "t1", "op": "SUM", "group": "world", "async_op": False}),
                C("new_subgroups", {"size": 2}, {"g": "g2"}),
                C(
                    "all_gather",
                    {"out": T("int64", (5,)), "n_delta": 0, "t": "t1", "group": "g2", "async_op": False},
                    {"outs": "l3"},
                ),
            ],
        }
        for _ in range(3):
            r = s.run(bad)
            assert r["kind"] == "ok"
            r = s.run(good)
            fs, info = classify(good, r)
            assert r["kind"] == "ok" and info["exc"] == 0 and fs == [], (info, fs)
    finally:
        s.stop(hard=True)
