import copy
import json
import random

import pytest
import torch

from distfuzz.collectives import oracle
from distfuzz.collectives import prog as P
from distfuzz.collectives.reference import Val, op_ok, reduce_vals, run_reference


def T(dtype="float32", shape=(2,), layout="contig", seed=1, kind="int"):
    return {"dtype": dtype, "shape": list(shape), "layout": layout, "seed": seed, "kind": kind}


def call(op, args, rets=None, div=None):
    return {"op": op, "args": args, "rets": rets or {}, "div": div or {}}


# reduce semantics (ground truth: probe)

# Measured on torch 2.11.0+cpu / Gloo, 4 ranks: which (op, dtype) combos are accepted.
ACCEPTED = {
    ("SUM", "complex64"),
    ("AVG", "complex64"),  # torch view_as_real()s complex for SUM/AVG only
    *[
        (op, dt)
        for op in ("SUM", "PRODUCT", "MIN", "MAX", "BAND", "BOR", "BXOR")
        for dt in ("int8", "uint8", "int32", "int64", "bool")
    ],
    *[
        (op, dt)
        for op in ("SUM", "PRODUCT", "MIN", "MAX", "AVG")
        for dt in ("float32", "float64", "float16", "bfloat16")
    ],
}
REJECTED = {
    *[(op, "complex64") for op in ("PRODUCT", "MIN", "MAX", "BAND", "BOR", "BXOR")],
    *[("AVG", dt) for dt in ("int8", "uint8", "int32", "int64", "bool")],
    *[(op, dt) for op in ("BAND", "BOR", "BXOR") for dt in ("float32", "float64", "float16", "bfloat16")],
}


@pytest.mark.parametrize("op,dt", sorted(ACCEPTED))
def test_op_ok_accepts_what_gloo_accepts(op, dt):
    # op_ok may be conservative ("uncertain") for accepted combos - that is a blind spot, not a false positive.
    if not op_ok(op, dt):
        pytest.xfail(
            f"blind spot: reference marks {op}/{dt} uncertain although Gloo computes it (bool bitwise, complex AVG)"
        )


@pytest.mark.parametrize("op,dt", sorted(REJECTED))
def test_op_ok_rejects_what_torch_rejects(op, dt):
    # R3: torch raises ValueError for these, so treating them as valid produced VALID_ERROR false positives
    assert not op_ok(op, dt), f"reference treats {op} on {dt} as valid but torch.distributed raises"


def test_reduce_vals_bool_sum_is_or_product_is_and():
    a, b = torch.tensor([True, False, False]), torch.tensor([False, False, True])
    assert reduce_vals("SUM", [a, b]).tolist() == [True, False, True]
    assert reduce_vals("PRODUCT", [a, b]).tolist() == [False, False, False]


def test_reduce_vals_int_overflow_wraps_like_gloo():
    v = [torch.tensor([100], dtype=torch.int8)] * 2
    assert reduce_vals("SUM", v).item() == -56  # int8 wraparound, same as C++ two's complement in Gloo


def test_reduce_vals_avg_float():
    v = [torch.tensor([1.0]), torch.tensor([2.0]), torch.tensor([4.0]), torch.tensor([5.0])]
    assert reduce_vals("AVG", v).item() == 3.0


# reference-model bugs


def test_R1_ring_respects_inflight_async_work():
    """Regression for R1, which was: `ring` never calls touch(); an un-waited async op on the same tensor is a
    real race but ref.racy stays False, so the oracle compares values -> WRONG_RESULT false positives."""
    p = {
        "world": 4,
        "calls": [
            call("tensor", {"spec": T()}, {"out": "t1"}),
            call("all_reduce", {"t": "t1", "op": "SUM", "group": "world", "async_op": True}, {"w": "w2"}),
            call("ring", {"t": "t1", "shift": 1}, {"out": "t3"}),
        ],
    }
    ref = run_reference(p)
    assert ref.status == "valid"
    assert ref.racy, "ring reads t1 while all_reduce(async) may still be writing it: must be flagged racy"


def test_R1b_local_after_async_is_racy_for_comparison():
    p = {
        "world": 4,
        "calls": [
            call("tensor", {"spec": T()}, {"out": "t1"}),
            call("all_reduce", {"t": "t1", "op": "SUM", "group": "world", "async_op": True}, {"w": "w2"}),
            call("local", {"t": "t1", "fn": "add1"}),
        ],
    }
    assert run_reference(p).racy  # control: the same pattern with `local` IS detected


def test_R2_all_to_all_single_nonmember_outputs_undefined():
    """Regression for R2, which was: non-member ranks of all_to_all_single get `in`/`out` tensors in the
    interpreter but not in the reference; a later use is reported as
    'rank r uses undefined tX' -> SILENT_ACCEPT false positives."""
    p = {
        "world": 4,
        "calls": [
            call("new_group", {"ranks": [2], "local_sync": False}, {"g": "g1"}),
            call(
                "all_to_all_single",
                {"group": "g1", "inspec": T(shape=(2, 2)), "matrix": [[3]], "even": True, "async_op": False},
                {"out": "t2", "in": "t3"},
            ),
            call("all_reduce", {"group": "world", "t": "t2", "op": "SUM", "async_op": False}),
        ],
    }
    ref = run_reference(p)
    assert ref.status != "invalid", ref.why  # real torch: every rank has a [2,2] t2 and the all_reduce succeeds


def test_R4_metadata_lost_after_reduce_crashes_reference():
    """Regression for R4, which was: after `reduce`, non-root values are None and any later collective on
    that tensor dereferences .dtype on None -> AttributeError -> REFERENCE_BUG finding."""
    p = {
        "world": 4,
        "calls": [
            call("tensor", {"spec": T()}, {"out": "t1"}),
            call(
                "reduce",
                {"group": "world", "t": "t1", "root": 1, "root_mode": "global", "op": "SUM", "async_op": False},
            ),
            call("all_reduce", {"group": "world", "t": "t1", "op": "SUM", "async_op": False}),
        ],
    }
    try:
        ref = run_reference(p)
    except AttributeError as e:
        pytest.fail(f"reference crashed instead of producing a status: {e}")
    assert ref.status in ("valid", "uncertain")


def test_R5_distinct_groups_with_same_members_are_not_the_same_group():
    """Regression for R5, which was: participants() compares member lists, not group handles.  A rank using a
    different ProcessGroup with identical membership is treated as joining the same collective -> 'valid',
    but real Gloo times out."""
    p = {
        "world": 4,
        "calls": [
            call("new_subgroups", {"size": 4}, {"g": "g1"}),
            call("tensor", {"spec": T()}, {"out": "t2"}),
            call(
                "broadcast",
                {"group": "world", "t": "t2", "root": 0, "root_mode": "global", "async_op": False},
                div={"1": {"group": "g1"}},
            ),
        ],
    }
    ref = run_reference(p)
    assert ref.status == "invalid", "rank 1 broadcasts on g1, ranks 0,2,3 on world: not the same collective"


def test_R6_async_flag_taken_from_first_member_only():
    """Regression for R6, which was: asy = a0.get('async_op') uses rank m[0]'s flag; a divergent async_op on another
    rank is neither flagged invalid nor tracked as in-flight, so a real race is value-checked."""
    p = {
        "world": 4,
        "calls": [
            call("tensor", {"spec": T()}, {"out": "t1"}),
            call(
                "all_reduce",
                {"t": "t1", "op": "SUM", "group": "world", "async_op": False},
                div={"2": {"async_op": True}},
            ),
            call("local", {"t": "t1", "fn": "add1"}),
        ],
    }
    ref = run_reference(p)
    assert ref.status == "invalid" or ref.racy, "rank 2's async all_reduce is still in flight when local runs"


def test_R7_valid_prefix_of_uncertain_program_is_checked():
    """Tensors fully determined before the first 'uncertain' call are still compared (S1)."""
    p = {
        "world": 4,
        "calls": [
            call("tensor", {"spec": T()}, {"out": "t1"}),
            call("all_reduce", {"t": "t1", "op": "SUM", "group": "world", "async_op": False}),
            call("tensor", {"spec": T(dtype="complex64")}, {"out": "t2"}),
            call("all_reduce", {"t": "t2", "op": "MAX", "group": "world", "async_op": False}),  # uncertain
        ],
    }
    ref = run_reference(p)
    assert ref.status == "uncertain" and ref.at == 3
    res = {
        "kind": "ok",
        "results": [
            {"outputs": {"t1": torch.full((2,), 999.0)}, "lists": {}, "exc": None, "wait_exc": []} for _ in range(4)
        ],
    }
    findings, info = oracle.classify(p, res, ref=ref)
    assert any(f["kind"] == "WRONG_RESULT" for f in findings)
    assert info["checked"]


# oracle / minimizer / stats


def test_O1_valid_error_attributes_to_secondary_exception():
    """Regression for O1, which was: the reported exception is the first non-'Timed out' one in rank order, not the
    earliest by call index; a rank waiting in new_group() for ranks that already raised reports a DistStoreError
    ('wait timeout' does not contain 'Timed out') and hides the root cause."""
    p = {
        "world": 4,
        "calls": [
            call("tensor", {"spec": T()}, {"out": "t1"}),
            call(
                "all_gather_into_tensor",
                {"group": "world", "out": T(shape=(4, 2)), "t": "t1", "async_op": False},
                {"out": "t2"},
            ),
            call("new_subgroups", {"size": 2}, {"g": "g3"}),
        ],
    }
    rs = []
    for r in range(4):
        exc = (
            {"idx": 2, "op": "new_subgroups", "type": "DistStoreError", "msg": "wait timeout after 1000ms"}
            if r == 0
            else {
                "idx": 1,
                "op": "all_gather_into_tensor",
                "type": "RuntimeError",
                "msg": "ProcessGroupGloo::allgather: invalid tensor size",
            }
        )
        rs.append({"outputs": {}, "lists": {}, "exc": exc, "wait_exc": []})
    findings, _ = oracle.classify(p, {"kind": "ok", "results": rs})
    sig = [f["sig"] for f in findings if f["kind"] == "VALID_ERROR"][0]
    assert "all_gather_into_tensor" in sig, sig


def test_O2_guard_signature_embeds_op_set_so_minimizer_cannot_reduce():
    """Regression for O2, which was: GUARD/HANG sigs contain the sorted op list; removing any call changes
    the sig, so ddmin can never accept a smaller candidate.  Also inflates 'findings' counts (126 GUARD 'findings'
    for one bug class)."""
    p = {
        "world": 4,
        "calls": [
            call("tensor", {"spec": T(layout="expanded", shape=(3, 2))}, {"out": "t1"}),
            call("all_reduce", {"t": "t1", "op": "SUM", "group": "world", "async_op": False}),
            call("barrier", {"group": "world", "async_op": False}),
        ],
    }
    rs = [{"outputs": {}, "lists": {}, "exc": None, "wait_exc": [], "guard_bad": 1} for _ in range(4)]
    sig_full = oracle.classify(p, {"kind": "ok", "results": rs})[0][0]["sig"]
    q = copy.deepcopy(p)
    del q["calls"][2]
    sig_small = oracle.classify(q, {"kind": "ok", "results": rs})[0][0]["sig"]
    assert sig_full == sig_small, (
        f"dropping an unrelated barrier changes the GUARD signature: {sig_full!r} vs {sig_small!r}"
    )


def test_O3_crash_signature_depends_on_logdir():
    """Regression for O3, which was: CRASH sig heads come from rank log tails; the
    minimizer Session and the emitted repro run with logdir=None, so their sig is 'CRASH|..|exitcodes=[..]' and
    never equals the recorded one -> CRASH findings can neither be minimized nor confirmed by the repro."""
    import inspect

    from distfuzz.collectives import minimize as M

    p = {"world": 4, "calls": []}
    with_log = oracle.classify(
        p,
        {
            "kind": "crash",
            "crash_log": "terminate called after throwing an instance of 'gloo::EnforceNotMet'",
            "exitcodes": [-6, None, None, None],
            "results": [None] * 4,
        },
    )[0][0]["sig"]
    without = oracle.classify(
        p, {"kind": "crash", "crash_log": "", "exitcodes": [-6, None, None, None], "results": [None] * 4}
    )[0][0]["sig"]
    assert with_log != without  # sig really depends on the log tail
    assert "logdir" in inspect.getsource(M.minimize) and "logdir=None" not in M.REPRO


@pytest.mark.xfail(strict=True, reason="attribution picks the trailing `local`, not the scatter into a strided tensor")
def test_O4_wrong_result_attribution_names_the_collective():
    """Minor: attribution is 'last call whose rets or arg t mention var', so a trailing
    `local` masks the collective that produced the wrong value."""
    p = {
        "world": 4,
        "calls": [
            call("tensor", {"spec": T(layout="noncontig", shape=(3,))}, {"out": "t1"}),
            call(
                "scatter",
                {
                    "group": "world",
                    "t": "t1",
                    "ins": T(shape=(3,)),
                    "n_delta": 0,
                    "root": 0,
                    "root_mode": "global",
                    "list_everywhere": False,
                    "async_op": False,
                },
            ),
            call("local", {"t": "t1", "fn": "mul2"}),
        ],
    }
    ref = run_reference(p)
    rs = [{"outputs": {"t1": torch.zeros(3)}, "lists": {}, "exc": None, "wait_exc": []} for _ in range(4)]
    f = oracle.classify(p, {"kind": "ok", "results": rs}, ref=ref)[0]
    assert f and f[0]["sig"].startswith("WRONG_RESULT|scatter"), f


def _reduce_then_all_gather(shape):
    return {
        "world": 4,
        "calls": [
            call("tensor", {"spec": T(shape=shape)}, {"out": "t1"}),
            call(
                "reduce",
                {"group": "world", "t": "t1", "root": 2, "root_mode": "global", "op": "SUM", "async_op": False},
            ),
            call(
                "all_gather",
                {"group": "world", "out": T(shape=shape, seed=7), "n_delta": 0, "t": "t1", "async_op": False},
                {"outs": "l3"},
            ),
        ],
    }


def _gathered(ref, root_entry, other=lambda e: torch.full_like(e, 99)):
    """What each rank would observe: the root's entry as given, every other entry overwritten by `other`."""
    expected_root = ref.t[2]["t1"].v
    out = [root_entry if k == 2 else other(expected_root) for k in range(4)]
    return [{"outputs": {}, "lists": {"l3": [e.clone() for e in out]}, "exc": None, "wait_exc": []} for _ in range(4)]


@pytest.mark.parametrize("shape", [(3,), (0, 3, 1)])
def test_O5_unknown_list_entries_are_not_compared(shape):
    """Regression for O5, which was: compare() checked list entries with `g == e` when the reference had no
    value (e is None), so every list built from a non-root `reduce` buffer was a WRONG_RESULT. Gloo really
    does leave partial sums there (see test_TP_reduce_nonroot_buffer_is_unspecified in the end-to-end tests)."""
    p = _reduce_then_all_gather(shape)
    ref = run_reference(p)
    assert ref.status == "valid"
    entries = ref.lists[0]["l3"]
    assert [isinstance(e, Val) for e in entries] == [True, True, False, True]
    assert all(e.v is None and e.shape == tuple(shape) for k, e in enumerate(entries) if k != 2)
    findings, info = oracle.classify(p, {"kind": "ok", "results": _gathered(ref, ref.t[2]["t1"].v)}, ref=ref)
    assert findings == [] and info["checked"]


def test_O5_known_list_entry_is_still_compared():
    """The root's entry is fully determined; a wrong value there must still be reported."""
    p = _reduce_then_all_gather((3,))
    ref = run_reference(p)
    wrong = ref.t[2]["t1"].v + 1
    findings, _ = oracle.classify(p, {"kind": "ok", "results": _gathered(ref, wrong)}, ref=ref)
    assert [f["kind"] for f in findings] == ["WRONG_RESULT"]
    assert "(0, 'l3')" in findings[0]["detail"]


@pytest.mark.parametrize(
    "other",
    [lambda e: torch.zeros(e.numel() + 1, dtype=e.dtype), lambda e: e.to(torch.float64)],
    ids=["shape", "dtype"],
)
def test_O5_unknown_list_entry_still_checks_shape_and_dtype(other):
    """An unspecified value still has a specified shape and dtype: all_gather cannot change either."""
    p = _reduce_then_all_gather((3,))
    ref = run_reference(p)
    findings, _ = oracle.classify(p, {"kind": "ok", "results": _gathered(ref, ref.t[2]["t1"].v, other)}, ref=ref)
    assert [f["kind"] for f in findings] == ["WRONG_RESULT"]


def test_S1_stats_count_value_checked_programs():
    p = {
        "world": 4,
        "calls": [
            call("tensor", {"spec": T()}, {"out": "t1"}),
            call("all_reduce", {"t": "t1", "op": "SUM", "group": "world", "async_op": False}),
        ],
    }
    ref = run_reference(p)
    good = [{"outputs": {"t1": ref.t[r]["t1"].v.clone()}, "lists": {}, "exc": None, "wait_exc": []} for r in range(4)]
    findings, info = oracle.classify(p, {"kind": "ok", "results": good}, ref=ref)
    assert findings == [] and info["checked"]


# prog layer invariants (pure python)


def _all_refs_defined(p):
    defined = set()
    for c in p["calls"]:
        if P.refs(c) - defined:
            return False
        defined |= set(c.get("rets", {}).values())
    return True


@pytest.mark.parametrize("seed", range(40))
def test_generate_and_mutate_keep_programs_well_formed(seed):
    rng = random.Random(seed)
    g = P.Generator(4, rng)
    m = P.Mutator(4, rng)
    corpus = [g.generate() for _ in range(3)]
    for p in corpus:
        assert _all_refs_defined(p)
        assert len({v for c in p["calls"] for v in c["rets"].values()}) == sum(len(c["rets"]) for c in p["calls"])
    for _ in range(20):
        q = m.mutate(rng.choice(corpus), corpus)
        assert _all_refs_defined(q), json.dumps(q)[:300]
        assert len(q["calls"]) <= m.max_calls
        names = [v for c in q["calls"] for v in c["rets"].values()]
        assert len(names) == len(set(names)), "duplicate resource names after mutation"
        json.dumps(q)  # serialisable


def test_analyze_counter_exceeds_every_var():
    p = {
        "world": 4,
        "calls": [call("tensor", {"spec": T()}, {"out": "t7"}), call("tensor", {"spec": T()}, {"out": "t2_ab"})],
    }
    st = P.analyze(p, 1)
    assert st.fresh("t") not in ("t7", "t2")


def test_P1_wait_on_undefined_work_is_not_checked_by_reference():
    """Minor: the interpreter raises FuzzerUndefinedVar for a wait on
    an unknown work var; the reference silently ignores it (status stays valid).  Harmless today because the
    oracle suppresses VALID_ERROR when FuzzerUndefinedVar is present, but the two models disagree."""
    p = {"world": 4, "calls": [call("wait", {"w": "w99"})]}
    assert run_reference(p).status == "valid"


def test_new_group_out_of_range_ranks_is_uncertain_not_valid():
    p = {"world": 4, "calls": [call("new_group", {"ranks": [0, 4], "local_sync": False}, {"g": "g1"})]}
    assert run_reference(p).status == "uncertain"
