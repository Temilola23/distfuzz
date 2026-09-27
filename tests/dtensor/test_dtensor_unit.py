import random
import re

import pytest
import torch

from distfuzz.dtensor.fuzz import opname, signature
from distfuzz.dtensor.gen import MUT_RE, Gen
from distfuzz.dtensor.minimize import ddmin, minimize
from distfuzz.dtensor.runtime import compare, gen_full, partial_pieces, tol_for


def test_opname_basic():
    assert opname("v3.sum(dim=0)") == "sum"
    assert opname("torch.linalg.vector_norm(v1, dim=-2)") == "vector_norm"
    assert opname("F.mse_loss(v1, v2)") == "mse_loss"
    assert opname("v1 + v2") == "+"
    assert opname("v1[:, 0]") == "getitem"
    assert opname("R(v1, ['R'])") == "redistribute"
    assert opname("CMP('lambda a: a.sum(dim=0)', v1)") == "compile:a.sum(dim=0)"


def test_signature_final_mismatch_not_keyed_by_variable_name():
    a = dict(kind="FINAL_MISMATCH", step=3, expr="v7", msg="values differ at [[0]] maxdiff=1")
    b = dict(kind="FINAL_MISMATCH", step=3, expr="v0", msg="values differ at [[0]] maxdiff=1")
    assert signature(a) == signature(b)


def test_signature_bw_setitem_grad_not_keyed_by_variable_name():
    f1 = dict(kind="DIST_ERR", step=3, expr="BW(v7)", msg="RuntimeError: aten.cat.default: got mixed")
    f2 = dict(kind="DIST_ERR", step=3, expr="BW(v10)", msg="RuntimeError: aten.cat.default: got mixed")
    assert signature(f1) == signature(f2)
    s1 = dict(kind="MISMATCH", step=3, expr="SETITEM(v10, 0, 5)", msg="values differ")
    s2 = dict(kind="MISMATCH", step=3, expr="SETITEM(v6, 6, 5)", msg="values differ")
    assert signature(s1) == signature(s2)
    g1 = dict(kind="GRAD_MISMATCH", step=-1, expr="mk(0).grad", msg="values differ")
    g2 = dict(kind="GRAD_MISMATCH", step=-1, expr="mk(4).grad", msg="values differ")
    assert signature(g1) == signature(g2)


def test_signature_dist_err_one_root_cause_is_one_signature():
    # the guided campaign recorded 41 "unique" signatures for this single message before names were normalised
    m = "NotImplementedError: Operator aten.fill_.Tensor does not have a sharding strategy registered."
    sigs = {
        signature(dict(kind="DIST_ERR", step=1, expr=e, msg=m))
        for e in ("SETITEM(v1, 0, 5)", "SETITEM(v2, 3, 5)", "SETITEM(v1, slice(0, 1), 0)")
    }
    assert len(sigs) == 1


def test_signature_rank_divergent_two_sigs_per_event():
    # the raising rank and the waiting ranks record different messages, so one event yields two signatures
    a = dict(kind="RANK_DIVERGENT_ERROR", step=1, expr="v0.max()", msg="RuntimeError: max(): Expected reduction dim")
    b = dict(kind="RANK_DIVERGENT_ERROR", step=1, expr="v0.max()", msg="other rank failed")
    assert signature(a) != signature(b)


def test_mut_re_matches_inplace_only():
    assert MUT_RE.search("v1.mul_(2)")
    assert MUT_RE.search("SETITEM(v1, 0, 5)")
    assert MUT_RE.search("BW(v3)")
    for e in (
        "torch.nan_to_num(v1.log())",
        "F.one_hot(v1, 3)",
        "F.layer_norm(v1, [4])",
        "torch.full_like(v1, 3)",
        "v1.new_zeros((3,))",
        "F.binary_cross_entropy_with_logits(v1, v2)",
    ):
        assert not MUT_RE.search(e), e


def test_compare_basic():
    assert compare(torch.tensor([1.0]), torch.tensor([1.0])) is None
    assert compare(torch.zeros(0), torch.zeros(0)) is None
    assert "shape" in compare(torch.zeros(2), torch.zeros(3))
    assert "dtype" in compare(torch.zeros(2), torch.zeros(2, dtype=torch.float64))
    assert compare(torch.tensor([float("nan")]), torch.tensor([float("nan")])) is None


def test_compare_names_inf_vs_nan():
    d = compare(torch.tensor([float("inf")]), torch.tensor([float("nan")]))
    assert d is not None and "nan/inf" in d and "maxdiff" not in d


def test_compare_bf16_partial_sum_cancellation_within_tolerance():
    # bf16 bmm with the contraction dim sharded: each rank rounds its partial sum to bf16, so outputs that
    # cancel to ~10 carry an error of a few units. The plain bf16 reference is equally far from the truth,
    # so the magnitude-scaled tolerance must accept it.
    a = gen_full(dict(shape=[12, 7, 8], dtype="bf16", seed=679081663))
    a = a.sum() + a
    b = gen_full(dict(shape=[12, 8, 5], dtype="bf16", seed=560630790))
    ref = torch.bmm(a, b)
    truth = torch.bmm(a.double(), b.double())
    parts = [torch.bmm(a[..., k : k + 2], b[:, k : k + 2, :]) for k in range(0, 8, 2)]
    got = parts[0] + parts[1] + parts[2] + parts[3]
    err_ref = (ref.double() - truth).abs().max().item()
    err_got = (got.double() - truth).abs().max().item()
    assert err_ref >= 1.0
    assert err_got <= 8 * max(err_ref, 1.0)
    assert not torch.equal(ref, got)
    assert compare(ref, got) is None


def test_compare_still_flags_real_bf16_mismatch():
    ref = torch.full((4,), 10.0, dtype=torch.bfloat16)
    assert compare(ref, ref + 4) is not None


def test_tol_for():
    assert tol_for(torch.bfloat16) == (2e-2, 2e-2)
    assert tol_for(torch.float32) == (1e-4, 1e-4)
    assert tol_for(torch.float32, 500.0) == (1e-4, 5e-2)
    assert tol_for(torch.int64) == (0.0, 0.0)


@pytest.mark.parametrize("dtype", ["f32", "f64", "bf16", "f16", "i64", "i32"])
@pytest.mark.parametrize("op", ["sum", "avg", "max", "min"])
def test_partial_pieces_reduce_exactly(dtype, op):
    if op == "avg" and dtype in ("i64", "i32"):
        pytest.skip("avg on int is not generated by rand_placements")
    T = gen_full(dict(shape=[6, 5], dtype=dtype, seed=3))
    pcs = partial_pieces(T, 4, op, 99)
    if op == "sum":
        red = sum(p.double() for p in pcs)
    elif op == "avg":
        red = sum(p.double() for p in pcs) / 4
    elif op == "max":
        red = torch.stack(pcs).max(0)[0].double()
    else:
        red = torch.stack(pcs).min(0)[0].double()
    assert torch.equal(red, T.double())


@pytest.mark.parametrize("op", ["sum", "max", "min"])
def test_partial_pieces_bool_reduce_exactly(op):
    T = gen_full(dict(shape=[6, 5], dtype="bool", seed=3))
    pcs = torch.stack(partial_pieces(T, 4, op, 1))
    red = pcs.all(0) if op == "min" else pcs.any(0)
    assert torch.equal(red, T)


def _ref_valid(gen, prog):
    return all(gen.replay(prog)[2])


def test_generate_programs_are_ref_valid():
    gen = Gen(4, random.Random(7))
    for _ in range(15):
        p = gen.generate()
        assert _ref_valid(gen, p), p["steps"]


def test_mutate_programs_are_ref_valid():
    gen = Gen(4, random.Random(11))
    parent = gen.generate()
    for _ in range(25):
        m = gen.mutate(parent)
        assert _ref_valid(gen, m), (m.get("mut"), m["steps"])


def test_dtype_mutation_regenerates_partial_placements():
    # a Partial(avg) input that becomes int or bool cannot be built; the mutation must re-draw its placements
    gen = Gen(4, random.Random(5))
    parent = dict(
        mesh="1d",
        grad=False,
        nv=1,
        inputs=[dict(shape=[4], dtype="f32", seed=1, rg=False, pl=["P:avg"])],
        steps=[dict(out="v0", expr="mk(0)")],
    )
    changed = []
    for _ in range(600):
        m = gen.mutate(parent)
        spec = m["inputs"][0]
        if m.get("mut") == "dtype_shape" and spec["dtype"] != "f32":
            changed.append((spec["dtype"], spec["pl"]))
    assert any(dt in ("i64", "i32", "bool") for dt, _ in changed), changed
    for dt, pl in changed:
        if dt in ("i64", "i32"):
            assert "P:avg" not in pl, (dt, pl)
        if dt == "bool":
            assert not any(p.startswith("P:") for p in pl), (dt, pl)


def test_mesh_mutation_keeps_malformed_redistribute_targets():
    # deliberate (10% of mesh mutations): exercises placement-length validation in redistribute
    gen = Gen(4, random.Random(3))
    p = dict(
        mesh="1d",
        inputs=[dict(shape=[4], dtype="f32", seed=1, rg=False, pl=["R"])],
        steps=[dict(out="v0", expr="mk(0)"), dict(out="v1", expr="R(v0, ['S0'])")],
        nv=2,
    )
    bad = 0
    for _ in range(200):
        gen.rng = random.Random(gen.rng.random())
        m = gen.mutate(p)
        if m["mesh"] == "2d":
            for st in m["steps"]:
                if st["expr"].startswith("R(") and st["expr"].count("'") == 2:
                    bad += 1
    assert bad > 0


def test_ddmin_basic():
    out = ddmin(list("abcdefgh"), lambda s: "c" in s and "f" in s)
    assert sorted(out) == ["c", "f"]


class StubTester:
    # honest=False mimics a worker that attributes a finding to a step that cannot run (undefined input)
    def __init__(self, trigger, honest=True):
        self.trigger, self.honest = trigger, honest
        self.runs = 0

    def sigs(self, prog):
        self.runs += 1
        defined, out = set(), set()
        for st in prog["steps"]:
            refs = set(re.findall(r"\bv\d+\b", st["expr"]))
            if self.honest and refs - defined:
                continue
            defined.add(st["out"])
            if self.trigger in st["expr"]:
                out.add("MISMATCH|max|values")
        return out


def _well_formed(steps):
    defined = set()
    for st in steps:
        if set(re.findall(r"\bv\d+\b", st["expr"])) - defined:
            return False
        defined.add(st["out"])
    return True


def test_minimize_keeps_definitions_when_worker_is_honest():
    prog = dict(
        mesh="1d",
        inputs=[dict(shape=[2], dtype="f32", seed=1, rg=False, pl=["S0"])],
        steps=[
            dict(out="v0", expr="mk(0)"),
            dict(out="v1", expr="v0.max(dim=0)[0]"),
            dict(out="v2", expr="v0 * 2"),
            dict(out="v3", expr="v2.clone()"),
        ],
        nv=4,
    )
    mp, status = minimize(prog, "MISMATCH|max|values", StubTester("max("), 4)
    assert status == "ok" and _well_formed(mp["steps"])
    assert [s["expr"] for s in mp["steps"]] == ["mk(0)", "v0.max(dim=0)[0]"]


def test_minimize_output_is_well_formed_even_if_attribution_is_wrong():
    prog = dict(
        mesh="1d",
        inputs=[dict(shape=[1, 1], dtype="i64", seed=1, rg=False, pl=["S0"])],
        steps=[
            dict(out="v0", expr="mk(0)"),
            dict(out="v2", expr="v0 * 2"),
            dict(out="v3", expr="v0.max(dim=-2)[0]"),
            dict(out="v4", expr="v2.clone()"),
        ],
        nv=5,
    )
    mp, status = minimize(prog, "MISMATCH|max|values", StubTester("clone(", honest=False), 4)
    assert status == "ok"
    assert _well_formed(mp["steps"]), mp["steps"]
    assert any("clone(" in s["expr"] for s in mp["steps"])


def test_minimize_target_step_not_pinned():
    # only the signature is matched, so which of two equivalent trigger steps survives is arbitrary
    prog = dict(
        mesh="1d",
        inputs=[dict(shape=[2], dtype="f32", seed=1, rg=False, pl=["S0"])],
        steps=[
            dict(out="v0", expr="mk(0)"),
            dict(out="v1", expr="v0.max(dim=0)[0]"),
            dict(out="v2", expr="v0 + 1"),
            dict(out="v3", expr="v2.max(dim=0)[0]"),
        ],
        nv=4,
    )
    mp, status = minimize(prog, "MISMATCH|max|values", StubTester("max("), 4)
    assert status == "ok"
    assert len(mp["steps"]) == 2
