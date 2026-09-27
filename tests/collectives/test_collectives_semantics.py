from functools import partial

import pytest
import torch
from ranks import run_cases

from distfuzz.collectives import tensors as T
from distfuzz.collectives.reference import op_ok, reduce_vals

pytestmark = pytest.mark.multirank

DTS = ["float32", "float64", "float16", "bfloat16", "int8", "uint8", "int32", "int64", "bool", "complex64"]
OPS = ["SUM", "PRODUCT", "MIN", "MAX", "BAND", "BOR", "BXOR", "AVG"]
FLOATISH = {"float16", "bfloat16", "float32", "float64", "complex64"}


def _vals(dt, rank, kind="int"):
    return T.fill_values({"dtype": dt, "shape": [6], "seed": 3, "kind": kind}, rank)


def _close(a, b, dt):
    if dt not in FLOATISH:
        return torch.equal(a, b)
    ca = torch.complex128 if a.is_complex() else torch.float64
    return bool(torch.allclose(a.to(ca), b.to(ca), rtol=2e-2, atol=2e-2))


def case_allreduce(rank, world, dt, op, kind="int"):
    import torch.distributed as dist

    t = _vals(dt, rank, kind).clone()
    dist.all_reduce(t, op=getattr(dist.ReduceOp, op))
    exp = reduce_vals(op, [_vals(dt, r, kind) for r in range(world)])
    return {"same": _close(t, exp, dt), "got": t.tolist()[:6], "exp": exp.tolist()[:6]}


def case_reduce_root(rank, world, dt, op):
    import torch.distributed as dist

    t = _vals(dt, rank).clone()
    dist.reduce(t, dst=2, op=getattr(dist.ReduceOp, op))
    exp = reduce_vals(op, [_vals(dt, r) for r in range(world)])
    return {"same": _close(t, exp, dt) if rank == 2 else None}


def case_a2a_uneven(rank, world):
    """all_to_all_single with a split matrix, checked against semantics.a2a_shapes + the reference's chunking."""
    import torch.distributed as dist

    from distfuzz.collectives.semantics import a2a_shapes

    M = [[2, 0, 1, 3], [1, 1, 1, 1], [0, 2, 0, 2], [3, 1, 2, 0]]
    args = {"inspec": {"shape": [0, 2]}, "matrix": M, "even": False}
    ish, osh, isp, osp = a2a_shapes(args, world, rank)
    inp = torch.arange(ish[0] * 2, dtype=torch.float32).reshape(ish) + 100 * rank
    out = torch.zeros(osh)
    dist.all_to_all_single(out, inp, output_split_sizes=osp, input_split_sizes=isp)
    chunks = []
    for y in range(world):
        isp_y = M[y]
        off = sum(isp_y[:rank])
        src = torch.arange(sum(isp_y) * 2, dtype=torch.float32).reshape(sum(isp_y), 2) + 100 * y
        chunks.append(src[off : off + isp_y[rank]])
    return torch.equal(out, torch.cat(chunks).reshape(osh))


def case_reduce_scatter_list(rank, world):
    import torch.distributed as dist

    ins = [torch.arange(3, dtype=torch.int64) * (k + 1) + rank for k in range(world)]
    out = torch.zeros(3, dtype=torch.int64)
    dist.reduce_scatter(out, ins, op=dist.ReduceOp.SUM)
    exp = reduce_vals("SUM", [torch.arange(3, dtype=torch.int64) * (rank + 1) + y for y in range(world)])
    return torch.equal(out, exp)


def case_subgroup_group_root(rank, world):
    """broadcast with group_src on new_subgroups(2): root resolves per subgroup (semantics.resolve_root)."""
    import torch.distributed as dist

    from distfuzz.collectives.semantics import resolve_root

    cur, subs = dist.new_subgroups(2)
    members = sorted(dist.get_process_group_ranks(cur))
    t = torch.tensor([float(rank)])
    dist.broadcast(t, group_src=1, group=cur)
    root = resolve_root({"root": 1, "root_mode": "group"}, members, rank)
    for s in subs:
        dist.destroy_process_group(s)
    return t.item() == float(root)


def case_agit_forms(rank, world, form):
    import torch.distributed as dist

    t = (torch.arange(6, dtype=torch.float32) + 10 * rank).reshape(2, 3)
    out = torch.zeros(world * 2, 3) if form == "concat" else torch.zeros(world, 2, 3)
    dist.all_gather_into_tensor(out, t)
    return True


def case_nonmember(rank, world):
    import torch.distributed as dist

    g = dist.new_group([0, 1, 3])
    t = torch.ones(2) * rank
    r1 = dist.all_reduce(t, group=g)
    r2 = dist.reduce(t, dst=3, group=g)
    out = torch.zeros(3)
    r3 = dist.all_to_all_single(out, torch.ones(3), group=g)
    if rank in (0, 1, 3):
        dist.destroy_process_group(g)
    return {"rets": [repr(r1), repr(r2), repr(r3)], "out_untouched": out.tolist() == [0.0, 0.0, 0.0]}


def case_strided(rank, world, which):
    import torch.distributed as dist

    if which == "all_reduce_1d_slice":
        buf = torch.full((8,), -1.0)
        v = buf[::2]
        v.copy_(torch.arange(4.0) + 10 * rank)
        dist.all_reduce(v)
        return {
            "correct": torch.equal(v, sum(torch.arange(4.0) + 10 * r for r in range(world))),
            "gaps": buf[1::2].tolist(),
        }
    if which == "scatter_transposed_out":
        v = torch.zeros(3, 2).t()
        lst = [torch.arange(6.0).reshape(2, 3) + 10 * k for k in range(world)] if rank == 0 else None
        dist.scatter(v, lst, src=0)
        return {"correct": torch.equal(v, torch.arange(6.0).reshape(2, 3) + 10 * rank)}
    if which == "all_reduce_expanded":
        base = torch.tensor([[1.0, 2.0]])
        e = base.expand(3, 2)
        dist.all_reduce(e)
        return {"correct": torch.equal(e, torch.tensor([[4.0, 8.0]]).expand(3, 2))}


def case_same_members_diff_groups(rank, world):
    from datetime import timedelta

    import torch.distributed as dist

    g = dist.new_group([0, 1, 2, 3], timeout=timedelta(seconds=2))
    t = torch.ones(2) * rank
    try:
        if rank == 1:
            dist.broadcast(t, src=0, group=g)
        else:
            dist.broadcast(t, src=0)
        return "ok"
    except Exception as e:  # noqa: BLE001
        return "exc:" + type(e).__name__
    finally:
        dist.destroy_process_group(g)


CASES = []
for _dt in DTS:
    for _op in OPS:
        CASES.append((f"allreduce/{_dt}/{_op}", partial(case_allreduce, dt=_dt, op=_op)))
for _dt in ["float16", "bfloat16", "float32"]:
    CASES.append((f"allreduce_randn/{_dt}/PRODUCT", partial(case_allreduce, dt=_dt, op="PRODUCT", kind="randn")))
for _dt in ["int32", "float32", "bool", "complex64"]:
    for _op in ["SUM", "PRODUCT", "MAX"]:
        CASES.append((f"reduce/{_dt}/{_op}", partial(case_reduce_root, dt=_dt, op=_op)))
CASES += [
    ("a2a_uneven", case_a2a_uneven),
    ("reduce_scatter_list", case_reduce_scatter_list),
    ("subgroup_group_root", case_subgroup_group_root),
    ("agit/concat", partial(case_agit_forms, form="concat")),
    ("agit/stack", partial(case_agit_forms, form="stack")),
    ("nonmember", case_nonmember),
    ("strided/all_reduce_1d_slice", partial(case_strided, which="all_reduce_1d_slice")),
    ("strided/scatter_transposed_out", partial(case_strided, which="scatter_transposed_out")),
    ("strided/all_reduce_expanded", partial(case_strided, which="all_reduce_expanded")),
    ("same_members_diff_groups", case_same_members_diff_groups),
]


@pytest.fixture(scope="module")
def R():
    # one session per family: an int/bool AVG failure raises inside the Gloo work and kills the pairs,
    # after which every collective in that session times out
    fams = {}
    for name, fn in CASES:
        fams.setdefault(
            name.split("/")[0] + "/" + name.split("/")[1] if name.startswith("allreduce/") else name.split("/")[0], []
        ).append((name, fn))
    out = {}
    for cases in fams.values():
        out.update(run_cases(cases, world=4, timeout=4.0, deadline=600))
    return out


def _ok(per):
    return all(s == "ok" for s, _ in per)


@pytest.mark.parametrize("dt", DTS)
@pytest.mark.parametrize("op", OPS)
def test_all_reduce_matches_reference_when_torch_accepts(R, dt, op):
    per = R[f"allreduce/{dt}/{op}"]
    if _ok(per):
        assert all(v["same"] for _, v in per), per[0][1]
        if not op_ok(op, dt):
            pytest.xfail(f"blind spot: {op}/{dt} works on Gloo but reference says uncertain")
    else:
        # torch rejects the combo: the reference must NOT call it valid, else VALID_ERROR false positives
        assert not op_ok(op, dt), f"torch raised {per[0][1]!r} but op_ok says valid"


@pytest.mark.parametrize("dt", ["float16", "bfloat16", "float32"])
def test_all_reduce_product_randn_within_oracle_tolerance(R, dt):
    from distfuzz.collectives.oracle import close

    per = R[f"allreduce_randn/{dt}/PRODUCT"]
    assert _ok(per)
    for _, v in per:
        assert close(torch.tensor(v["got"], dtype=T.DTYPES[dt]), torch.tensor(v["exp"], dtype=T.DTYPES[dt]))


@pytest.mark.parametrize("dt", ["int32", "float32", "bool", "complex64"])
@pytest.mark.parametrize("op", ["SUM", "PRODUCT", "MAX"])
def test_reduce_root_matches_reference(R, dt, op):
    per = R[f"reduce/{dt}/{op}"]
    if _ok(per):
        assert per[2][1]["same"] is True
    else:
        assert not op_ok(op, dt), f"torch raised {per[0][1]!r} for reduce but op_ok says valid"


def test_all_to_all_single_uneven_chunking(R):
    assert all(v is True for _, v in R["a2a_uneven"]), R["a2a_uneven"]


def test_reduce_scatter_list_semantics(R):
    assert all(v is True for _, v in R["reduce_scatter_list"])


def test_subgroup_group_relative_root(R):
    assert all(v is True for _, v in R["subgroup_group_root"])


def test_all_gather_into_tensor_stack_form_is_rejected_by_gloo(R):
    """Documented as supported (all_gather_into_tensor docstring form (ii)) but Gloo raises: real PyTorch
    behaviour (4 VALID_ERROR findings are TRUE positives w.r.t. the docs)."""
    assert _ok(R["agit/concat"])
    assert not _ok(R["agit/stack"]) and "invalid tensor size" in R["agit/stack"][0][1]


def test_nonmember_all_to_all_single_returns_none_and_leaves_output(R):
    """Ground truth for BUG R2: non-members get None and an untouched (but defined) output tensor."""
    assert all(v["rets"] == ["None", "None", "None"] for _, v in R["nonmember"])
    assert R["nonmember"][2][1]["out_untouched"]


def test_strided_collectives_are_really_wrong(R):
    """The dominant WRONG_RESULT/GUARD class is real Gloo behaviour (strides ignored, out-of-view writes)."""
    assert all(not v["correct"] for _, v in R["strided/all_reduce_1d_slice"])
    assert all(v["gaps"] != [-1.0] * 4 for _, v in R["strided/all_reduce_1d_slice"])  # wrote outside the view
    assert all(not v["correct"] for _, v in R["strided/scatter_transposed_out"])


def test_same_membership_different_group_objects_do_not_communicate(R):
    """Ground truth for BUG R5."""
    # rank 1 (on g) and the root never meet; a receiver may still get the root's data before the root times out
    assert R["same_members_diff_groups"][1][1].startswith("exc"), R["same_members_diff_groups"]
