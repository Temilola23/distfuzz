import json
import subprocess
import sys
import textwrap

import pytest

from distfuzz.dtensor.fuzz import signature
from distfuzz.dtensor.hang import hang_finding
from distfuzz.dtensor.minimize import minimize, sig_match


def _rep(kind, file="_view_ops.py", func="view_groups", line=485, label=None, coll=None, step=2, elapsed=5.2):
    return dict(
        kind=kind,
        file=file,
        func=func,
        line=line,
        lines=[line],
        label=label,
        coll=coll,
        step=step,
        expr="v1.view((0, 3))",
        held=5.0,
        elapsed=elapsed,
        stack=[],
    )


def test_hang_signature_names_the_view_groups_branch():
    s19 = hang_finding([_rep("cpu", label="src_exhausted")] * 4)
    b5 = hang_finding([_rep("cpu", line=489, label="dst_exhausted")] * 4)
    assert signature(s19) == "HANG|cpu|_view_ops.py:view_groups:src_exhausted"
    assert signature(b5) == "HANG|cpu|_view_ops.py:view_groups:dst_exhausted"
    assert s19["step"] == 2 and s19["expr"] == "v1.view((0, 3))"


def test_hang_signature_unknown_loop_not_keyed_by_sampled_line():
    a = hang_finding([_rep("cpu", file="_ops.py", func="spin", line=10)] * 2)
    b = hang_finding([_rep("cpu", file="_ops.py", func="spin", line=12)] * 2)
    assert signature(a) == signature(b) == "HANG|cpu|_ops.py:spin"


def test_hang_signature_collective_names_missing_ranks():
    reps = [_rep("collective", file="distributed_c10d.py", func="all_reduce", coll="all_reduce")] * 3
    reps.append(_rep("collective", file="distributed_c10d.py", func="all_gather", coll="all_gather@harness"))
    assert signature(hang_finding(reps)) == "HANG|collective|all_reduce|ranks-missing:[3]"
    reps[3] = None
    assert signature(hang_finding(reps)) == "HANG|collective|all_reduce|ranks-missing:[3]"


def test_hang_cpu_loop_wins_over_ranks_waiting_for_it():
    reps = [_rep("cpu", label="dst_exhausted")] + [
        _rep("collective", file="distributed_c10d.py", func="all_gather", coll="all_gather@harness")
    ] * 3
    assert signature(hang_finding(reps)) == "HANG|cpu|_view_ops.py:view_groups:dst_exhausted"


def test_hang_without_reports_keeps_legacy_signature():
    assert signature(hang_finding([None] * 4)) == "HANG||program timeout"


def test_hang_stuck_at_deadline_is_keyed_by_line():
    f = hang_finding([_rep("deadline", file="_dispatch.py", func="dispatch", line=170)] * 4)
    assert signature(f) == "HANG|stuck|_dispatch.py:dispatch:L170"


def test_hang_finding_is_json_serialisable():
    json.dumps(hang_finding([_rep("cpu", label="src_exhausted"), None]))


def test_sig_match_hang_is_exact():
    s19 = "HANG|cpu|_view_ops.py:view_groups:src_exhausted"
    assert sig_match(s19, {s19})
    assert not sig_match(s19, {"HANG|cpu|_view_ops.py:view_groups:dst_exhausted"})
    assert not sig_match(s19, {"HANG||program timeout"})


class FirstHangStub:
    # the first hanging step decides the program's hang, as in a real world
    def __init__(self):
        self.runs = 0

    def sigs(self, prog):
        self.runs += 1
        for e in [s["expr"] for s in prog["steps"]]:
            if e == "v0.view((0, 3))":
                return {"HANG|cpu|_view_ops.py:view_groups:src_exhausted"}
            if e == "v0.flatten()":
                return {"HANG|cpu|_view_ops.py:view_groups:dst_exhausted"}
        return set()


def test_minimize_keeps_the_specific_hang():
    # with "any hang" matching, ddmin may drop the S19 step and keep the later B5 step, which also hangs
    prog = dict(
        mesh="1d",
        inputs=[dict(shape=[0], dtype="f32", seed=1, rg=False, pl=["S0"])],
        steps=[
            dict(out="v0", expr="mk(0)"),
            dict(out="v1", expr="v0.view((0, 3))"),
            dict(out="v2", expr="v0.flatten()"),
        ],
    )
    mp, status = minimize(prog, "HANG|cpu|_view_ops.py:view_groups:src_exhausted", FirstHangStub(), 4)
    assert status == "ok" and [s["expr"] for s in mp["steps"]] == ["mk(0)", "v0.view((0, 3))"]
    # the B5 step is shadowed by the S19 step before it, so a B5 target does not reproduce from this program
    mp, status = minimize(prog, "HANG|cpu|_view_ops.py:view_groups:dst_exhausted", FirstHangStub(), 4)
    assert mp is None and status == "not reproducible"


SPIN = textwrap.dedent(
    """
    import json, os, sys
    from torch.distributed.tensor._ops._view_ops import view_groups
    from distfuzz.dtensor.hang import Watchdog

    def emit(rep):
        print(json.dumps(rep), flush=True)
        os._exit(0)

    wd = Watchdog(emit, period=0.1)
    wd.arm(dict(deadline=60, loop=1.0, coll=12))
    wd.step = (1, "spin")
    view_groups(SRC, DST)
    print(json.dumps(dict(kind="returned")), flush=True)
    """
)


@pytest.mark.parametrize(
    "src,dst,label",
    [
        ((4, 0), (4, 2, 0, 0, 0), "src_exhausted"),
        ((0,), (0, 5), "src_exhausted"),
        ((0, 7, 4, 4), (0,), "dst_exhausted"),
        ((3, 0, 5), (3, 0), "dst_exhausted"),
    ],
)
def test_watchdog_names_the_spinning_view_groups_branch(src, dst, label):
    pytest.importorskip("torch")
    code = SPIN.replace("SRC", repr(src)).replace("DST", repr(dst))
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60)
    rep = json.loads(r.stdout.strip().splitlines()[-1])
    assert rep["kind"] == "cpu", rep
    assert (rep["file"], rep["func"], rep["label"]) == ("_view_ops.py", "view_groups", label)
    assert rep["step"] == 1 and rep["elapsed"] < 5


def test_watchdog_quiet_when_view_groups_returns():
    pytest.importorskip("torch")
    code = SPIN.replace("SRC", "(0,)").replace("DST", "(2, 0)")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60)
    assert json.loads(r.stdout.strip().splitlines()[-1]) == dict(kind="returned")


def test_deadlocked_rank_divergent_error_keeps_its_error_signature():
    # ranks 2 and 3 raised in the step and wait in the harness's flag exchange; 0 and 1 wait in the op's collective
    reps = [_rep("collective", file="_functional_collectives.py", func="wait_tensor", coll="wait_tensor")] * 2
    err = "RuntimeError: The size of tensor a (N) must match the size of tensor b (N) at non-singleton dimension N"
    reps += [
        dict(_rep("collective", func="all_gather", coll="all_gather@harness"), err=err, expr="v1.view((0, 3))")
    ] * 2
    f = hang_finding(reps)
    assert f["kind"] == "RANK_DIVERGENT_ERROR" and f["msg"] == err
    assert f["hang"] == "collective|wait_tensor|ranks-missing:[2, 3]"
    assert signature(f) == signature(dict(kind="RANK_DIVERGENT_ERROR", step=1, expr="v1.view((0, 3))", msg=err))


def test_step_collective_is_the_reference_even_when_most_ranks_are_in_the_harness():
    # argmin over a dim sharded 1/0/0/0: ranks 1-3 raise on their empty shards, rank 0 waits in the op's collective
    err = "IndexError: min(): Expected reduction dim N to have non-zero size."
    reps = [_rep("collective", file="_functional_collectives.py", func="wait_tensor", coll="wait_tensor")]
    reps += [dict(_rep("collective", func="all_gather", coll="all_gather@harness"), err=err)] * 3
    f = hang_finding(reps)
    assert f["kind"] == "RANK_DIVERGENT_ERROR" and f["msg"] == err
    assert f["hang"] == "collective|wait_tensor|ranks-missing:[1, 2, 3]"
    reps[1:] = [_rep("collective", func="all_gather", coll="all_gather@harness")] * 3
    assert signature(hang_finding(reps)) == "HANG|collective|wait_tensor|ranks-missing:[1, 2, 3]"
