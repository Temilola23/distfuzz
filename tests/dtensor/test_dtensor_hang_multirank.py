import time

import pytest

from distfuzz.dtensor.fuzz import signature
from distfuzz.dtensor.hang import hang_finding
from distfuzz.dtensor.world import World

pytestmark = pytest.mark.multirank


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    w = World(4, str(tmp_path_factory.mktemp("hang-logs")), cov=False, loop_secs=2.0, coll_secs=3.0)
    yield w
    w.close()


def _prog(steps, shape=(0,)):
    return dict(
        id=1,
        mesh="1d",
        inputs=[dict(shape=list(shape), dtype="f32", seed=1, rg=False, pl=["R"])],
        steps=[dict(out="v0", expr="mk(0)")] + [dict(out=f"v{i + 1}", expr=e) for i, e in enumerate(steps)],
    )


@pytest.mark.parametrize(
    "shape,expr,label",
    [
        ((0,), "v0.view((0, 5))", "src_exhausted"),
        ((4, 0), "v0.reshape((4, 2, 0, 0, 0))", "src_exhausted"),
        ((3, 0, 5), "v0.flatten(1, 2)", "dst_exhausted"),
        ((0, 7, 4, 4), "v0.view(-1)", "dst_exhausted"),
    ],
)
def test_view_groups_hangs_get_branch_signatures_quickly(world, shape, expr, label):
    t0 = time.time()
    status, res = world.run(_prog([expr], shape), 30)
    took = time.time() - t0
    assert status == "hang", (status, res)
    f = hang_finding(res)
    assert signature(f) == f"HANG|cpu|_view_ops.py:view_groups:{label}", f
    assert f["step"] == 1 and f["expr"] == expr
    # the world restart is included; the 30 s program timeout is not reached
    assert took < 25, took


def test_collective_hang_names_the_missing_rank(world):
    # ranks 0-2 enter an all_reduce that rank 3 never joins; rank 3 waits in the harness's flag exchange instead
    expr = "torch.distributed.all_reduce(torch.ones(1)) if torch.distributed.get_rank() != 3 else None"
    status, res = world.run(_prog([expr], (2,)), 30)
    assert status == "hang", (status, res)
    assert signature(hang_finding(res)) == "HANG|collective|all_reduce|ranks-missing:[3]", res


def test_clean_program_after_hang(world):
    status, res = world.run(_prog(["v0 + 1"], (4,)), 30)
    assert status == "ok" and all(not r["findings"] for r in res), res


def test_rank_divergent_error_inside_a_collective_step_is_reported_as_such(world):
    # rank 3 raises before the all_reduce the others enter; without the error the deadlock would look like a hang
    expr = (
        "torch.distributed.all_reduce(v0.clone()) if torch.distributed.get_rank() != 3 "
        "else (_ for _ in ()).throw(RuntimeError('rank 3 only'))"
    )
    status, res = world.run(_prog([expr], (2,)), 30)
    assert status == "hang", (status, res)
    f = hang_finding(res)
    assert signature(f) == "RANK_DIVERGENT_ERROR|torch.distributed.al|RuntimeError: rank N only", f
