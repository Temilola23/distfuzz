import datetime
import json
import os
import socket
import tempfile

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from distfuzz.dtensor.fuzz import signature
from distfuzz.dtensor.world import World

pytestmark = pytest.mark.multirank


def _port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def _spawn(fn, *args):
    fd, out = tempfile.mkstemp(suffix=".json")
    os.close(fd)
    mp.spawn(_entry, args=(fn, out, args, _port()), nprocs=4, join=True)
    with open(out) as f:
        res = json.load(f)
    os.remove(out)
    return res


def _entry(rank, fn, out, args, port):
    from torch.distributed.device_mesh import init_device_mesh

    dist.init_process_group(
        "gloo",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=4,
        timeout=datetime.timedelta(seconds=30),
    )
    mesh = init_device_mesh("cpu", (4,))
    mesh2 = init_device_mesh("cpu", (2, 2))
    res = fn(rank, mesh, mesh2, *args)
    dist.barrier()
    if rank == 0:
        with open(out, "w") as f:
            json.dump(res, f)
    dist.destroy_process_group()


def _mk_all(rank, mesh, mesh2):
    from distfuzz.dtensor.runtime import Ctx, compare, gen_full, make_env

    bad = []
    for dtype in ("f32", "f64", "bf16", "f16", "i64", "i32"):
        for op in ("sum", "max", "min", "avg"):
            if op == "avg" and dtype in ("i64", "i32"):
                continue
            for pl, m in (
                (["P:" + op], mesh),
                (["P:" + op, "P:" + op], mesh2),
                (["P:" + op, "S0"], mesh2),
                (["S1", "P:" + op], mesh2),
                (["S0"], mesh),
                (["S1", "S1"], mesh2),
            ):
                spec = dict(shape=[7, 5], dtype=dtype, seed=11, pl=pl)
                d = make_env(Ctx("dist", m, [spec]))["mk"](0)
                diff = compare(gen_full(spec), d.full_tensor())
                if diff:
                    bad.append((dtype, pl, diff))
    return bad


def test_mk_partial_and_shard_construction_exact():
    assert _spawn(_mk_all) == []


def _r_alias(rank, mesh, mesh2):
    from torch.distributed.tensor import Replicate, Shard, distribute_tensor

    from distfuzz.dtensor.runtime import Ctx, make_env

    env = make_env(Ctx("dist", mesh, []))
    res = []
    for src in ([Replicate()], [Shard(0)], [Shard(1)]):
        x = distribute_tensor(torch.arange(16.0).reshape(4, 4), mesh, src)
        for tgt in (["R"], ["S0"], ["S1"]):
            y = env["R"](x, tgt)
            same = y.to_local().untyped_storage().data_ptr() == x.to_local().untyped_storage().data_ptr()
            res.append((str(src), tgt, y is x, same))
    return res


def test_redistribute_result_never_aliases_input():
    for src, tgt, is_self, same in _spawn(_r_alias):
        assert not is_self and not same, (src, tgt)


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    w = World(4, str(tmp_path_factory.mktemp("dtensor-logs")), cov=False)
    yield w
    w.close()


def _sigs(world, prog, timeout=30):
    status, res = world.run(prog, timeout)
    sigs = {}
    if res and status in ("ok", "aborted"):
        for r in res:
            for f in r.get("findings", []):
                sigs.setdefault(signature(f), []).append((r["rank"], f))
    return status, sigs


def test_worker_clean_program_has_no_findings(world):
    prog = dict(
        id=1,
        mesh="1d",
        inputs=[
            dict(shape=[8, 4], dtype="f32", seed=1, rg=True, pl=["S0"]),
            dict(shape=[8, 4], dtype="f32", seed=2, rg=True, pl=["R"]),
        ],
        steps=[
            dict(out="v0", expr="mk(0)"),
            dict(out="v1", expr="mk(1)"),
            dict(out="v2", expr="v0 * v1"),
            dict(out="v3", expr="v2.sum(dim=0)"),
            dict(out="v4", expr="R(v3, ['S0'])"),
            dict(out="v5", expr="BW(v4)"),
        ],
    )
    status, sigs = _sigs(world, prog)
    assert status == "ok" and sigs == {}, sigs


def test_worker_grad_check_is_keyed_by_input_index(world):
    # mk(0) fails on the DTensor path only; a positional zip of made inputs would compare mk(1)'s grad to mk(2)'s
    prog = dict(
        id=2,
        mesh="1d",
        inputs=[
            dict(shape=[2], dtype="bool", seed=1, rg=False, pl=["P:max"]),
            dict(shape=[3], dtype="f32", seed=2, rg=True, pl=["R"]),
            dict(shape=[3], dtype="f32", seed=3, rg=True, pl=["S0"]),
        ],
        steps=[
            dict(out="v0", expr="mk(0)"),
            dict(out="v1", expr="mk(1)"),
            dict(out="v2", expr="mk(2)"),
            dict(out="v3", expr="v1 * v1"),
            dict(out="v4", expr="BW(v3)"),
        ],
    )
    status, sigs = _sigs(world, prog)
    assert status == "ok"
    assert not [s for s in sigs if s.startswith("GRAD")], list(sigs)


@pytest.mark.parametrize(
    "spec",
    [
        dict(shape=[2], dtype="bool", seed=1, rg=False, pl=["P:sum"]),
        dict(shape=[4], dtype="i64", seed=1, rg=False, pl=["P:avg"]),
    ],
)
def test_worker_reports_input_construction_failure_as_fuzzer_error(world, spec):
    prog = dict(id=3, mesh="1d", inputs=[spec], steps=[dict(out="v0", expr="mk(0)"), dict(out="v1", expr="v0 * 1")])
    status, sigs = _sigs(world, prog)
    assert not any("|mk(" in s and not s.startswith("FUZZER_INPUT_ERR") for s in sigs), list(sigs)


def test_worker_detects_rank_divergent_error(world):
    # A6: full max() with empty shards on ranks 2 and 3
    prog = dict(
        id=5,
        mesh="1d",
        inputs=[dict(shape=[2], dtype="f32", seed=1, rg=False, pl=["S0"])],
        steps=[dict(out="v0", expr="mk(0)"), dict(out="v1", expr="v0.max()")],
    )
    status, sigs = _sigs(world, prog)
    assert status == "aborted"
    assert any(s.startswith("RANK_DIVERGENT_ERROR|max") for s in sigs), list(sigs)


def test_worker_step_attribution_is_stable_after_desync(world):
    # step 2 references an undefined name and can never run; sync flags use their own process group, so a
    # rank still inside a DTensor collective must not pair with another rank's flag exchange
    prog = dict(
        id=6,
        mesh="1d",
        inputs=[dict(shape=[1, 1], dtype="i64", seed=1, rg=False, pl=["S0"])],
        steps=[
            dict(out="v0", expr="mk(0)"),
            dict(out="v3", expr="v0.max(dim=-2)[0]"),
            dict(out="v4", expr="v2.clone()"),
        ],
    )
    for _ in range(3):
        _, sigs = _sigs(world, prog)
        steps = {f["step"] for lst in sigs.values() for _, f in lst}
        assert 2 not in steps, sigs


def test_worker_setitem_sharded_is_real_bug(world):
    # A1, confirmed through the worker without any redistribute
    prog = dict(
        id=7,
        mesh="1d",
        inputs=[dict(shape=[9], dtype="f32", seed=1, rg=False, pl=["S0"])],
        steps=[dict(out="v0", expr="mk(0)"), dict(out="v1", expr="SETITEM(v0, 6, 5)")],
    )
    _, sigs = _sigs(world, prog)
    assert any(s.startswith("MISMATCH|SETITEM") for s in sigs), list(sigs)
