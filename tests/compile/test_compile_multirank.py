import pytest

from distfuzz.compile.fuzz import collect, signature
from distfuzz.compile.world import World

pytestmark = pytest.mark.multirank

CFG = dict(backend="aot_eager", fullgraph=False, dynamic="static", ret="all", variant=None)


def _run(tmp_path, prog):
    w = World(4, str(tmp_path / "logs"), pg_timeout=60)
    try:
        status, res = w.run(prog, 120)
        findings, infos = collect(status, res, w, prog)
    finally:
        w.close()
    return status, [signature(f, prog) for f in findings], infos


def test_clean_program_has_no_findings(tmp_path):
    prog = dict(
        track="op",
        mesh="1d",
        cfg=dict(CFG),
        inputs=[dict(shape=[6, 3], dtype="f32", seed=1, rg=True, pl=["S0"])],
        steps=[
            dict(out="v0", expr="mk(0)"),
            dict(out="v1", expr="torch.relu(v0) + 1"),
            dict(out="v2", expr="R(v1, ['R'])"),
            dict(out="v3", expr="v2.sum(dim=0)"),
            dict(out="v4", expr="BW(v3)"),
        ],
    )
    status, sigs, infos = _run(tmp_path, prog)
    assert status == "ok" and sigs == []
    assert infos[0]["graphs"] >= 1 and not infos[0].get("skip")


def test_known_compiled_embedding_error_is_reported(tmp_path):
    # pytorch#159843 / #160697: a row-sharded embedding under torch.compile fails while eager works
    prog = dict(
        track="op",
        mesh="1d",
        cfg=dict(CFG),
        inputs=[
            dict(shape=[8, 3], dtype="f32", seed=1, rg=False, pl=["S0"]),
            dict(shape=[5], dtype="i64", seed=2, rg=False, pl=["R"], kind="index", high=8),
        ],
        steps=[dict(out="v0", expr="mk(0)"), dict(out="v1", expr="mk(1)"), dict(out="v2", expr="F.embedding(v1, v0)")],
    )
    status, sigs, _ = _run(tmp_path, prog)
    assert status == "ok"
    assert any(s.startswith("COMPILE|aot_eager|static|ERR|") for s in sigs), sigs


def test_variant_leg_reports_stale_global_shape(tmp_path):
    # VERIFIED.md section 52: 7 -> 8 rows keeps the (2, 3) local shard on ranks 0-2, which reuse the first graph
    x7 = dict(shape=[7, 3], dtype="f32", seed=1, rg=False, pl=["S0"])
    x8 = dict(x7, shape=[8, 3])
    prog = dict(
        track="op",
        mesh="1d",
        cfg=dict(CFG, variant=dict(old=7, new=8, inputs=[x8])),
        inputs=[x7],
        steps=[dict(out="v0", expr="mk(0)"), dict(out="v1", expr="v0 * 2")],
    )
    status, sigs, _ = _run(tmp_path, prog)
    assert status == "ok"
    assert "COMPILE|aot_eager|static|VARIANT_MISMATCH|*|shape" in sigs, sigs
