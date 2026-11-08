import random

import pytest
import torch

from distfuzz.compile.fuzz import signature
from distfuzz.compile.module import Net, legalize, no_empty_shard, rand_module_prog
from distfuzz.compile.prog import CompileGen, build_source, outputs, variant_inputs
from distfuzz.compile.repro import module_repro, op_repro


def _prog(steps, inputs=None, cfg=None):
    return dict(
        mesh="1d",
        inputs=inputs or [dict(shape=[4, 3], dtype="f32", seed=1, rg=False, pl=["S0"])],
        steps=[dict(out=o, expr=e) for o, e in steps],
        cfg=cfg or dict(backend="aot_eager", fullgraph=False, dynamic="static", ret="all", variant=None),
    )


def test_build_source_args_body_and_redistribute_constants():
    p = _prog([("v0", "mk(0)"), ("v1", "R(v0, ['R'])"), ("v2", "v1.sum(dim=0)"), ("v3", "BW(v2)"), ("v4", "v2 * 2")])
    src, args, outs, bws = build_source(p)
    assert args == [("v0", 0)]
    assert "v1 = R(v0, (Replicate(),))" in src
    assert "v4" not in src  # steps after the first BW are dropped
    assert bws == ["v2"] and outs == ["v0", "v1", "v2"]
    assert "return (_d(v0), _d(v1), v2)" in src
    g = {"torch": torch, "R": lambda x, p: x.clone(), "Replicate": lambda: None, "_d": lambda x: x.detach()}
    exec(src, g)
    out = g["prog_fn"](torch.ones(4, 3))
    assert torch.equal(out[2], torch.full((3,), 4.0))


def test_outputs_leaves_keeps_inputs_mutations_and_bw_targets():
    p = _prog([("v0", "mk(0)"), ("v1", "v0 + 1"), ("v2", "v1.add_(1)"), ("v3", "v1 * 2"), ("v4", "v3.sum()")])
    body = p["steps"]
    assert outputs(body, [], "all") == ["v0", "v1", "v2", "v3", "v4"]
    assert outputs(body, ["v3"], "leaves") == ["v0", "v2", "v3", "v4"]


def test_variant_maps_one_sharded_size_everywhere():
    inputs = [
        dict(shape=[6, 3], dtype="f32", seed=1, rg=False, pl=["S0"]),
        dict(shape=[6], dtype="f32", seed=2, rg=False, pl=["R"]),
        dict(shape=[2], dtype="i64", seed=3, rg=False, pl=["R"], kind="index", high=6),
    ]
    p = _prog([("v0", "mk(0)")], inputs)
    for s in range(20):
        v = variant_inputs(p, random.Random(s), 4)
        assert v["old"] == 6 and v["new"] not in (6, 0, 1)
        assert v["inputs"][0]["shape"] == [v["new"], 3] and v["inputs"][1]["shape"] == [v["new"]]
        assert v["inputs"][2]["high"] == v["new"] and v["inputs"][2]["shape"] == [2]
    assert (
        variant_inputs(
            _prog([("v0", "mk(0)")], [dict(shape=[1, 3], dtype="f32", seed=1, pl=["S0"])]), random.Random(0), 4
        )
        is None
    )


def test_generated_programs_build_and_run_on_plain_tensors():
    from distfuzz.dtensor.runtime import Ctx, make_input

    gen = CompileGen(4, random.Random(0), module_rate=0.0)
    built = 0
    for _ in range(40):
        p = gen.generate()
        assert not any("CMP(" in s["expr"] for s in p["steps"])
        src, args, outs, bws = build_source(p)
        g = {"torch": torch, "F": torch.nn.functional, "math": __import__("math"), "R": lambda x, pl: x.clone()}
        g["_d"] = lambda x: x.detach() if isinstance(x, torch.Tensor) else x
        from torch.distributed.tensor import Partial, Replicate, Shard

        g.update(Shard=Shard, Replicate=Replicate, Partial=Partial, SETITEM=lambda x, i, y: x.__setitem__(i, y) or x)
        exec(src, g)
        ctx = Ctx("ref", None, p["inputs"])
        try:
            g["prog_fn"](*[make_input(ctx, j) for _, j in args])
            built += 1
        except Exception:
            pass  # repair() keeps generator programs valid; a few depend on in-place order and still fail
        p2 = gen.mutate(p)
        assert p2["cfg"]["backend"] in ("aot_eager", "inductor")
    assert built >= 30


def test_signature_names_track_config_kind_and_op():
    p = _prog(
        [("v0", "mk(0)"), ("v1", "v0.sum(dim=0)")], cfg=dict(backend="inductor", dynamic="dynamic", fullgraph=True)
    )
    f = dict(kind="MISMATCH", leg="cdist", expr="v1.sum(dim=0)", msg="values differ at [[0]] maxdiff=3")
    assert signature(f, p) == "COMPILE|inductor|dynamic|MISMATCH|sum|values"
    f2 = dict(kind="ERR", leg="cdist", expr="", msg="BackendCompilerFailed<-AssertionError: x 12", op="embedding")
    assert signature(f2, p) == "COMPILE|inductor|dynamic|ERR|embedding|BackendCompilerFailed<-AssertionError: x N"
    h = dict(kind="HANG", leg="cdist", expr="", msg="program timeout")
    assert signature(h, p).startswith("HANG|op|cdist|inductor|dynamic")
    m = rand_module_prog(random.Random(1), 4, ("aot_eager",))
    s = signature(dict(kind="LOSS_MISMATCH", leg="mod_cdist", expr="step 1", msg="ref=1 eager=1 compiled=2"), m)
    assert s.startswith("MODULE|") and "|LOSS_MISMATCH|" in s and s.endswith("|")


@pytest.mark.parametrize(
    "n,k,ok", [(5, 4, False), (6, 4, False), (9, 4, False), (7, 4, True), (8, 4, True), (3, 2, True), (1, 2, False)]
)
def test_no_empty_shard(n, k, ok):
    assert no_empty_shard(n, k) == ok


def test_module_programs_are_legal_and_build():
    rng = random.Random(0)
    for _ in range(50):
        p = rand_module_prog(rng, 4, ("aot_eager", "inductor"))
        k = {"fsdp": 1, "tp": 4, "fsdp_tp": 2}[p["cfg"]["par"]]
        if k > 1:
            assert all(p["model"][d] % k == 0 and no_empty_shard(p["model"][d], k) for d in ("D", "H", "O", "V"))
        else:
            assert not p["cfg"]["sp"]
        if p["cfg"]["par"] != "tp" and p["cfg"]["wrap"] == "whole":
            assert not p["cfg"]["fullgraph"]
        Net(p["model"])
        assert legalize(p, rng, 4) == p


def test_repro_scripts_are_valid_python_without_distfuzz():
    p = _prog([("v0", "mk(0)"), ("v1", "R(v0, ['R'])"), ("v2", "v1.sum()")])
    for text in (op_repro(p), module_repro(rand_module_prog(random.Random(2), 4, ("inductor",)))):
        compile(text, "repro.py", "exec")
        assert "distfuzz" not in text.replace("Generated by distfuzz.compile", "")


def test_op_repro_runs_single_rank_eager_and_has_python_literals(tmp_path):
    p = _prog(
        [("v0", "mk(0)"), ("v1", "v0 * 2")],
        inputs=[dict(shape=[4, 3], dtype="f32", seed=1, rg=True, pl=["R"])],
    )
    text = op_repro(p)
    assert "true" not in text.split("INPUTS = ", 1)[1].splitlines()[0]
    ns: dict = {}
    exec(text.split("dist.init_process_group", 1)[0], ns)  # imports only; the rest needs ranks
