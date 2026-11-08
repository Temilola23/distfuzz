"""Training profile on real ranks: rediscover the two production-relevant bugs and keep clean configs clean."""

import os
import random
import tempfile

import pytest

from distfuzz.dcp.fuzz import CK_ROOT, Pools, run_scenario, signature
from distfuzz.dcp.train_gen import TrainGen

pytestmark = pytest.mark.multirank


def base(**kw):
    s = dict(
        profile="train",
        tmpl="accel-fsdp2",
        seed=3,
        dtype="f64",
        batch=24,
        zg_none=True,
        flat_osd=False,
        accum=1,
        no_sync=False,
        ac="none",
        clip=None,
        model=dict(
            kind="vec", seq=2, d0=8, head_bias=True, out=4, frozen=[], tie=False, lora_only=False,
            blocks=[dict(t="lin", out=5, bias=True, act="tanh")],
        ),
        optim=dict(name="adamw", lr=1e-3, wd=0.0, foreach=None, fused=False, amsgrad=False),
    )  # fmt: skip
    s.update(kw)
    return s


FS = dict(units=["0"], raf=True, raf_root=True, resave=False)


@pytest.fixture(scope="module")
def pools():
    p = Pools(tempfile.mkdtemp(prefix="trainmr_"), cov=False)
    yield p
    p.close()


def run(pools, scn):
    assert TrainGen(random.Random(0)).valid(scn)
    tags = {}
    fs, _ = run_scenario(pools, scn, os.path.join(CK_ROOT, "trainmr"), timeout=30, tags=tags)
    return [signature(dict(f, tags=dict(tags)), scn) for f in fs]


@pytest.mark.parametrize("out", [5, 8])
def test_inf_norm_clip_with_an_empty_fsdp_shard(pools, out):
    s = base(clip=dict(max=1.0, norm="inf", foreach=None, impl="torch"))
    s["model"]["blocks"][0]["out"] = out
    s["segs"] = [dict(w=4, par="fsdp", steps=2, **FS)]
    sigs = run(pools, s)
    if out == 5:  # dim 0 of 5 on 4 ranks: shards 2, 2, 1, 0
        assert sigs and sigs[0].startswith("TRAIN|PARTIAL_ERR|train:clip|fsdp2|clip:inf|clip-impl:torch|empty-shard|")
        assert "infinity norm on an empty tensor" in sigs[0]
    else:
        assert sigs == []


@pytest.mark.parametrize("mode", ["full_bcast", "full_bcast_split"])
def test_broadcast_from_rank0_joint_set_state_dict_hangs(pools, mode):
    s = base(optim=dict(name="sgd", lr=0.01, wd=0.0, foreach=None, fused=False, momentum=0.9, nesterov=False))
    s["segs"] = [dict(w=2, par="fsdp", steps=2, **FS), dict(w=2, par="fsdp", steps=2, **FS)]
    TrainGen(random.Random(0)).fix_chain_min(s)
    s["segs"][0]["save"]["mode"] = "full_offload"
    s["segs"][1]["load"]["mode"] = mode
    sigs = run(pools, s)
    if mode == "full_bcast":
        assert len(sigs) == 1 and sigs[0].startswith("TRAIN|HANG|load|fsdp2|load:sd:full_bcast|")
        assert "state_dict.py:set_state_dict > distributed_c10d.py:broadcast" in sigs[0]
    else:  # the split API (torchtune, accelerate) loads cleanly
        assert sigs == []


@pytest.mark.parametrize("impl", ["functional", "c10d"])
def test_expert_parallel_moe_matches_the_reference(pools, impl):
    blocks = [
        dict(t="ln", affine=True, bias=False, rms=True),
        dict(t="moe", E=8, topk=1, eh=4, score="softmax", route_norm=False, shared=True),
    ]
    s = base(
        model=dict(
            kind="tok", seq=2, d0=8, head_bias=False, vocab=7, out=7, frozen=[], tie=False, lora_only=False,
            blocks=blocks,
        ),
        clip=dict(max=1.0, norm=2.0, foreach=None, impl="titan"),
        accum=2,
    )  # fmt: skip
    s["segs"] = [dict(w=4, par="ep", steps=2, units=["1"], raf=True, raf_root=True, a2a=impl, resave=False)]
    assert run(pools, s) == []
