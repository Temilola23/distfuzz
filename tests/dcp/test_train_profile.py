import copy
import json
import random

import pytest

from distfuzz.dcp.blocks import chunk_sizes
from distfuzz.dcp.fuzz import axis_tags, hang_finding, signature
from distfuzz.dcp.templates import SRC, TEMPLATES, provenance
from distfuzz.dcp.train_gen import TrainGen
from distfuzz.dcp.world import stack_location


def scenarios(n, seed=0, max_w=4):
    g = TrainGen(random.Random(seed), max_w=max_w)
    return g, [g.generate() for _ in range(n)]


def test_chunk_sizes_follow_torch_chunk():
    torch = pytest.importorskip("torch")
    for n in range(13):
        for w in range(1, 5):
            want = [len(c) for c in torch.arange(n).chunk(w)] if n else []
            want += [0] * (w - len(want))
            assert chunk_sizes(n, w) == want


def test_every_template_has_provenance():
    for name, t in TEMPLATES.items():
        assert t["src"] and all(k in SRC for k in t["src"])
        assert all("@" in p for p in provenance(name))


@pytest.mark.parametrize("seed", [0, 5])
def test_generate_is_deterministic(seed):
    _, a = scenarios(30, seed)
    _, b = scenarios(30, seed)
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


def test_generated_training_scenarios_are_valid():
    g, scns = scenarios(300)
    tmpls, pars, norms = set(), set(), set()
    for s in scns:
        assert s["profile"] == "train" and g.valid(s)
        tmpls.add(s["tmpl"])
        assert s["batch"] % s["accum"] == 0
        if s["clip"]:
            norms.add(s["clip"]["norm"])
        for seg in s["segs"]:
            pars.add(seg["par"])
            assert 2 <= seg["w"] <= 4
            if seg["par"] == "ep":
                assert all(b["E"] % seg["w"] == 0 for b in s["model"]["blocks"] if b["t"] == "moe")
                assert not seg.get("mp")
            if seg["par"] in ("tp", "fsdp_tp"):
                assert not any(b["t"] in ("moe", "lora") for b in s["model"]["blocks"])
    assert tmpls == set(TEMPLATES)
    assert {"fsdp", "ddp", "tp", "fsdp_tp", "ep"} <= pars
    assert {2.0, "inf"} <= norms


def test_mutate_keeps_validity():
    g, scns = scenarios(30, seed=2)
    for parent in scns:
        before = copy.deepcopy(parent)
        child = g.mutate(parent)
        assert g.valid(child) and child["profile"] == "train"
        assert parent == before


def test_simplifications_are_valid_and_shrink_sizes():
    g, scns = scenarios(30, seed=4)
    shrank = False
    for s in scns:
        for c in g.simplifications(s):
            assert g.valid(c)
            if c["model"]["d0"] < s["model"]["d0"]:
                shrank = True
    assert shrank


def _inf_clip_finding():
    scn = dict(
        profile="train",
        clip=dict(max=1.0, norm="inf", foreach=None, impl="torch"),
        optim=dict(name="adamw"),
        model=dict(blocks=[dict(t="lin")]),
        segs=[dict(w=4, par="fsdp")],
    )
    f = dict(
        kind="PARTIAL_ERR",
        phase="train:clip",
        seg=0,
        msg="RuntimeError: _foreach_norm cannot compute the infinity norm on an empty tensor",
        tags=dict(empty_shard=True),
    )
    return f, scn


def test_signature_names_the_axes_of_the_inf_norm_crash():
    f, scn = _inf_clip_finding()
    sig = signature(f, scn)
    assert sig.startswith("TRAIN|PARTIAL_ERR|train:clip|fsdp2|clip:inf|clip-impl:torch|empty-shard|")
    f2 = dict(f, msg=f["msg"] + " on rank 3")
    assert signature(f2, scn).split("|")[:7] == sig.split("|")[:7]


def test_axis_tags_depend_on_phase():
    f, scn = _inf_clip_finding()
    assert "clip:inf" in axis_tags(f, scn)
    fwd = dict(f, phase="train:fwd")
    assert not any(t.startswith("clip") for t in axis_tags(fwd, scn))


def test_hang_with_one_rank_raising_is_a_partial_error():
    rep = [
        dict(rank=0, returned=False, err=None, phase="train:clip", loc="clip_grad.py:x > c10d.py:wait"),
        dict(rank=1, returned=False, err="RuntimeError: boom", err_phase="train:clip", phase="train:clip", loc=None),
    ]
    f = hang_finding(rep, dict(seg=0, par="fsdp", w=2))
    assert f["kind"] == "PARTIAL_ERR" and f["phase"] == "train:clip" and f["err_ranks"] == [1]
    rep[1]["err"] = None
    rep[1]["phase"] = "load"
    h = hang_finding(rep, dict(seg=0, par="fsdp", w=2))
    assert h["kind"] == "HANG" and h["phase"] == "load"


def test_stack_location_reads_the_current_thread():
    dump = """Thread 0x00007f (most recent call first):
  File "/usr/lib/python3.12/threading.py", line 359 in wait

Current thread 0x00007e (most recent call first):
  File "/x/torch/distributed/distributed_c10d.py", line 2913 in broadcast
  File "/x/torch/distributed/checkpoint/state_dict.py", line 609 in _broadcast_state_dict
  File "/x/torch/distributed/checkpoint/state_dict.py", line 1537 in set_state_dict
  File "/src/distfuzz/dcp/runtime.py", line 600 in load_ckpt
  File "/src/distfuzz/dcp/worker.py", line 73 in main
"""
    assert stack_location(dump) == "state_dict.py:set_state_dict > distributed_c10d.py:broadcast"


def test_clip_variants_agree_on_plain_tensors():
    torch = pytest.importorskip("torch")
    from distfuzz.dcp.blocks import clip_grads

    for norm in (2.0, 1.0, "inf", 3.0):
        got = []
        for impl in ("torch", "titan", "accel"):
            torch.manual_seed(0)
            ps = [torch.nn.Parameter(torch.randn(3, 2)), torch.nn.Parameter(torch.randn(4))]
            for p in ps:
                p.grad = torch.randn_like(p)
            got.append(clip_grads(ps, dict(max=0.5, norm=norm, impl=impl)))
        assert got[0] == pytest.approx(got[1]) == pytest.approx(got[2])


def test_moe_routes_every_token_once():
    torch = pytest.importorskip("torch")
    from distfuzz.dcp.blocks import MoE

    torch.manual_seed(0)
    b = dict(E=4, topk=2, eh=3, score="softmax", route_norm=True, shared=False)
    moe = MoE(5, b).double()
    x = torch.randn(2, 3, 5, dtype=torch.float64)
    y = moe(x)
    # dense check: every token through its top-k experts, weighted
    top, idx = moe.router(x.reshape(-1, 5))
    ref = torch.zeros(6, 5, dtype=torch.float64)
    ex = moe.experts
    for t in range(6):
        for j in range(2):
            e = int(idx[t, j])
            xe = x.reshape(-1, 5)[t]
            h = torch.nn.functional.silu(xe @ ex.w1[e].T) * (xe @ ex.w3[e].T)
            ref[t] += top[t, j] * (h @ ex.w2[e].T)
    assert torch.allclose(y, x + ref.reshape(2, 3, 5))
