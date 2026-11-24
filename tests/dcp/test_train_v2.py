"""Host tests for the training profile's second round: known-bug predictor, SP / PP / compile / FSDP knobs,
torchtitan checkpoint API pairing."""

import copy
import random

import pytest

from distfuzz.dcp import known
from distfuzz.dcp.train_gen import MULTI, TrainGen, is_pp, pp_units_of


def gen(n, seed=0):
    g = TrainGen(random.Random(seed))
    return g, [g.generate() for _ in range(n)]


def test_predictor_flags_known_configs():
    _, scns = gen(400, 1)
    flagged = [s for s in scns if known.predict(s)]
    assert 0 < len(flagged) < len(scns) // 2
    for s in flagged:
        for k in known.predict(s):
            assert k[0] in "KNL"


def test_predictor_joint_bcast_and_inf_clip():
    g, scns = gen(300, 2)
    s = next(x for x in scns if len(x["segs"]) > 1 and not is_pp(x["segs"][1]) and x["segs"][1]["load"]["api"] == "sd")
    s = copy.deepcopy(s)
    s["segs"][1]["load"]["mode"] = "full_bcast"
    assert "K6-joint-bcast-load" in known.predict(s)


def test_classify_mirrors_triage_rules():
    f = dict(kind="PARTIAL_ERR", msg="RuntimeError: _foreach_norm cannot compute the infinity norm on an empty tensor")
    sig = "TRAIN|PARTIAL_ERR|train:clip|fsdp2|clip:inf|clip-impl:torch|empty-shard|" + f["msg"]
    assert known.classify(sig, f, {}) == "K30-inf-clip-empty-shard"
    assert known.classify("TRAIN|ERR|train:fwd|pp|something new", dict(kind="ERR"), {}) is None


def test_sp_segments_are_complete_plans():
    _, scns = gen(1500, 3)
    sp = [(s, g) for s in scns for g in s["segs"] if g.get("sp")]
    assert sp, "no sequence-parallel segment generated"
    for s, g in sp:
        m = s["model"]
        k = g["w"] if g["par"] == "tp" else g["mesh2d"][1]
        assert m["kind"] == "tok" and m["seq"] % k == 0
        assert all(str(i) in g["tp"] for i in range(len(m["blocks"])))
        assert "emb" in g["tp"] and "head" in g["tp"]


def test_pp_segments_valid():
    _, scns = gen(1500, 4)
    pps = [(s, g) for s in scns for g in s["segs"] if is_pp(g)]
    assert {g["pp"]["sched"] for _, g in pps} & set(MULTI)
    for s, g in pps:
        pp, dp = g["ppmesh"]
        assert pp * dp == g["w"]
        assert len(pp_units_of(s)) >= pp * g["pp"]["vps"]
        assert s.get("accum", 1) == 1 and not s["model"].get("tie")
        for k in ("save", "load"):
            if k in g:
                assert g[k]["api"] == "titan"


def test_titan_api_pairs_and_named_optimizer():
    _, scns = gen(800, 5)
    for s in scns:
        for i, g in enumerate(s["segs"][1:], 1):
            assert (g["load"]["api"] == "titan") == (s["segs"][i - 1]["save"]["api"] == "titan")
        if any(g.get("save", {}).get("api") == "titan" for g in s["segs"]):
            assert s["named_opt"]


def test_pp_split_and_stage_ids():
    torch = pytest.importorskip("torch")  # noqa: F841
    from distfuzz.dcp.runtime import pp_split, pp_stage_ids

    scn = dict(model=dict(kind="tok", blocks=[{}] * 5))
    sp = pp_split(scn, 4)
    assert [u for st in sp for u in st] == ["emb", "0", "1", "2", "3", "4", "head"]
    assert all(sp) and max(map(len, sp)) - min(map(len, sp)) <= 1
    assert pp_stage_ids(0, 2, "Interleaved1F1B", 4) == (0, 2)
    assert pp_stage_ids(1, 2, "ZBVZeroBubble", 4) == (1, 2)
    assert pp_stage_ids(1, 2, "GPipe", 2) == (1,)


def test_mutation_keeps_v2_scenarios_valid():
    g, scns = gen(50, 6)
    for s in scns:
        for _ in range(5):
            s = g.mutate(s)
            assert g.valid(s)


def _saves(device):
    g = TrainGen(random.Random(3), device=device)
    return [s["save"] for _ in range(150) for s in g.generate()["segs"] if s.get("save")]


def test_cuda_generates_no_async_saves():
    # thread async_save needs a CPU backend, and the workers' groups are NCCL-only under cuda
    assert any(sv.get("async", "none") != "none" for sv in _saves("cpu"))
    cuda = _saves("cuda")
    assert cuda and all(sv.get("async", "none") == "none" for sv in cuda)


@pytest.mark.parametrize("cc,ok", [((7, 5), False), ((8, 0), False), ((8, 9), True), ((9, 0), True)])
def test_float8_needs_compute_capability_8_9_on_cuda(monkeypatch, cc, ok):
    import torch

    from distfuzz.dcp import train_gen

    monkeypatch.setattr(train_gen, "HAVE_AO", True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a: cc)
    assert train_gen.float8_ok("cuda") is ok and train_gen.float8_ok("cpu")
    g = TrainGen(random.Random(0), device="cuda")
    assert g.f8_ok is ok
    scn = g.generate()
    scn["f8"] = dict(ag=True, precompute=False)
    if not ok:
        assert not g.valid(scn)
