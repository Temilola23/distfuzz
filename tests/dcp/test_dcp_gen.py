import copy
import json
import random

import pytest

from distfuzz.dcp.fuzz import signature
from distfuzz.dcp.gen import PARS, Gen, divisors
from distfuzz.dcp.minimize import key
from distfuzz.dcp.report import bucket
from distfuzz.dcp.runtime import model_dims_ok


def scenarios(n, seed=0, max_w=4):
    g = Gen(random.Random(seed), max_w=max_w)
    return g, [g.generate() for _ in range(n)]


def size(scn):
    return (
        len(scn["segs"])
        + sum(s["w"] + s["steps"] for s in scn["segs"])
        + len(scn["model"]["blocks"])
        + sum(len(s.get("tp", {})) + len(s.get("units", [])) for s in scn["segs"])
    )


def test_divisors():
    assert divisors(1) == [1]
    assert divisors(4) == [1, 2, 4]
    assert divisors(6) == [1, 2, 3, 6]


@pytest.mark.parametrize("seed", [0, 1, 7])
def test_generate_is_deterministic_per_seed(seed):
    _, a = scenarios(20, seed)
    _, b = scenarios(20, seed)
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


def test_generated_scenarios_are_valid_chains():
    g, scns = scenarios(300)
    for s in scns:
        assert g.valid(s)
        assert model_dims_ok(s["model"])
        segs = s["segs"]
        assert 2 <= len(segs) <= 4
        assert "load" not in segs[0] and "save" not in segs[-1]
        for i, seg in enumerate(segs):
            assert 1 <= seg["w"] <= 4
            assert seg["par"] in PARS
            if seg["par"] == "none":
                assert seg["w"] == 1
            if seg["par"] in ("fsdp_tp", "hsdp"):
                a, b = seg["mesh2d"]
                assert a * b == seg["w"]
            if i < len(segs) - 1:
                assert "save" in seg
            if i > 0:
                assert Gen.load_ok(seg, segs[i - 1]["save"])


def test_generator_covers_every_parallelism():
    _, scns = scenarios(300)
    seen = {seg["par"] for s in scns for seg in s["segs"]}
    assert seen == set(PARS)


def test_max_world_is_respected():
    _, scns = scenarios(100, max_w=2)
    assert max(seg["w"] for s in scns for seg in s["segs"]) <= 2


def test_mutate_keeps_validity_and_leaves_parent_alone():
    g, scns = scenarios(50, seed=3)
    for parent in scns:
        before = copy.deepcopy(parent)
        for _ in range(4):
            child = g.mutate(parent)
            assert g.valid(child)
            assert "id" not in child
        assert parent == before


def test_load_must_match_save_flattening():
    g, scns = scenarios(30, seed=5)
    s = next(x for x in scns if x["segs"][1]["load"]["api"] != "stateful")
    s["segs"][1]["load"]["planner"]["flat"] = not s["segs"][0]["save"]["planner"]["flat"]
    assert not g.valid(s)


def test_simplifications_are_valid_and_never_grow():
    g, scns = scenarios(40, seed=11)
    total = 0
    for s in scns:
        for cand in g.simplifications(s):
            total += 1
            assert g.valid(cand)
            assert cand != s
            assert size(cand) <= size(s)
    assert total > 0


def test_simplifications_can_drop_a_segment():
    g, scns = scenarios(40, seed=2)
    s = next(x for x in scns if len(x["segs"]) > 2)
    assert any(len(c["segs"]) < len(s["segs"]) for c in g.simplifications(s))


def test_signature_ignores_numbers():
    a = dict(kind="ERR", phase="load", msg="KeyError: 'param_groups.0.params.3'")
    b = dict(kind="ERR", phase="load", msg="KeyError: 'param_groups.1.params.7'")
    assert signature(a) == signature(b)
    m = dict(kind="LOAD_MISMATCH", pair="save->load", cats=["p", "o"], msg="x 1")
    assert signature(m) == "LOAD_MISMATCH|save->load|p,o"


def test_minimize_key_matches_across_numbers():
    a = dict(kind="ERR", phase="load", msg="KeyError: 'param_groups.0.params.3'\ntraceback")
    b = dict(kind="ERR", phase="load", msg="KeyError: 'param_groups.2.params.9'")
    assert key(a) == key(b)
    assert key(dict(kind="HANG", msg="segment timeout")) == ("HANG",)


def test_report_buckets_unflattened_load_as_a1():
    g, scns = scenarios(30, seed=5)
    s = scns[0]
    s["segs"][0]["save"]["planner"]["flat"] = False
    rec = dict(sig="LOAD_MISMATCH|x|p", finding=dict(kind="LOAD_MISMATCH", seg=1), scn=s)
    assert bucket(rec).startswith("A1 ")


def test_emitted_repro_is_self_contained(tmp_path):
    from distfuzz.dcp.minimize import emit

    _, scns = scenarios(1, seed=4)
    path = tmp_path / "repro.py"
    emit(scns[0], ("ERR", "load", "KeyError"), str(path))
    src = path.read_text()
    compile(src, str(path), "exec")
    assert 'if "run_segment" not in globals()' not in src
    assert "def run_segment(" in src and "def run_standalone(" in src
