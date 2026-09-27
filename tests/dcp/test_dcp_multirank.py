import random

import pytest

from distfuzz.dcp.gen import Gen
from distfuzz.dcp.standalone import run_standalone

pytestmark = pytest.mark.multirank


def fsdp_to_tp(flat=True):
    scn = dict(
        seed=1,
        dtype="f64",
        batch=12,
        model=dict(
            kind="vec", seq=2, d0=4, head_bias=True, out=3, frozen=[],
            blocks=[dict(t="mlp", h=8, bias=True, act="tanh")],
        ),
        optim=dict(name="adam", lr=0.01, wd=0.0, foreach=None, fused=False, amsgrad=False),
        clip=None,
        zg_none=True,
        flat_osd=False,
        segs=[
            dict(w=2, steps=2, par="fsdp", units=["0"], raf=True, raf_root=True, resave=False),
            dict(w=2, steps=1, par="tp", tp={"0": "mlp"}, resave=False),
        ],
    )  # fmt: skip
    g = Gen(random.Random(0))
    g.fix_chain_min(scn)
    scn["segs"][0]["save"]["planner"]["flat"] = flat
    scn["segs"][1]["load"]["planner"]["flat"] = flat
    assert g.valid(scn)
    return scn


def test_fsdp_to_tp_reshard_is_clean():
    assert run_standalone(fsdp_to_tp(), timeout=90, verbose=False) == []


def test_unflattened_load_drops_state():
    fs = run_standalone(fsdp_to_tp(flat=False), timeout=90, verbose=False)
    assert fs, "flatten_state_dict=False load restored everything (bug A1 fixed?)"
    assert {f["kind"] for f in fs} & {"LOAD_MISMATCH", "LOAD_EXTRA", "ROUNDTRIP_MISMATCH", "ERR"}
