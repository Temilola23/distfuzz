"""C++ coverage from real Gloo ranks. Runs only in the distfuzz:cov image (docker/Dockerfile.cov)."""

import pytest
import torch  # noqa: F401  (loads the instrumented libraries before the availability check)

from distfuzz.collectives import cxxcov
from distfuzz.collectives.executor import Session

pytestmark = [
    pytest.mark.multirank,
    pytest.mark.skipif(not cxxcov.available(), reason="needs torch built by docker/Dockerfile.cov"),
]


def T(dtype="float32", shape=(4,)):
    return {"dtype": dtype, "shape": list(shape), "layout": "contig", "seed": 1, "kind": "int"}


def C(op, args, rets=None):
    return {"op": op, "args": args, "rets": rets or {}, "div": {}}


def prog(*calls, shape=(4,)):
    return {"world": 4, "calls": [C("tensor", {"spec": T(shape=shape)}, {"out": "t1"}), *calls]}


ALL_REDUCE = C("all_reduce", {"t": "t1", "op": "SUM", "group": "world", "async_op": False})
BROADCAST = C("broadcast", {"t": "t1", "root": 0, "root_mode": "global", "group": "world", "async_op": False})


@pytest.fixture(scope="module")
def sess():
    s = Session(4, timeout=5.0, coverage="py+cxx")
    s.start()
    yield s
    s.stop(hard=True)


def run(sess, p):
    res = sess.run(p)
    assert res["kind"] == "ok", res
    cov = set().union(*(r.get("cov", set()) for r in res["results"]))
    return {f for f in cov if isinstance(f, int)}, {f for f in cov if not isinstance(f, int)}


def test_collectives_reach_instrumented_gloo_code(sess):
    base, _ = run(sess, prog())
    ar, _ = run(sess, prog(ALL_REDUCE))
    bc, _ = run(sess, prog(BROADCAST))
    assert ar - base and bc - base, "a collective added no C++ edges over a program without one"
    assert ar - base != bc - base


def test_counters_are_cleared_between_programs(sess):
    """Edges all_reduce reliably hits never show up in a no-op program run right after it. Background threads
    make the no-op program's own coverage vary, so it is compared against the union of many no-op runs."""
    noise = set().union(*(run(sess, prog())[0] for _ in range(30)))
    ar = set.intersection(*(run(sess, prog(ALL_REDUCE))[0] for _ in range(3))) - noise
    assert ar
    for _ in range(5):
        run(sess, prog(ALL_REDUCE))
        after, _ = run(sess, prog())
        assert not after & ar


def test_cxx_signal_sees_what_python_coverage_cannot(sess):
    """The same Python path with a much larger tensor: no new Python lines, but Gloo does different work."""
    _, py_seen = run(sess, prog(ALL_REDUCE))
    small, py_small = run(sess, prog(ALL_REDUCE))
    large, py_large = run(sess, prog(ALL_REDUCE, shape=(1 << 16,)))
    assert not py_large - py_seen - py_small
    assert large - small


def test_only_distributed_code_is_instrumented():
    """Plain tensor math must not touch any instrumented edge, or the allowlist leaked into ATen."""
    cov = cxxcov.CxxCoverage()
    a = torch.randn(64, 64)
    (a @ a).sum().item()
    torch.sort(torch.randn(1000))
    assert cov.take() == set()
