import ctypes
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from distfuzz.collectives import cxxcov
from distfuzz.collectives.executor import signals

SRC = Path(__file__).resolve().parents[2] / "docker" / "cov" / "dfcov.c"
CC = shutil.which("cc") or shutil.which("clang") or shutil.which("gcc")


HAMMER = r"""
#include <pthread.h>
#include <stdint.h>
void __sanitizer_cov_trace_pc_guard(uint32_t *guard);
static uint32_t *g; static uint32_t n; static volatile int stop;
static void *loop(void *arg) {
    (void)arg;
    for (uint64_t i = 0; !stop; i++) __sanitizer_cov_trace_pc_guard(&g[(i * 2654435761u) % n]);
    return 0;
}
static pthread_t t;
void hammer_start(uint32_t *guards, uint32_t count) { g = guards; n = count; stop = 0; pthread_create(&t, 0, loop, 0); }
void hammer_stop(void) { stop = 1; pthread_join(t, 0); }
"""


@pytest.fixture
def runtime(tmp_path):
    """A fresh copy of the real runtime: its counters are process-wide statics, so each test gets its own."""
    if CC is None:
        pytest.skip("no C compiler")
    so = tmp_path / ("libdfcov.dylib" if sys.platform == "darwin" else "libdfcov.so")
    (tmp_path / "hammer.c").write_text(HAMMER)
    subprocess.run([CC, "-O2", "-fPIC", "-shared", str(SRC), str(tmp_path / "hammer.c"), "-o", str(so)], check=True)
    lib = ctypes.CDLL(str(so))
    lib.dfcov_edges.restype = ctypes.c_uint32
    return so, lib


def guards(n):
    return (ctypes.c_uint32 * n)()


def init(lib, g):
    end = ctypes.cast(ctypes.addressof(g) + ctypes.sizeof(g), ctypes.POINTER(ctypes.c_uint32))
    lib.__sanitizer_cov_trace_pc_guard_init(g, end)


def hit(lib, g, i, times=1):
    p = ctypes.pointer(ctypes.c_uint32.from_buffer(g, i * 4))
    for _ in range(times):
        lib.__sanitizer_cov_trace_pc_guard(p)


def afl_bucket(n):
    """Written independently of cxxcov.BUCKET, from AFL's count_class_lookup8."""
    if n == 0:
        return 0
    if n <= 3:
        return n
    for b, hi in ((4, 7), (5, 15), (6, 31), (7, 127)):
        if n <= hi:
            return b
    return 8


def test_bucket_table_matches_afl():
    assert [int(cxxcov.BUCKET[n]) for n in range(256)] == [afl_bucket(n) for n in range(256)]


def test_guards_are_numbered_once_and_across_modules(runtime):
    _, lib = runtime
    a, b = guards(3), guards(2)
    init(lib, a)
    init(lib, a)  # a module initialised twice keeps its numbers
    init(lib, b)
    assert list(a) == [1, 2, 3] and list(b) == [4, 5]
    assert lib.dfcov_edges() == 5


def test_counts_saturate_and_reset_clears(runtime):
    so, lib = runtime
    g = guards(4)
    init(lib, g)
    hit(lib, g, 0)
    hit(lib, g, 2, times=300)
    cov = cxxcov.CxxCoverage(str(so))  # its constructor drops everything counted so far
    assert cov.take() == set()
    hit(lib, g, 0, times=5)
    hit(lib, g, 2, times=300)
    hit(lib, g, 3)  # the last edge: reset must clear it too
    counts = cov.counts.copy()
    assert counts.tolist() == [0, 5, 0, 255, 1]
    assert cov.take() == {1 * 16 + 4, 3 * 16 + 8, 4 * 16 + 1}
    assert cov.take() == set() and not cov.counts.any()


def test_same_edges_different_counts_are_different_features(runtime):
    so, lib = runtime
    g = guards(1)
    init(lib, g)
    cov = cxxcov.CxxCoverage(str(so))
    seen = []
    for times in (1, 2, 3, 5, 40):
        hit(lib, g, 0, times)
        seen.append(cov.take())
    assert len({frozenset(s) for s in seen}) == 5


def test_uninstrumented_process_is_an_error_not_empty_coverage(runtime):
    so, _ = runtime
    with pytest.raises(RuntimeError, match="no code is instrumented"):
        cxxcov.CxxCoverage(str(so))
    assert not cxxcov.available(str(so))


def test_missing_runtime_is_not_available():
    assert not cxxcov.available("libdfcov-does-not-exist.so")


def test_features_ignore_overflow_slot():
    assert cxxcov.features(np.array([9, 0, 1], dtype=np.uint8)) == {2 * 16 + 1}


@pytest.mark.parametrize(
    "coverage,expected",
    [(False, set()), (True, {"py"}), ("py", {"py"}), ("cxx", {"cxx"}), ("py+cxx", {"py", "cxx"})],
)
def test_signals(coverage, expected):
    assert signals(coverage) == expected


def test_unknown_signal_is_rejected():
    with pytest.raises(ValueError):
        signals("python")


def test_take_is_safe_while_another_thread_is_counting(runtime):
    """Regression: features() ran np.flatnonzero on the live map; Gloo's background threads changed it under
    numpy, which raised and killed the rank (false CRASH findings)."""
    so, lib = runtime
    g = guards(1 << 18)
    init(lib, g)
    cov = cxxcov.CxxCoverage(str(so))
    lib.hammer_start(g, ctypes.c_uint32(len(g)))
    try:
        for _ in range(300):
            cov.take()
    finally:
        lib.hammer_stop()
