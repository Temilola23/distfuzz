"""C++ edge coverage from a PyTorch build instrumented with SanitizerCoverage (docker/Dockerfile.cov).

The instrumented Gloo and c10d code bumps a per-edge hit counter in libdfcov (docker/cov/dfcov.c). A feature
is an (edge, hit-count bucket) pair, as in AFL, so running a loop 3 times and 40 times are different features.
"""

from __future__ import annotations

import ctypes

import numpy as np

LIB = "libdfcov.so"

# AFL's buckets: 1, 2, 3, 4-7, 8-15, 16-31, 32-127, 128-255
BUCKET = np.zeros(256, dtype=np.int64)
for lo, hi, b in ((1, 1, 1), (2, 2, 2), (3, 3, 3), (4, 7, 4), (8, 15, 5), (16, 31, 6), (32, 127, 7), (128, 255, 8)):
    BUCKET[lo : hi + 1] = b


def features(counts: np.ndarray) -> set[int]:
    """Features for a counter map whose index 0 is the overflow slot."""
    idx = np.flatnonzero(counts[1:]) + 1
    return set((idx * 16 + BUCKET[counts[idx]]).tolist())


class CxxCoverage:
    def __init__(self, lib: str = LIB):
        self.lib = ctypes.CDLL(lib)
        self.lib.dfcov_edges.restype = ctypes.c_uint32
        self.lib.dfcov_buffer.restype = ctypes.POINTER(ctypes.c_uint8)
        self.n = self.lib.dfcov_edges()
        if not self.n:
            raise RuntimeError(f"{lib} is loaded but no code is instrumented; is torch built with Dockerfile.cov?")
        self.counts = np.ctypeslib.as_array(self.lib.dfcov_buffer(), shape=(self.n + 1,))
        self.take()  # drop what process start-up and init_process_group executed

    def take(self) -> set[int]:
        s = features(self.counts)
        self.lib.dfcov_reset()
        return s


def available(lib: str = LIB) -> bool:
    try:
        return CxxCoverage(lib).n > 0
    except (OSError, RuntimeError):
        return False
