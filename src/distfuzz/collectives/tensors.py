from __future__ import annotations

import math

import torch

DTYPES = {
    "float32": torch.float32,
    "float64": torch.float64,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "int8": torch.int8,
    "uint8": torch.uint8,
    "int32": torch.int32,
    "int64": torch.int64,
    "bool": torch.bool,
    "complex64": torch.complex64,
}
NAME = {v: k for k, v in DTYPES.items()}
FLOAT_DTYPES = {"float32", "float64", "float16", "bfloat16"}
INT_DTYPES = {"int8", "uint8", "int32", "int64"}
LAYOUTS = ["contig", "noncontig", "offset", "expanded"]
GUARD = 32


def fill_values(spec, rank: int, salt: int = 0) -> torch.Tensor:
    dt = DTYPES[spec["dtype"]]
    n = math.prod(spec["shape"])
    g = torch.Generator().manual_seed((int(spec.get("seed", 0)) * 1000003 + rank * 7919 + salt) & 0x7FFFFFFF)
    if spec["dtype"] == "bool":
        v = torch.randint(0, 2, (n,), generator=g).to(torch.bool)
    elif spec["dtype"] == "uint8":
        v = torch.randint(0, 6, (n,), generator=g).to(dt)
    elif spec.get("kind", "int") == "randn" and spec["dtype"] in FLOAT_DTYPES | {"complex64"}:
        v = torch.randn(n, generator=g).to(dt)
    else:
        v = torch.randint(-4, 5, (n,), generator=g).to(dt)
    return v.reshape(spec["shape"])


def guard_value(dtype: torch.dtype):
    if dtype == torch.bool:
        return True
    if dtype == torch.uint8:
        return 0xA5
    if dtype == torch.int8:
        return -91
    if dtype.is_complex:
        return complex(-12345.0, 4321.0)
    if dtype.is_floating_point:
        return -12345.0 if dtype != torch.float16 else -1234.0
    return -123456789 if dtype != torch.int32 else -1234567


class Guarded:
    __slots__ = ("base", "view", "lo", "hi")

    def __init__(self, base, view, lo, hi):
        self.base, self.view, self.lo, self.hi = base, view, lo, hi

    def guards_ok(self) -> bool:
        gv = guard_value(self.base.dtype)
        return bool((self.base[: self.lo] == gv).all()) and bool((self.base[self.hi :] == gv).all())


def make(spec, rank: int, salt: int = 0) -> Guarded:
    dt = DTYPES[spec["dtype"]]
    shape = list(spec["shape"])
    layout = spec.get("layout", "contig")
    vals = fill_values(spec, rank, salt)
    n = math.prod(shape)
    if layout == "noncontig" and len(shape) >= 2:
        base = torch.full((n + 2 * GUARD,), guard_value(dt), dtype=dt)
        view = base[GUARD : GUARD + n].view(shape[:-2] + [shape[-1], shape[-2]]).transpose(-1, -2)
        view.copy_(vals)
        return Guarded(base, view, GUARD, GUARD + n)
    if layout == "noncontig":
        # gaps between strided elements keep the sentinel but only the outer guards are checked
        storage_n = max(2 * n, 1)
        base = torch.full((storage_n + 2 * GUARD,), guard_value(dt), dtype=dt)
        inner = base[GUARD : GUARD + storage_n]
        view = inner[::2][:n].view(shape) if len(shape) else inner[0:1].view(())
        view.copy_(vals)
        return Guarded(base, view, GUARD, GUARD + storage_n)
    if layout == "expanded" and n > 0 and len(shape) >= 1:
        row_shape = [1] + shape[1:]
        rn = math.prod(row_shape)
        base = torch.full((rn + 2 * GUARD,), guard_value(dt), dtype=dt)
        row = base[GUARD : GUARD + rn].view(row_shape)
        row.copy_(vals[:1])
        return Guarded(base, row.expand(shape), GUARD, GUARD + rn)
    off = 3 if layout == "offset" else 0
    base = torch.full((n + off + 2 * GUARD,), guard_value(dt), dtype=dt)
    view = base[GUARD + off : GUARD + off + n].view(shape)
    view.copy_(vals)
    return Guarded(base, view, GUARD, GUARD + off + n)


def expected_initial(spec, rank: int, salt: int = 0) -> torch.Tensor:
    return make(spec, rank, salt).view.clone()
