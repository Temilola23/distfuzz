from __future__ import annotations

import copy

SUPPORTED = {
    "tensor",
    "local",
    "wait",
    "new_group",
    "new_subgroups",
    "all_reduce",
    "broadcast",
    "reduce",
    "all_gather",
    "all_gather_into_tensor",
    "reduce_scatter",
    "reduce_scatter_tensor",
    "all_to_all_single",
    "scatter",
    "gather",
    "barrier",
    "p2p",
}


def _df():
    from distfuzz.collectives import prog as P
    from distfuzz.collectives import reference as REF
    from distfuzz.collectives import semantics as S
    from distfuzz.collectives import tensors as T

    return P, REF, S, T


def normalize(rec):
    """Use the minimized program if the finding carries one (the collectives minimizer writes prog_min)."""
    rec = copy.deepcopy(rec)
    if rec.get("prog_min"):
        rec["prog"] = rec["prog_min"]
    return rec


def lit(t):
    dt = str(t.dtype)
    vals = t.flatten().tolist()
    if t.dtype.is_complex:
        return f"torch.tensor({[complex(v) for v in vals]!r}, dtype={dt}).reshape({tuple(t.shape)!r})"
    return f"torch.tensor({vals!r}, dtype={dt}).reshape({tuple(t.shape)!r})"


class Gen:
    def __init__(self, rec):
        self.P, self.REF, self.S, self.T = _df()
        self.rec = rec
        self.p = rec["prog"]
        self.W = self.p["world"]
        self.L = []  # body lines (inside `def program(rank)`)
        self.members = {"world": [list(range(self.W))] * self.W}  # var -> per-rank members|None
        self.lastop = {}  # var -> last op touching it

    def emit(self, s, ind=1):
        self.L.append("    " * ind + s)

    def per_rank(self, fn):
        """Emit fn(rank)->list[str] once if identical on all ranks, else as an if/elif chain."""
        outs = [fn(r) for r in range(self.W)]
        if all(o == outs[0] for o in outs):
            for s in outs[0]:
                self.emit(s)
            return
        for r, o in enumerate(outs):
            self.emit(("if" if r == 0 else "elif") + f" rank == {r}:")
            for s in o or ["pass"]:
                self.emit(s, 2)

    def mk_expr(self, spec, rank, salt):
        v = self.T.fill_values(spec, rank, salt)
        return f"mk({lit(v)}, {spec.get('layout', 'contig')!r})"

    def call(self, i, c):
        P, S = self.P, self.S
        op, rets = c["op"], c.get("rets", {})
        if op not in SUPPORTED:
            raise NotImplementedError(f"codegen for {op}")
        eff = [P.effective_args(c, r) for r in range(self.W)]
        self.emit(f"# call {i}: {op}" + (f"  (per-rank divergence: {sorted(c['div'])})" if c.get("div") else ""))
        for v in rets.values():
            self.lastop[v] = op
        if isinstance(c["args"].get("t"), str):
            self.lastop[c["args"]["t"]] = op

        def grp(a):
            g = a["group"]
            return "WPG" if g == "world" else f"G[{g!r}]"

        def body(r):
            a = eff[r]
            if a is None:
                return ["pass  # this rank skips the call"]
            asy = bool(a.get("async_op"))
            ret_w = f"W[{rets['w']!r}] = " if "w" in rets else ("PENDING.append(" if asy else "")
            close = ")" if (asy and "w" not in rets) else ""
            if op == "tensor":
                return [f"T[{rets['out']!r}] = {self.mk_expr(a['spec'], r, S.salt(i, 'out'))}"]
            if op == "local":
                return [f"local_op(T[{a['t']!r}].view, {a['fn']!r})"]
            if op == "wait":
                return [f"if W.get({a['w']!r}) is not None: W[{a['w']!r}].wait()", f"W[{a['w']!r}] = None"]
            if op == "new_group":
                return [
                    f"G[{rets['g']!r}] = dist.new_group({a['ranks']!r}, timeout=TO, "
                    f"use_local_synchronization={bool(a.get('local_sync'))})"
                ]
            if op == "new_subgroups":
                return [f"G[{rets['g']!r}], _ = dist.new_subgroups({a['size']}, timeout=TO)"]
            if op == "p2p":
                out = []
                if r == a["src"]:
                    fn = "dist.isend" if asy else "dist.send"
                    out.append(
                        f"{'PENDING.append(' if asy else ''}{fn}(T[{a['t']!r}].view, {a['dst']}, tag={a['tag']}){')' if asy else ''}"
                    )
                if r == a["dst"]:
                    fn = "dist.irecv" if asy else "dist.recv"
                    out.append(
                        f"{'PENDING.append(' if asy else ''}{fn}(T[{a['t']!r}].view, {a['src']}, tag={a['tag']}){')' if asy else ''}"
                    )
                return out or ["pass  # not a peer"]
            mem = self.members_of(a["group"], r)
            n = len(mem) if mem else 0
            kw = f"group={grp(a)}, async_op={asy}"
            if op == "all_reduce":
                return [f"{ret_w}dist.all_reduce(T[{a['t']!r}].view, op=RO.{a['op']}, {kw}){close}"]
            if op in ("broadcast", "reduce"):
                rk = "src" if op == "broadcast" else "dst"
                rk = ("group_" + rk) if a.get("root_mode") == "group" else rk
                extra = f", op=RO.{a['op']}" if op == "reduce" else ""
                return [f"{ret_w}dist.{op}(T[{a['t']!r}].view{extra}, {rk}={a['root']}, {kw}){close}"]
            if op == "all_gather":
                k = S.list_len(a, n)
                outs = ", ".join(self.mk_expr(a["out"], r, S.salt(i, "outs", j)) for j in range(k))
                return [
                    f"L[{rets['outs']!r}] = [{outs}]",
                    f"{ret_w}dist.all_gather([o.view for o in L[{rets['outs']!r}]], T[{a['t']!r}].view, {kw}){close}",
                ]
            if op == "all_gather_into_tensor":
                return [
                    f"T[{rets['out']!r}] = {self.mk_expr(a['out'], r, S.salt(i, 'out'))}",
                    f"{ret_w}dist.all_gather_into_tensor(T[{rets['out']!r}].view, T[{a['t']!r}].view, {kw}){close}",
                ]
            if op == "reduce_scatter":
                k = S.list_len(a, n)
                ins = ", ".join(self.mk_expr(a["ins"], r, S.salt(i, "ins", j)) + ".view" for j in range(k))
                return [
                    f"T[{rets['out']!r}] = {self.mk_expr(a['out'], r, S.salt(i, 'out'))}",
                    f"{ret_w}dist.reduce_scatter(T[{rets['out']!r}].view, [{ins}], op=RO.{a['op']}, {kw}){close}",
                ]
            if op == "reduce_scatter_tensor":
                return [
                    f"T[{rets['out']!r}] = {self.mk_expr(a['out'], r, S.salt(i, 'out'))}",
                    f"{ret_w}dist.reduce_scatter_tensor(T[{rets['out']!r}].view, T[{a['t']!r}].view, op=RO.{a['op']}, {kw}){close}",
                ]
            if op == "all_to_all_single":
                gr = mem.index(r) if mem and r in mem else 0
                ish, osh, isp, osp = S.a2a_shapes(a, max(n, 1), gr)
                spec = dict(a["inspec"])
                return [
                    f"T[{rets['in']!r}] = {self.mk_expr(dict(spec, shape=ish), r, S.salt(i, 'in'))}",
                    f"T[{rets['out']!r}] = {self.mk_expr(dict(spec, shape=osh, layout='contig'), r, S.salt(i, 'out'))}",
                    f"{ret_w}dist.all_to_all_single(T[{rets['out']!r}].view, T[{rets['in']!r}].view, "
                    f"output_split_sizes={osp!r}, input_split_sizes={isp!r}, {kw}){close}",
                ]
            if op in ("scatter", "gather"):
                root = S.resolve_root(a, mem, r)
                gr = mem.index(r) if mem and r in mem else None
                is_root = (root == r) if root is not None else (gr == a["root"])
                rk = "src" if op == "scatter" else "dst"
                rk = ("group_" + rk) if a.get("root_mode") == "group" else rk
                k = S.list_len(a, n)
                if is_root or a.get("list_everywhere"):
                    key = "ins" if op == "scatter" else "out"
                    role = "ins" if op == "scatter" else "outs"
                    lst = "[" + ", ".join(self.mk_expr(a[key], r, S.salt(i, role, j)) for j in range(k)) + "]"
                else:
                    lst = "None"
                out = [f"_lst = {lst}"]
                if op == "gather" and "outs" in rets:
                    out.append(f"L[{rets['outs']!r}] = _lst if _lst is not None else []")
                out.append(
                    f"{ret_w}dist.{op}(T[{a['t']!r}].view, None if _lst is None else [x.view for x in _lst], "
                    f"{rk}={a['root']}, {kw}){close}"
                )
                return out
            if op == "barrier":
                return [f"{ret_w}dist.barrier({kw}){close}"]
            raise NotImplementedError(op)

        self.per_rank(body)
        self.emit(f"STEP[0] = {i + 1}")
        # track group membership for later calls (same rules as the reference model)
        if op == "new_group":
            mem = sorted(set(eff[0]["ranks"]))
            self.members[rets["g"]] = [mem if r in mem else None for r in range(self.W)]
        elif op == "new_subgroups":
            s = eff[0]["size"]
            self.members[rets["g"]] = [list(range((r // s) * s, (r // s) * s + s)) for r in range(self.W)]

    def members_of(self, g, r):
        return self.members.get(g, [None] * self.W)[r]

    def expected(self):
        ref = self.REF.run_reference(self.p)
        if ref.status != "valid" or ref.racy:
            return None, f"reference status={ref.status} racy={ref.racy} ({ref.why}); values not checked"
        exp_t = {}
        for r in range(self.W):
            for var, val in ref.t[r].items():
                if val.v is not None:
                    exp_t.setdefault(var, {})[r] = val.v
        exp_l = {}
        for r in range(self.W):
            for var, lst in ref.lists[r].items():
                if all(hasattr(x, "shape") for x in lst):
                    exp_l.setdefault(var, {})[r] = lst
        return (exp_t, exp_l), "reference status=valid"

    def generate(self, name="repro"):
        for i, c in enumerate(self.p["calls"]):
            self.call(i, c)
        exp, why = self.expected()
        if exp:
            et, el = exp
            exp_lines = ["EXPECTED = {"]
            for var, per in et.items():
                exp_lines.append(f"    {var!r}: {{" + ", ".join(f"{r}: {lit(v)}" for r, v in per.items()) + "},")
            exp_lines.append("}")
            exp_lines.append("EXPECTED_LISTS = {")
            for var, per in el.items():
                exp_lines.append(
                    f"    {var!r}: {{"
                    + ", ".join(f"{r}: [" + ", ".join(lit(x) for x in lst) + "]" for r, lst in per.items())
                    + "},"
                )
            exp_lines.append("}")
        else:
            exp_lines = ["EXPECTED, EXPECTED_LISTS = {}, {}  # " + why]
        return TEMPLATE.format(
            sig=self.rec.get("sig", "?"),
            world=self.W,
            name=name,
            why=why,
            body="\n".join(self.L),
            expected="\n".join(exp_lines),
            lastop=repr(self.lastop),
            ncalls=len(self.p["calls"]),
        )


TEMPLATE = '''#!/usr/bin/env python3
"""distfuzz.repro standalone repro: generated from a distfuzz.collectives program (plain PyTorch, no fuzzer imports).

Fuzzer signature: {sig}
Run:  python {name}.py        # spawns {world} CPU/Gloo ranks
Oracle: {why}.
Every buffer handed to torch.distributed is a view into a larger buffer whose head/tail hold a
sentinel; a changed sentinel means the backend wrote outside the tensor it was given.
"""
import datetime
import faulthandler
import os
import socket
import sys
import time
import traceback

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

WORLD = {world}
GUARD = 32
TO = datetime.timedelta(seconds=20)
RO = dist.ReduceOp
LASTOP = {lastop}


def sentinel(dt):
    if dt == torch.bool:
        return True
    if dt == torch.uint8:
        return 0xA5
    if dt == torch.int8:
        return -91
    if dt.is_complex:
        return complex(-12345.0, 4321.0)
    if dt.is_floating_point:
        return -12345.0 if dt != torch.float16 else -1234.0
    return -123456789 if dt != torch.int32 else -1234567


class Buf:
    """A view with the given layout inside a sentinel-guarded buffer."""

    def __init__(self, base, view, lo, hi):
        self.base, self.view, self.lo, self.hi = base, view, lo, hi

    def guards_ok(self):
        s = sentinel(self.base.dtype)
        return bool((self.base[:self.lo] == s).all()) and bool((self.base[self.hi:] == s).all())


def mk(vals, layout):
    """Materialise `vals` as: contig | noncontig (transposed / stride-2) | offset | expanded (stride 0)."""
    dt, shape = vals.dtype, list(vals.shape)
    n = vals.numel()
    if layout == "noncontig" and len(shape) >= 2:
        base = torch.full((n + 2 * GUARD,), sentinel(dt), dtype=dt)
        view = base[GUARD:GUARD + n].view(shape[:-2] + [shape[-1], shape[-2]]).transpose(-1, -2)
        view.copy_(vals)
        return Buf(base, view, GUARD, GUARD + n)
    if layout == "noncontig":
        sn = max(2 * n, 1)
        base = torch.full((sn + 2 * GUARD,), sentinel(dt), dtype=dt)
        inner = base[GUARD:GUARD + sn]
        view = inner[::2][:n].view(shape) if shape else inner[0:1].view(())
        view.copy_(vals)
        return Buf(base, view, GUARD, GUARD + sn)
    if layout == "expanded" and n > 0 and shape:
        rs = [1] + shape[1:]
        rn = int(torch.Size(rs).numel())
        base = torch.full((rn + 2 * GUARD,), sentinel(dt), dtype=dt)
        row = base[GUARD:GUARD + rn].view(rs)
        row.copy_(vals[:1])
        return Buf(base, row.expand(shape), GUARD, GUARD + rn)
    off = 3 if layout == "offset" else 0
    base = torch.full((n + off + 2 * GUARD,), sentinel(dt), dtype=dt)
    view = base[GUARD + off:GUARD + off + n].view(shape)
    view.copy_(vals)
    return Buf(base, view, GUARD, GUARD + off + n)


def local_op(t, fn):
    if t.dtype == torch.bool:
        t.logical_not_() if fn in ("add1", "neg") else (t.zero_() if fn == "zero" else None)
    elif fn == "add1":
        t.add_(1)
    elif fn == "mul2":
        t.mul_(2)
    elif fn == "zero":
        t.zero_()
    else:
        t.neg_() if t.dtype != torch.uint8 else t.add_(1)


{expected}

T, L, G, W, PENDING, STEP = {{}}, {{}}, {{}}, {{}}, [], [0]
WORLD_PG = None


def program(rank):
    WPG = WORLD_PG
{body}


def same(a, b):
    if a.dtype != b.dtype or tuple(a.shape) != tuple(b.shape):
        return False
    if a.numel() == 0:
        return True
    if a.is_floating_point() or a.is_complex():
        tol = 2e-2 if a.dtype in (torch.float16, torch.bfloat16) else 1e-4
        wide = torch.complex128 if a.is_complex() else torch.float64
        return bool(torch.allclose(a.to(wide), b.to(wide), rtol=tol, atol=tol, equal_nan=True))
    return bool(torch.equal(a, b))


def main(rank, port):
    global WORLD_PG
    faulthandler.dump_traceback_later(60, exit=True)   # hang watchdog
    dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{{port}}", rank=rank, world_size=WORLD, timeout=TO)
    WORLD_PG = dist.group.WORLD
    try:
        program(rank)
        for w in list(W.values()) + PENDING:
            if w is not None:
                w.wait()
    except Exception as e:
        print(f"[rank {{rank}}] raised {{type(e).__name__}}: {{(str(e).splitlines() or [''])[0][:300]}} (call {{STEP[0]}})", flush=True)
        traceback.print_exc()
        sys.stdout.flush(); sys.stderr.flush()
        time.sleep(3)  # let the other ranks report before mp.spawn tears the job down
        os._exit(1)
    bad = 0
    for var, per in EXPECTED.items():
        if rank in per and var in T:
            got = T[var].view.detach().clone().contiguous()
            if not same(per[rank], got):
                bad += 1
                print(f"{{LASTOP.get(var, '?')}}: MISMATCH rank {{rank}} var {{var}}\\n"
                      f"   expected {{tuple(per[rank].shape)}} {{per[rank].dtype}}: {{per[rank].flatten()[:12].tolist()}}\\n"
                      f"   actual   {{tuple(got.shape)}} {{got.dtype}}: {{got.flatten()[:12].tolist()}}", flush=True)
    for var, per in EXPECTED_LISTS.items():
        if rank in per and var in L:
            got = [x.view.detach().clone() for x in L[var]]
            if len(got) != len(per[rank]) or not all(same(e, g) for e, g in zip(per[rank], got)):
                print(f"{{LASTOP.get(var, '?')}}: MISMATCH rank {{rank}} list {{var}}", flush=True)
    for var, b in list(T.items()) + [(k, x) for k, lst in L.items() for x in lst]:
        if not b.guards_ok():
            print(f"[rank {{rank}}] OUT-OF-BOUNDS WRITE in {{LASTOP.get(var, '?')}} (sentinel around {{var}} changed)", flush=True)
    dist.barrier()
    if rank == 0:
        print(f"[distfuzz-repro] DONE torch {{torch.__version__}}", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    s = socket.socket(); s.bind(("127.0.0.1", 0)); port = s.getsockname()[1]; s.close()
    mp.spawn(main, args=(port,), nprocs=WORLD, join=True)
'''


def generate(rec, name="repro"):
    return Gen(rec).generate(name)


def units(rec):
    return list(range(len(rec["prog"]["calls"])))


def with_units(rec, keep):
    P = _df()[0]
    r = copy.deepcopy(rec)
    r["prog"]["calls"] = [c for i, c in enumerate(rec["prog"]["calls"]) if i in keep]
    P.sanitize(r["prog"])
    return r


def is_valid(rec):
    return bool(rec["prog"]["calls"])
