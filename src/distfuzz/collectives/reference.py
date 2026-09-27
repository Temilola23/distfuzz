from __future__ import annotations

import math

import torch

from . import tensors as T
from .desc import CALLS
from .prog import effective_args
from .semantics import a2a_shapes, list_len, resolve_root, salt


class Stop(Exception):
    def __init__(self, status, why):
        super().__init__(why)
        self.status, self.why = status, why


def op_ok(op, dt):
    if dt == "bool" or dt in T.INT_DTYPES:
        return op != "AVG"
    if dt == "complex64":
        return op in ("SUM", "AVG")
    return op in ("SUM", "PRODUCT", "MIN", "MAX", "AVG")


def reduce_vals(op, vals):
    acc = vals[0].clone()
    for v in vals[1:]:
        if op in ("SUM", "AVG"):
            acc = acc | v if acc.dtype == torch.bool else acc + v
        elif op == "PRODUCT":
            acc = acc & v if acc.dtype == torch.bool else acc * v
        elif op == "MIN":
            acc = torch.minimum(acc, v)
        elif op == "MAX":
            acc = torch.maximum(acc, v)
        elif op == "BAND":
            acc = acc & v
        elif op == "BOR":
            acc = acc | v
        elif op == "BXOR":
            acc = acc ^ v
    if op == "AVG":
        acc = (acc / len(vals)).to(vals[0].dtype)
    return acc


class Val:
    __slots__ = ("v", "layout", "dtype", "shape")

    def __init__(self, v, layout, meta=None):
        self.v, self.layout = v, layout
        self.dtype, self.shape = (v.dtype, tuple(v.shape)) if v is not None else meta

    def numel(self):
        return math.prod(self.shape)


class Reference:
    def __init__(self, prog):
        self.p = prog
        self.W = prog["world"]
        self.t = [{} for _ in range(self.W)]
        self.lists = [{} for _ in range(self.W)]
        self.groups = [{} for _ in range(self.W)]
        self.works = [{} for _ in range(self.W)]
        self.inflight = [[] for _ in range(self.W)]
        self.racy = False
        self.status, self.at, self.why = "valid", None, ""

    def mk(self, spec, rank, s):
        return Val(T.expected_initial(spec, rank, s), spec.get("layout", "contig"))

    def get(self, r, var):
        if var not in self.t[r]:
            raise Stop("invalid", f"rank {r} uses undefined {var}")
        return self.t[r][var]

    def members(self, r, g):
        if g == "world":
            return list(range(self.W))
        if g not in self.groups[r]:
            raise Stop("invalid", f"rank {r} uses undefined group {g}")
        return self.groups[r][g]

    def touch(self, r, reads, writes, asy, wvar=None):
        for er, ew in self.inflight[r]:
            if (writes & (er | ew)) or (reads & ew):
                self.racy = True
        if asy:
            ent = (set(reads), set(writes))
            self.inflight[r].append(ent)
            if wvar:
                self.works[r][wvar] = ent

    def check_write(self, val):
        if val.layout == "expanded":
            raise Stop("uncertain", "write into stride-0 expanded tensor")

    def run(self):
        for i, c in enumerate(self.p["calls"]):
            try:
                self.step(i, c)
            except Stop as s:
                self.status, self.at, self.why = s.status, i, s.why
                break
        return self

    def participants(self, i, c):
        eff = [effective_args(c, r) for r in range(self.W)]
        parts = {}
        for r in range(self.W):
            if eff[r] is None:
                continue
            m = self.members(r, eff[r]["group"])
            if m is not None and r in m:
                parts[r] = (eff[r], m)
        groups = {}
        for r, (a, m) in parts.items():
            for x in m:
                if x not in parts:
                    raise Stop("invalid", f"rank {x} does not join {c['op']} of group {m}")
                if parts[x][1] != m:
                    raise Stop("invalid", f"ranks {r},{x} disagree on group membership")
                if parts[x][0]["group"] != a["group"]:
                    raise Stop(
                        "invalid", f"ranks {r},{x} use different process groups {a['group']},{parts[x][0]['group']}"
                    )
            groups[tuple(m)] = m
        return eff, parts, list(groups.values())

    def same(self, parts, m, key):
        vals = {repr(parts[x][0].get(key)) for x in m}
        if len(vals) > 1:
            raise Stop("invalid", f"ranks disagree on {key}")

    def same_meta(self, vals, what):
        dts = {str(v.dtype) for v in vals}
        shp = {tuple(v.shape) for v in vals}
        if len(dts) > 1:
            raise Stop("invalid", f"dtype mismatch across ranks in {what}")
        if len(shp) > 1:
            if len({v.numel() for v in vals}) == 1:
                raise Stop("uncertain", f"same numel, different shapes in {what}")
            raise Stop("invalid", f"shape mismatch across ranks in {what}")

    def step(self, i, c):
        op, rets, W = c["op"], c.get("rets", {}), self.W
        if CALLS[op].get("fault"):
            # no value semantics: only the crash, hang and guard oracles apply
            self.racy = True
            raise Stop("uncertain", f"fault op {op}")
        eff = [effective_args(c, r) for r in range(W)]
        if op == "tensor":
            for r in range(W):
                self.t[r][rets["out"]] = self.mk(eff[r]["spec"], r, salt(i, "out"))
            return
        if op == "local":
            for r in range(W):
                if eff[r] is None:
                    continue
                x = self.get(r, eff[r]["t"])
                self.check_write(x)
                self.touch(r, {eff[r]["t"]}, {eff[r]["t"]}, False)
                fn, v = eff[r]["fn"], x.v
                if v is None:
                    continue
                if v.dtype == torch.bool:
                    x.v = ~v if fn in ("add1", "neg") else (torch.zeros_like(v) if fn == "zero" else v)
                elif fn == "add1":
                    x.v = v + 1
                elif fn == "mul2":
                    x.v = v * 2
                elif fn == "zero":
                    x.v = torch.zeros_like(v)
                else:
                    x.v = -v if v.dtype != torch.uint8 else v + 1
                x.v = x.v.to(v.dtype)
            return
        if op == "wait":
            for r in range(W):
                if eff[r] is None:
                    continue
                w = eff[r]["w"]
                ent = self.works[r].pop(w, None)
                if ent is not None and ent in self.inflight[r]:
                    self.inflight[r].remove(ent)
            return
        if op in ("new_group", "new_subgroups"):
            if any(e is None for e in eff):
                raise Stop("invalid", f"{op} not called on every rank")
            if len({repr(e) for e in eff}) > 1:
                raise Stop("invalid", f"{op} args differ across ranks")
            a = eff[0]
            if op == "new_group":
                rk = a["ranks"]
                if a.get("local_sync"):
                    raise Stop("uncertain", "use_local_synchronization")
                if len(set(rk)) != len(rk) or any(not (0 <= x < W) for x in rk):
                    raise Stop("uncertain", "bad ranks list")
                mem = sorted(rk)
                for r in range(W):
                    self.groups[r][rets["g"]] = mem if r in mem else None
            else:
                s = a["size"]
                if s <= 0 or W % s:
                    raise Stop("uncertain", "subgroup size does not divide world")
                for r in range(W):
                    self.groups[r][rets["g"]] = list(range((r // s) * s, (r // s) * s + s))
            return
        if op == "p2p":
            sends, recvs = [], []
            for r in range(W):
                a = eff[r]
                if a is None:
                    continue
                if not (0 <= a["src"] < W and 0 <= a["dst"] < W) or a["src"] == a["dst"]:
                    if r in (a["src"], a["dst"]):
                        raise Stop("uncertain", "bad p2p peer")
                    continue
                if r == a["src"]:
                    sends.append((r, a["dst"], a["tag"], a["t"], a["async_op"]))
                if r == a["dst"]:
                    recvs.append((a["src"], r, a["tag"], a["t"], a["async_op"]))
            if sorted(x[:3] for x in sends) != sorted(x[:3] for x in recvs):
                raise Stop("invalid", "unmatched send/recv")
            for s, d, tag, tv, asy in sends:
                rv = [x for x in recvs if x[:3] == (s, d, tag)][0]
                src, dst = self.get(s, tv), self.get(d, rv[3])
                if src.layout != "contig" and src.layout != "offset" or dst.layout not in ("contig", "offset"):
                    raise Stop("uncertain", "non-contiguous p2p tensor")
                self.same_meta([src, dst], "p2p")
                self.touch(s, {tv}, set(), asy)
                self.touch(d, set(), {rv[3]}, asy)
                dst.v = None if src.v is None else src.v.clone()
            return
        if op == "ring":
            if any(e is None for e in eff):
                raise Stop("invalid", "ring skipped on some rank")
            if len({e["shift"] for e in eff}) > 1:
                raise Stop("invalid", "ring shift differs")
            s = eff[0]["shift"]
            if s % W == 0:
                raise Stop("uncertain", "self send in ring")
            vals = [self.get(r, eff[r]["t"]) for r in range(W)]
            if any(v.layout not in ("contig", "offset") for v in vals):
                raise Stop("uncertain", "non-contiguous p2p tensor")
            self.same_meta(vals, "ring")
            for r in range(W):
                self.touch(r, {eff[r]["t"]}, {rets["out"]}, False)  # batch_isend_irecv + wait: synchronous
            for r in range(W):
                src = vals[(r - s) % W]
                self.t[r][rets["out"]] = Val(None if src.v is None else src.v.clone(), "contig", (src.dtype, src.shape))
            return

        eff, parts, groups = self.participants(i, c)
        for m in groups:
            self.collective(i, c, op, rets, parts, m)
        for r in range(W):
            if eff[r] is not None and r not in parts:
                if op in ("all_gather_into_tensor", "reduce_scatter", "reduce_scatter_tensor"):
                    self.t[r][rets["out"]] = self.mk(eff[r]["out"], r, salt(i, "out"))
                if op == "all_to_all_single":  # interp materialises non-member in/out with n=1, grank=0
                    ish, osh, _, _ = a2a_shapes(eff[r], 1, 0)
                    spec = dict(eff[r]["inspec"])
                    self.t[r][rets["in"]] = self.mk(dict(spec, shape=ish), r, salt(i, "in"))
                    self.t[r][rets["out"]] = self.mk(dict(spec, shape=osh, layout="contig"), r, salt(i, "out"))
                if "w" in rets:
                    self.works[r][rets["w"]] = None

    def collective(self, i, c, op, rets, parts, m):
        n = len(m)
        A = {x: parts[x][0] for x in m}
        a0 = A[m[0]]
        asy = {x: bool(A[x].get("async_op")) for x in m}
        if "op" in a0:
            self.same(parts, m, "op")
        wv = rets.get("w")

        def root_of():
            roots = {x: resolve_root(A[x], m, x) for x in m}
            if any(v is None for v in roots.values()):
                raise Stop("uncertain", "root not in group")
            if len(set(roots.values())) > 1:
                raise Stop("invalid", "ranks disagree on root")
            return roots[m[0]]

        if op == "all_reduce":
            vals = [self.get(x, A[x]["t"]) for x in m]
            self.same_meta(vals, op)
            if not op_ok(a0["op"], T.NAME[vals[0].dtype]):
                raise Stop("uncertain", f"{a0['op']} on {vals[0].dtype}")
            for v in vals:
                self.check_write(v)
            res = None if any(v.v is None for v in vals) else reduce_vals(a0["op"], [v.v for v in vals])
            for x, v in zip(m, vals):
                self.touch(x, {A[x]["t"]}, {A[x]["t"]}, asy[x], wv)
                v.v = None if res is None else res.clone()
        elif op in ("broadcast", "reduce"):
            root = root_of()
            vals = [self.get(x, A[x]["t"]) for x in m]
            self.same_meta(vals, op)
            for v in vals:
                self.check_write(v)
            if op == "broadcast":
                src = vals[m.index(root)].v
                for x, v in zip(m, vals):
                    self.touch(x, {A[x]["t"]}, {A[x]["t"]}, asy[x], wv)
                    v.v = None if src is None else src.clone()
            else:
                if not op_ok(a0["op"], T.NAME[vals[0].dtype]):
                    raise Stop("uncertain", f"{a0['op']} on {vals[0].dtype}")
                res = None if any(v.v is None for v in vals) else reduce_vals(a0["op"], [v.v for v in vals])
                for x, v in zip(m, vals):
                    self.touch(x, {A[x]["t"]}, {A[x]["t"]}, asy[x], wv)
                    # non-root buffers are unspecified after reduce (may be scratch space)
                    v.v = (None if res is None else res.clone()) if x == root else None
        elif op == "all_gather":
            vals = [self.get(x, A[x]["t"]) for x in m]
            self.same_meta(vals, op)
            self.same(parts, m, "n_delta")
            if list_len(a0, n) != n:
                raise Stop("uncertain", "output list length != group size")
            for x in m:
                o = A[x]["out"]
                if (
                    o["dtype"] != T.NAME[vals[0].dtype]
                    or list(o["shape"]) != list(vals[0].shape)
                    or o["layout"] == "expanded"
                ):
                    raise Stop("uncertain", "all_gather output spec differs from input")
            for x in m:
                self.touch(x, {A[x]["t"]}, set(), asy[x], wv)
                self.lists[x][rets["outs"]] = [None if v.v is None else v.v.clone() for v in vals]
        elif op == "all_gather_into_tensor":
            vals = [self.get(x, A[x]["t"]) for x in m]
            self.same_meta(vals, op)
            for x in m:
                o = A[x]["out"]
                if o["dtype"] != T.NAME[vals[0].dtype] or math.prod(o["shape"]) != n * vals[0].numel():
                    raise Stop("uncertain", "all_gather_into_tensor size/dtype")
                if o["layout"] not in ("contig", "offset") or vals[0].layout not in ("contig", "offset"):
                    raise Stop("uncertain", "non-contiguous all_gather_into_tensor")
                ishape = list(vals[0].shape)
                concat = ishape and list(o["shape"]) == [n * ishape[0]] + ishape[1:]
                stack = list(o["shape"]) == [n] + ishape
                if not (concat or stack):
                    raise Stop("uncertain", "output is neither concat nor stack form")
            flat = None if any(v.v is None for v in vals) else torch.cat([v.v.reshape(-1) for v in vals])
            for x in m:
                self.touch(x, {A[x]["t"]}, {rets["out"]}, asy[x], wv)
                o = A[x]["out"]
                self.t[x][rets["out"]] = Val(
                    None if flat is None else flat.reshape(o["shape"]).clone(),
                    o["layout"],
                    (vals[0].dtype, tuple(o["shape"])),
                )
        elif op == "reduce_scatter":
            self.same(parts, m, "n_delta")
            if list_len(a0, n) != n:
                raise Stop("uncertain", "input list length != group size")
            outs = {x: A[x]["out"] for x in m}
            for x in m:
                if A[x]["ins"]["dtype"] != outs[x]["dtype"] or list(A[x]["ins"]["shape"]) != list(outs[x]["shape"]):
                    raise Stop("uncertain", "reduce_scatter in/out spec differ")
                if outs[x]["layout"] == "expanded":
                    raise Stop("uncertain", "expanded output")
            if len({(o["dtype"], tuple(o["shape"])) for o in outs.values()}) > 1:
                raise Stop("invalid", "reduce_scatter shapes differ across ranks")
            if not op_ok(a0["op"], outs[m[0]]["dtype"]):
                raise Stop("uncertain", "op/dtype")
            ins = {x: [T.expected_initial(A[x]["ins"], x, salt(i, "ins", k)) for k in range(n)] for x in m}
            for gi, x in enumerate(m):
                res = reduce_vals(a0["op"], [ins[y][gi] for y in m])
                self.touch(x, set(), {rets["out"]}, asy[x], wv)
                self.t[x][rets["out"]] = Val(res, outs[x]["layout"])
        elif op == "reduce_scatter_tensor":
            vals = [self.get(x, A[x]["t"]) for x in m]
            self.same_meta(vals, op)
            outs = {x: A[x]["out"] for x in m}
            if len({(o["dtype"], tuple(o["shape"])) for o in outs.values()}) > 1:
                raise Stop("invalid", "reduce_scatter_tensor out shapes differ")
            o0 = outs[m[0]]
            if o0["dtype"] != T.NAME[vals[0].dtype] or vals[0].numel() != n * math.prod(o0["shape"]):
                raise Stop("uncertain", "reduce_scatter_tensor size/dtype")
            if any(o["layout"] not in ("contig", "offset") for o in outs.values()) or vals[0].layout not in (
                "contig",
                "offset",
            ):
                raise Stop("uncertain", "non-contiguous reduce_scatter_tensor")
            if not op_ok(a0["op"], o0["dtype"]):
                raise Stop("uncertain", "op/dtype")
            res = None if any(v.v is None for v in vals) else reduce_vals(a0["op"], [v.v.reshape(n, -1) for v in vals])
            for gi, x in enumerate(m):
                self.touch(x, {A[x]["t"]}, {rets["out"]}, asy[x], wv)
                self.t[x][rets["out"]] = Val(
                    None if res is None else res[gi].reshape(o0["shape"]).clone(),
                    outs[x]["layout"],
                    (vals[0].dtype, tuple(o0["shape"])),
                )
        elif op == "all_to_all_single":
            self.same(parts, m, "even")
            if not a0.get("even"):
                self.same(parts, m, "matrix")
                M = a0["matrix"]
                if len(M) != n or any(len(row) != n for row in M):
                    raise Stop("uncertain", "split matrix size != group size")
            specs = {x: A[x]["inspec"] for x in m}
            if (
                len(
                    {
                        (s["dtype"], tuple(s["shape"][1:]), (s["shape"] or [1])[0] if a0.get("even") else 0)
                        for s in specs.values()
                    }
                )
                > 1
            ):
                raise Stop("invalid", "all_to_all_single specs differ")
            if not specs[m[0]]["shape"]:
                raise Stop("uncertain", "0-d all_to_all_single")
            if specs[m[0]]["layout"] not in ("contig", "offset"):
                raise Stop("uncertain", "non-contiguous all_to_all_single")
            ins, shapes = {}, {}
            for gi, x in enumerate(m):
                ish, osh, isp, osp = a2a_shapes(A[x], n, gi)
                ins[x] = T.expected_initial(dict(specs[x], shape=ish), x, salt(i, "in"))
                shapes[x] = (osh, isp)
                self.t[x][rets["in"]] = Val(ins[x].clone(), specs[x]["layout"])
            for gi, x in enumerate(m):
                chunks = []
                for y in m:
                    isp_y = shapes[y][1]
                    if isp_y is None:
                        k = ins[y].shape[0] // n
                        chunks.append(ins[y][gi * k : (gi + 1) * k])
                    else:
                        off = sum(isp_y[:gi])
                        chunks.append(ins[y][off : off + isp_y[gi]])
                self.touch(x, {rets["in"]}, {rets["out"]}, asy[x], wv)
                self.t[x][rets["out"]] = Val(torch.cat(chunks).reshape(shapes[x][0]), "contig")
        elif op == "all_to_all":
            raise Stop("uncertain", "all_to_all unsupported on gloo")
        elif op == "scatter":
            root = root_of()
            self.same(parts, m, "n_delta")
            if any(A[x].get("list_everywhere") for x in m):
                raise Stop("uncertain", "scatter_list on non-src")
            if list_len(a0, n) != n:
                raise Stop("uncertain", "scatter list length")
            vals = [self.get(x, A[x]["t"]) for x in m]
            self.same_meta(vals, op)
            ins = A[root]["ins"]
            if (
                ins["dtype"] != T.NAME[vals[0].dtype]
                or list(ins["shape"]) != list(vals[0].shape)
                or any(v.layout == "expanded" for v in vals)
            ):
                raise Stop("uncertain", "scatter list spec differs")
            lst = [T.expected_initial(ins, root, salt(i, "ins", k)) for k in range(n)]
            for gi, (x, v) in enumerate(zip(m, vals)):
                self.touch(x, set(), {A[x]["t"]}, asy[x], wv)
                v.v = lst[gi].clone()
        elif op == "gather":
            root = root_of()
            self.same(parts, m, "n_delta")
            if any(A[x].get("list_everywhere") for x in m):
                raise Stop("uncertain", "gather_list on non-dst")
            if list_len(a0, n) != n:
                raise Stop("uncertain", "gather list length")
            vals = [self.get(x, A[x]["t"]) for x in m]
            self.same_meta(vals, op)
            o = A[root]["out"]
            if (
                o["dtype"] != T.NAME[vals[0].dtype]
                or list(o["shape"]) != list(vals[0].shape)
                or o["layout"] == "expanded"
            ):
                raise Stop("uncertain", "gather list spec differs")
            if o["dtype"] == "complex64":  # gloo gather has no complex path (all_gather does): 'Invalid scalar type'
                raise Stop("uncertain", "complex gather unsupported on gloo")
            for x in m:
                self.touch(x, {A[x]["t"]}, set(), asy[x], wv)
            if "outs" in rets:
                self.lists[root][rets["outs"]] = [None if v.v is None else v.v.clone() for v in vals]
        elif op in ("barrier", "monitored_barrier"):
            for x in m:
                self.touch(x, set(), set(), asy[x], wv)
        elif op == "all_gather_object":
            objs = [{"r": x, "s": A[x]["seed"]} for x in m]
            for x in m:
                self.lists[x][rets["outs"]] = list(objs)
        elif op == "broadcast_object_list":
            root = root_of()
            self.same(parts, m, "k")
            lst = [{"r": root, "s": A[root]["seed"], "i": k} for k in range(a0["k"])]
            for x in m:
                self.lists[x][rets["outs"]] = list(lst)
        else:
            raise Stop("uncertain", f"no reference for {op}")


def run_reference(prog):
    return Reference(prog).run()
