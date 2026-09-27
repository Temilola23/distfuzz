from __future__ import annotations

import copy
import json
import random

from .desc import CALLS, FAULT_OPS, FAULT_WEIGHTS, GENERATABLE
from .tensors import LAYOUTS

DTYPE_W = [
    ("float32", 30),
    ("float64", 8),
    ("float16", 6),
    ("bfloat16", 6),
    ("int32", 10),
    ("int64", 12),
    ("int8", 5),
    ("uint8", 5),
    ("bool", 3),
    ("complex64", 3),
]


def wchoice(rng, pairs):
    tot = sum(w for _, w in pairs)
    x = rng.uniform(0, tot)
    for v, w in pairs:
        x -= w
        if x <= 0:
            return v
    return pairs[-1][0]


def effective_args(call, rank):
    d = call.get("div", {}).get(str(rank))
    if d and d.get("__skip__"):
        return None
    a = dict(call["args"])
    if d:
        a.update({k: v for k, v in d.items() if k != "__skip__"})
    return a


def refs(call):
    out = set()
    desc = CALLS[call["op"]]
    for argsrc in [call["args"], *call.get("div", {}).values()]:
        for k, v in argsrc.items():
            t = desc["args"].get(k)
            if t in ("tensor", "work") and isinstance(v, str):
                out.add(v)
            elif t == "group" and isinstance(v, str) and v != "world":
                out.add(v)
    return out


def dumps(p):
    return json.dumps(p, sort_keys=True)


class State:
    def __init__(self, world):
        self.world = world
        self.tensors = {}  # var -> spec (base)
        self.groups = {}  # var -> per-rank member lists (None = not a member)
        self.works = []
        self.counter = 0

    def fresh(self, prefix):
        self.counter += 1
        return f"{prefix}{self.counter}"

    def members(self, g, rank):
        if g == "world":
            return list(range(self.world))
        return self.groups[g][rank]

    def any_members(self, g):
        for r in range(self.world):
            m = self.members(g, r)
            if m:
                return m
        return list(range(self.world))


def analyze(p, upto=None):
    st = State(p["world"])
    calls = p["calls"] if upto is None else p["calls"][:upto]
    mx = 0
    for c in p["calls"]:  # counter must exceed every var in the *whole* program
        for v in c.get("rets", {}).values():
            try:
                mx = max(mx, int(v[1:].split("_")[0]))
            except ValueError:
                pass
    for c in calls:
        _apply_state(st, c)
    st.counter = mx + 1
    return st


def _apply_state(st, c):
    op, a, rets = c["op"], c["args"], c.get("rets", {})
    W = st.world
    if op == "tensor":
        st.tensors[rets["out"]] = a["spec"]
    elif op == "new_group":
        mem = sorted({r for r in a["ranks"] if 0 <= r < W})
        st.groups[rets["g"]] = [mem if r in mem else None for r in range(W)]
    elif op == "new_subgroups":
        s = a["size"]
        if s > 0 and W % s == 0:
            st.groups[rets["g"]] = [list(range((r // s) * s, (r // s) * s + s)) for r in range(W)]
        else:
            st.groups[rets["g"]] = [list(range(W)) for _ in range(W)]
    elif op in ("all_gather_into_tensor", "reduce_scatter", "reduce_scatter_tensor"):
        if "out" in rets:
            st.tensors[rets["out"]] = a["out"]
    elif op == "all_to_all_single":
        st.tensors[rets["out"]] = a["inspec"]
        st.tensors[rets["in"]] = a["inspec"]
    elif op == "ring":
        st.tensors[rets["out"]] = st.tensors.get(a["t"], {"dtype": "float32", "shape": [1]})
    if "w" in rets:
        st.works.append(rets["w"])
    if op in ("wait", "drop_work", "wait_late") and a.get("w") in st.works:
        st.works.remove(a["w"])


class Generator:
    def __init__(self, world, rng: random.Random, fault=False):
        self.world = world
        self.rng = rng
        self.fault = fault

    def rand_spec(self, shape=None, dtype=None):
        r = self.rng
        if shape is None:
            nd = wchoice(r, [(0, 1), (1, 5), (2, 3), (3, 1)])
            shape = [wchoice(r, [(0, 1), (1, 4), (2, 6), (3, 5), (4, 4), (5, 2), (7, 1)]) for _ in range(nd)]
        return {
            "dtype": dtype or wchoice(r, DTYPE_W),
            "shape": list(shape),
            "layout": wchoice(r, [("contig", 70), ("noncontig", 14), ("offset", 12), ("expanded", 4)]),
            "seed": r.randrange(1000),
            "kind": wchoice(r, [("int", 80), ("randn", 20)]),
        }

    def value(self, typ, st: State, call_args=None):
        r, W = self.rng, self.world
        if isinstance(typ, tuple):
            if typ[0] == "enum":
                return r.choice(typ[1])
            if typ[0] == "int":
                return r.randint(typ[1], typ[2])
        if typ == "bool":
            return r.random() < 0.35
        if typ == "bool_rare":
            return r.random() < 0.08
        if typ == "redop":
            return wchoice(
                r,
                [
                    ("SUM", 40),
                    ("PRODUCT", 8),
                    ("MIN", 10),
                    ("MAX", 10),
                    ("BAND", 5),
                    ("BOR", 5),
                    ("BXOR", 5),
                    ("AVG", 6),
                ],
            )
        if typ == "root_mode":
            return wchoice(r, [("global", 70), ("group", 30)])
        if typ == "n_delta":
            return wchoice(r, [(0, 92), (-1, 4), (1, 4)])
        if typ == "ranks":
            k = r.randint(1, W)
            return sorted(r.sample(range(W), k))
        if typ == "subgroup_size":
            return wchoice(r, [(1, 1), (2, 5), (W, 2), (3, 1)])
        if typ == "rank":
            return r.randrange(W)
        if typ == "shift":
            return r.randint(1, W - 1)
        if typ == "tspec":
            return self.rand_spec()
        if typ == "resize_how":
            return r.choice(["grow", "shrink", "zero", "set_"])
        if typ == "batch_kind":
            return r.choice(["send_norecv", "recv_nosend", "selfloop", "cross", "double_recv"])
        if typ == "crash_when":
            return r.choice(["before", "during_async", "during_barrier"])
        raise KeyError(typ)

    def need_tensor(self, st, pre, spec=None, reuse=0.6):
        if spec is None and st.tensors and self.rng.random() < reuse:
            return self.rng.choice(sorted(st.tensors))
        spec = spec or self.rand_spec()
        v = st.fresh("t")
        c = {"op": "tensor", "args": {"spec": spec}, "rets": {"out": v}, "div": {}}
        pre.append(c)
        _apply_state(st, c)
        return v

    def need_group(self, st, pre):
        x = self.rng.random()
        if x < 0.6:
            return "world"
        if st.groups and x < 0.88:
            return self.rng.choice(sorted(st.groups))
        if self.rng.random() < 0.6:
            c = {
                "op": "new_group",
                "args": {"ranks": self.value("ranks", st), "local_sync": False},
                "rets": {"g": st.fresh("g")},
                "div": {},
            }
        else:
            c = {
                "op": "new_subgroups",
                "args": {"size": wchoice(self.rng, [(1, 1), (2, 5)])},
                "rets": {"g": st.fresh("g")},
                "div": {},
            }
        pre.append(c)
        _apply_state(st, c)
        return c["rets"]["g"]

    def gen_call(self, st: State, op=None):
        r = self.rng
        if op is None:
            pool = [(n, CALLS[n]["weight"]) for n in GENERATABLE]
            if self.fault:
                pool += [(n, FAULT_WEIGHTS[n]) for n in FAULT_OPS]
            op = wchoice(r, pool)
            if op == "wait" and not st.works:
                op = "all_reduce"
        d = CALLS[op]
        if d.get("fault"):
            return self.gen_fault_call(st, op)
        pre: list[dict] = []
        a = {}
        g = None
        if "group" in d["args"]:
            g = self.need_group(st, pre)
            a["group"] = g
        mem = st.any_members(g) if g else list(range(self.world))
        n = len(mem)
        for name, typ in d["args"].items():
            if name in a:
                continue
            if typ == "tensor":
                a[name] = None  # filled below
            elif typ == "work":
                a[name] = r.choice(st.works)
            elif typ == "root":
                a[name] = None
            elif typ == "split_matrix":
                a[name] = [[r.randint(0, 3) for _ in range(n)] for _ in range(n)]
            else:
                a[name] = self.value(typ, st)
        if "root" in a:
            if isinstance(g, str) and g != "world" and st.groups[g][0] != st.groups[g][-1]:
                a["root_mode"] = "group"  # subgroups: per-rank roots only make sense group-relative
            a["root"] = r.randrange(n) if a.get("root_mode") == "group" else r.choice(mem)
        if op in ("all_reduce", "broadcast", "reduce", "local"):
            a["t"] = self.need_tensor(st, pre)
        elif op == "all_gather":
            a["t"] = self.need_tensor(st, pre)
            base = st.tensors[a["t"]]
            a["out"] = dict(self.rand_spec(base["shape"], base["dtype"]))
        elif op == "all_gather_into_tensor":
            a["t"] = self.need_tensor(st, pre)
            base = st.tensors[a["t"]]
            numel = 1
            for s in base["shape"]:
                numel *= s
            shp = [n * numel] if r.random() < 0.6 else [n] + list(base["shape"])
            a["out"] = self.rand_spec(shp, base["dtype"])
        elif op in ("reduce_scatter", "all_to_all"):
            a["out"] = self.rand_spec()
            a["ins"] = dict(self.rand_spec(a["out"]["shape"], a["out"]["dtype"]))
        elif op == "reduce_scatter_tensor":
            k = r.randint(0, 4)
            rest: list[int] = r.choice([[], [2], [3]])
            dt = wchoice(r, DTYPE_W)
            a["t"] = self.need_tensor(st, pre, spec=self.rand_spec([n * k] + rest, dt))
            a["out"] = self.rand_spec([k] + rest, dt)
        elif op == "all_to_all_single":
            a["inspec"] = self.rand_spec([r.randint(0, 3)] + r.choice([[], [2]]))
        elif op in ("scatter", "gather"):
            a["t"] = self.need_tensor(st, pre)
            base = st.tensors[a["t"]]
            key = "ins" if op == "scatter" else "out"
            a[key] = dict(self.rand_spec(base["shape"], base["dtype"]))
        elif op == "p2p":
            a["t"] = self.need_tensor(st, pre)
            a["src"], a["dst"] = r.sample(range(self.world), 2)
        elif op == "ring":
            a["t"] = self.need_tensor(st, pre)
        rets = {}
        for role, kind in d["rets"].items():
            if role == "w" and not a.get("async_op"):
                continue
            rets[role] = st.fresh({"tensor": "t", "work": "w", "group": "g", "list": "l"}[kind])
        c = {"op": op, "args": a, "rets": rets, "div": {}}
        _apply_state(st, c)
        return pre + [c]

    def _need_async_work(self, st, pre):
        if st.works:
            return self.rng.choice(st.works)
        t = self.need_tensor(st, pre)
        c = {
            "op": "all_reduce",
            "args": {"t": t, "op": "SUM", "group": "world", "async_op": True},
            "rets": {"w": st.fresh("w")},
            "div": {},
        }
        pre.append(c)
        _apply_state(st, c)
        return c["rets"]["w"]

    def _need_real_group(self, st, pre):
        if st.groups and self.rng.random() < 0.7:
            return self.rng.choice(sorted(st.groups))
        c = {
            "op": "new_group",
            "args": {"ranks": self.value("ranks", st), "local_sync": False},
            "rets": {"g": st.fresh("g")},
            "div": {},
        }
        pre.append(c)
        _apply_state(st, c)
        return c["rets"]["g"]

    def gen_fault_call(self, st: State, op):
        r = self.rng
        pre: list[dict] = []
        a = {}
        if op in ("drop_work", "wait_twice", "wait_late"):
            a["w"] = self._need_async_work(st, pre)
            if op == "drop_work":
                a["gc"] = r.random() < 0.7
        elif op in ("destroy_group", "abort_group"):
            a["group"] = self._need_real_group(st, pre)
        elif op in ("async_resize", "async_free"):
            shape = [r.randint(1, 6)] + r.choice([[], [2], [3]])
            a["t"] = self.need_tensor(st, pre, spec=self.rand_spec(shape, r.choice(["float32", "int64", "int32"])))
            a["op"] = "SUM"
            a["group"] = "world" if r.random() < 0.7 else self._need_real_group(st, pre)
            if op == "async_resize":
                a["how"] = self.value("resize_how", st)
        elif op == "batch_mismatch":
            a["t"] = self.need_tensor(st, pre)
            a["kind"] = self.value("batch_kind", st)
        elif op == "crash_rank":
            a["group"] = "world" if r.random() < 0.6 else self._need_real_group(st, pre)
            a["victim"] = r.randrange(self.world)
            a["when"] = self.value("crash_when", st)
        elif op == "short_timeout_probe":
            a["group"] = "world" if r.random() < 0.6 else self._need_real_group(st, pre)
        c = {"op": op, "args": a, "rets": {}, "div": {}}
        _apply_state(st, c)
        return [*pre, c]

    def generate(self, ncalls=None):
        ncalls = ncalls or self.rng.randint(1, 8)
        p = {"world": self.world, "calls": []}
        st = State(self.world)
        while len(p["calls"]) < ncalls:
            p["calls"].extend(self.gen_call(st))
        return p


def sanitize(p):
    defined = set()
    out = []
    for c in p["calls"]:
        if refs(c) - defined:
            continue
        out.append(c)
        defined.update(c.get("rets", {}).values())
    p["calls"] = out
    return p


class Mutator:
    def __init__(self, world, rng: random.Random, max_calls=24, diverge_weight=6, fault=False):
        self.world = world
        self.rng = rng
        self.gen = Generator(world, rng, fault=fault)
        self.max_calls = max_calls
        self.diverge_weight = diverge_weight

    def mutate(self, p, corpus=()):
        p = copy.deepcopy(p)
        r = self.rng
        ops = [
            ("insert", 10),
            ("remove", 4),
            ("arg", 12),
            ("splice", 2 if corpus else 0),
            ("diverge", self.diverge_weight),
        ]
        for _ in range(64):
            kind = wchoice(r, ops)
            ok = getattr(self, "_" + kind)(p, corpus)
            if ok and p["calls"] and r.random() < 0.5:
                break
        sanitize(p)
        if not p["calls"]:
            p = self.gen.generate()
        p["calls"] = p["calls"][: self.max_calls]
        return sanitize(p)

    def _insert(self, p, corpus):
        if len(p["calls"]) >= self.max_calls:
            return False
        idx = len(p["calls"]) - int(abs(self.rng.gauss(0, 2))) if p["calls"] else 0
        idx = max(0, min(len(p["calls"]), idx))
        st = analyze(p, idx)
        new = self.gen.gen_call(st)
        p["calls"][idx:idx] = new
        return True

    def _remove(self, p, corpus):
        if not p["calls"]:
            return False
        del p["calls"][self.rng.randrange(len(p["calls"]))]
        sanitize(p)
        return True

    def _splice(self, p, corpus):
        if not corpus:
            return False
        other = copy.deepcopy(self.rng.choice(corpus))
        suffix = "_" + format(self.rng.randrange(1 << 20), "x")
        ren = {}
        for c in other["calls"]:
            for role, v in c.get("rets", {}).items():
                ren[v] = v + suffix
                c["rets"][role] = v + suffix
        for c in other["calls"]:
            for src in [c["args"]] + list(c.get("div", {}).values()):
                for k, v in list(src.items()):
                    if isinstance(v, str) and v in ren:
                        src[k] = ren[v]
        idx = self.rng.randrange(len(p["calls"]) + 1)
        p["calls"][idx:idx] = other["calls"]
        return True

    def mutate_value(self, typ, cur, st, call):
        r = self.rng
        if typ == "tspec":
            s = copy.deepcopy(cur)
            what = r.choice(["dtype", "shape", "layout", "seed", "kind", "dim"])
            if what == "dtype":
                s["dtype"] = wchoice(r, DTYPE_W)
            elif what == "shape" and s["shape"]:
                i = r.randrange(len(s["shape"]))
                s["shape"][i] = max(0, s["shape"][i] + r.choice([-1, 1, 2]))
            elif what == "dim":
                if s["shape"] and r.random() < 0.5:
                    s["shape"].pop()
                else:
                    s["shape"].append(r.randint(0, 4))
            elif what == "layout":
                s["layout"] = r.choice(LAYOUTS)
            elif what == "seed":
                s["seed"] = r.randrange(1000)
            else:
                s["kind"] = "randn" if s.get("kind") == "int" else "int"
            return s
        if typ == "tensor":
            return r.choice(sorted(st.tensors)) if st.tensors else cur
        if typ == "group":
            opts = ["world"] + sorted(st.groups)
            return r.choice(opts)
        if typ == "work":
            return r.choice(st.works) if st.works else cur
        if typ == "root":
            return r.randint(0, self.world)  # occasionally out of range
        if typ == "split_matrix":
            m = copy.deepcopy(cur)
            if m and r.random() < 0.8:
                i, j = r.randrange(len(m)), r.randrange(len(m))
                m[i][j] = r.randint(0, 3)
            else:
                n = r.randint(1, self.world)
                m = [[r.randint(0, 3) for _ in range(n)] for _ in range(n)]
            return m
        if typ == "ranks":
            if r.random() < 0.1:
                return [r.randrange(self.world + 1) for _ in range(r.randint(1, self.world))]
            return self.gen.value("ranks", st)
        if typ == "shift":
            return r.randint(0, self.world)
        if typ in ("bool", "bool_rare"):
            return not cur
        return self.gen.value(typ, st)

    def _pick_arg(self, p):
        cands = [i for i, c in enumerate(p["calls"]) if CALLS[c["op"]]["args"]]
        if not cands:
            return None
        i = self.rng.choice(cands)
        c = p["calls"][i]
        name = self.rng.choice(sorted(CALLS[c["op"]]["args"]))
        return i, c, name, CALLS[c["op"]]["args"][name]

    def _arg(self, p, corpus):
        x = self._pick_arg(p)
        if not x:
            return False
        i, c, name, typ = x
        st = analyze(p, i)
        c["args"][name] = self.mutate_value(typ, c["args"].get(name), st, c)
        if typ == "tensor" and not st.tensors:
            return False
        return True

    def _diverge(self, p, corpus):
        if not p["calls"]:
            return False
        i = self.rng.randrange(len(p["calls"]))
        c = p["calls"][i]
        rank = str(self.rng.randrange(self.world))
        if self.rng.random() < 0.2 and c["op"] != "tensor":
            c.setdefault("div", {})[rank] = {"__skip__": True}
            return True
        args = CALLS[c["op"]]["args"]
        if not args:
            return False
        name = self.rng.choice(sorted(args))
        st = analyze(p, i)
        cur = effective_args(c, int(rank)) or c["args"]
        c.setdefault("div", {}).setdefault(rank, {})[name] = self.mutate_value(args[name], cur.get(name), st, c)
        return True


def has_divergence(p):
    return any(c.get("div") for c in p["calls"])
