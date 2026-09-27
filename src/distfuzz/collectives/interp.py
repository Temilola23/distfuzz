from __future__ import annotations

import sys
from datetime import timedelta

import torch
import torch.distributed as dist

from . import tensors as T
from .desc import CALLS
from .prog import effective_args
from .semantics import a2a_shapes, list_len, resolve_root, salt

RO = {n: getattr(dist.ReduceOp, n) for n in ["SUM", "PRODUCT", "MIN", "MAX", "BAND", "BOR", "BXOR", "AVG"]}


class FuzzerUndefinedVar(Exception):
    pass


class Coverage:
    def __init__(self, pattern="torch/distributed/"):
        self.pattern = pattern
        self.new = set()
        self.fcache = {}
        mon = sys.monitoring
        self.mon = mon
        self.tool = mon.COVERAGE_ID
        mon.use_tool_id(self.tool, "distfuzz")
        E = mon.events
        mon.register_callback(self.tool, E.LINE, self._line)
        mon.register_callback(self.tool, E.BRANCH, self._branch)
        mon.set_events(self.tool, E.LINE | E.BRANCH)

    def _fid(self, code):
        fn = code.co_filename
        v = self.fcache.get(fn)
        if v is None:
            i = fn.find(self.pattern)
            v = fn[i:] if i >= 0 else False
            self.fcache[fn] = v
        return v

    def _line(self, code, line):
        f = self._fid(code)
        if f:
            self.new.add(f"{f}:{line}")
        return self.mon.DISABLE  # each line location fires once per process

    def _branch(self, code, src, dst):
        f = self._fid(code)
        if not f:
            return self.mon.DISABLE
        self.new.add(f"{f}:{code.co_firstlineno}:b{src}>{dst}")

    def take(self):
        s, self.new = self.new, set()
        return s


class RankEnv:
    def __init__(self, rank, world, pg, timeout):
        self.rank, self.world, self.pg, self.timeout = rank, world, pg, timeout
        self.t = {}
        self.lists = {}
        self.groups = {}
        self.works = {}
        self.pending = []
        self.all_guarded = []
        self.created_groups = []

    def G(self, g):
        if g == "world":
            return self.pg, list(range(self.world))
        if g not in self.groups:
            raise FuzzerUndefinedVar(g)
        return self.groups[g]

    def tensor(self, v):
        if v not in self.t:
            raise FuzzerUndefinedVar(v)
        return self.t[v].view

    def mk(self, spec, s):
        g = T.make(spec, self.rank, s)
        self.all_guarded.append(g)
        return g


def _spec_like(t):
    return {"dtype": T.NAME[t.dtype], "shape": list(t.shape), "layout": "contig", "seed": 0, "kind": "int"}


def exec_call(i, c, env: RankEnv):
    a = effective_args(c, env.rank)
    if a is None:
        return
    op, rets, rank = c["op"], c.get("rets", {}), env.rank
    to = timedelta(seconds=env.timeout)
    work = None
    if op == "tensor":
        env.t[rets["out"]] = env.mk(a["spec"], salt(i, "out"))
        return
    if CALLS[op].get("fault"):
        from .faults import exec_fault

        exec_fault(op, a, env)
        return
    if op == "local":
        t = env.tensor(a["t"])
        fn = a["fn"]
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
        return
    if op == "wait":
        w = env.works.get(a["w"], "undef")
        if w == "undef":
            raise FuzzerUndefinedVar(a["w"])
        if w is not None:
            w.wait()
        env.works[a["w"]] = None
        return
    if op == "new_group":
        g = dist.new_group(a["ranks"], timeout=to, use_local_synchronization=bool(a.get("local_sync")))
        mem = sorted(set(a["ranks"]))
        member = rank in mem
        env.groups[rets["g"]] = (g, mem if member else None)
        if member and g is not None and g != dist.GroupMember.NON_GROUP_MEMBER:
            env.created_groups.append(g)
        return
    if op == "new_subgroups":
        cur, subs = dist.new_subgroups(a["size"], timeout=to)
        env.created_groups.extend(subs)
        env.groups[rets["g"]] = (cur, sorted(dist.get_process_group_ranks(cur)))
        return
    if op == "p2p":
        t = env.tensor(a["t"])
        src, dst, tag, asy = a["src"], a["dst"], a["tag"], a["async_op"]
        if rank == src:
            if asy:
                env.pending.append(dist.isend(t, dst, group=env.pg, tag=tag))
            else:
                dist.send(t, dst, group=env.pg, tag=tag)
        if rank == dst:
            if asy:
                env.pending.append(dist.irecv(t, src, group=env.pg, tag=tag))
            else:
                dist.recv(t, src, group=env.pg, tag=tag)
        return
    if op == "ring":
        t = env.tensor(a["t"])
        s = a["shift"]
        out = env.mk(_spec_like(t), salt(i, "out"))
        env.t[rets["out"]] = out
        ops = [
            dist.P2POp(dist.isend, t, (rank + s) % env.world, group=env.pg),
            dist.P2POp(dist.irecv, out.view, (rank - s) % env.world, group=env.pg),
        ]
        for r in dist.batch_isend_irecv(ops):
            r.wait()
        return

    grp, mem = env.G(a["group"])
    n = len(mem) if mem else 0
    grank = mem.index(rank) if mem and rank in mem else None
    asy = bool(a.get("async_op"))
    kw = dict(group=grp)
    if op == "all_reduce":
        work = dist.all_reduce(env.tensor(a["t"]), op=RO[a["op"]], async_op=asy, **kw)
    elif op in ("broadcast", "reduce"):
        t = env.tensor(a["t"])
        rk = "src" if op == "broadcast" else "dst"
        if a.get("root_mode") == "group":
            kw["group_" + rk] = a["root"]
        else:
            kw[rk] = a["root"]
        if op == "broadcast":
            work = dist.broadcast(t, async_op=asy, **kw)
        else:
            work = dist.reduce(t, op=RO[a["op"]], async_op=asy, **kw)
    elif op == "all_gather":
        outs = [env.mk(a["out"], salt(i, "outs", k)) for k in range(list_len(a, n))]
        env.lists[rets["outs"]] = outs
        work = dist.all_gather([o.view for o in outs], env.tensor(a["t"]), async_op=asy, **kw)
    elif op == "all_gather_into_tensor":
        out = env.mk(a["out"], salt(i, "out"))
        env.t[rets["out"]] = out
        work = dist.all_gather_into_tensor(out.view, env.tensor(a["t"]), async_op=asy, **kw)
    elif op == "reduce_scatter":
        out = env.mk(a["out"], salt(i, "out"))
        env.t[rets["out"]] = out
        ins = [env.mk(a["ins"], salt(i, "ins", k)) for k in range(list_len(a, n))]
        work = dist.reduce_scatter(out.view, [x.view for x in ins], op=RO[a["op"]], async_op=asy, **kw)
    elif op == "reduce_scatter_tensor":
        out = env.mk(a["out"], salt(i, "out"))
        env.t[rets["out"]] = out
        work = dist.reduce_scatter_tensor(out.view, env.tensor(a["t"]), op=RO[a["op"]], async_op=asy, **kw)
    elif op == "all_to_all_single":
        gr = grank if grank is not None else 0
        ish, osh, isp, osp = a2a_shapes(a, max(n, 1), gr)
        spec = dict(a["inspec"])
        inp = env.mk(dict(spec, shape=ish), salt(i, "in"))
        out = env.mk(dict(spec, shape=osh, layout="contig"), salt(i, "out"))
        env.t[rets["in"]], env.t[rets["out"]] = inp, out
        work = dist.all_to_all_single(
            out.view, inp.view, output_split_sizes=osp, input_split_sizes=isp, async_op=asy, **kw
        )
    elif op == "all_to_all":
        outs = [env.mk(a["out"], salt(i, "outs", k)) for k in range(n)]
        ins = [env.mk(a["ins"], salt(i, "ins", k)) for k in range(n)]
        env.lists[rets["outs"]] = outs
        work = dist.all_to_all([o.view for o in outs], [x.view for x in ins], async_op=asy, **kw)
    elif op in ("scatter", "gather"):
        t = env.tensor(a["t"])
        root = resolve_root(a, mem, rank)
        is_root = (root == rank) if root is not None else (grank == a["root"])
        rk = "src" if op == "scatter" else "dst"
        if a.get("root_mode") == "group":
            kw["group_" + rk] = a["root"]
        else:
            kw[rk] = a["root"]
        if op == "scatter":
            lst = None
            if is_root or a.get("list_everywhere"):
                lst = [env.mk(a["ins"], salt(i, "ins", k)).view for k in range(list_len(a, n))]
            work = dist.scatter(t, lst, async_op=asy, **kw)
        else:
            lst = None
            if is_root or a.get("list_everywhere"):
                outs = [env.mk(a["out"], salt(i, "outs", k)) for k in range(list_len(a, n))]
                lst = [o.view for o in outs]
                if "outs" in rets:
                    env.lists[rets["outs"]] = outs
            work = dist.gather(t, lst, async_op=asy, **kw)
    elif op == "barrier":
        work = dist.barrier(async_op=asy, **kw)
    elif op == "monitored_barrier":
        dist.monitored_barrier(timeout=to, wait_all_ranks=bool(a.get("wait_all")), **kw)
    elif op == "all_gather_object":
        lst = [None] * n
        dist.all_gather_object(lst, {"r": rank, "s": a["seed"]}, **kw)
        env.lists[rets["outs"]] = lst
    elif op == "broadcast_object_list":
        lst = [{"r": rank, "s": a["seed"], "i": k} for k in range(a["k"])]
        if a.get("root_mode") == "group":
            kw["group_src"] = a["root"]
        else:
            kw["src"] = a["root"]
        dist.broadcast_object_list(lst, **kw)
        env.lists[rets["outs"]] = lst
    else:
        raise ValueError(op)
    if "w" in rets:
        env.works[rets["w"]] = work if asy else None
    elif asy and work is not None:
        env.pending.append(work)


def _short(e):
    msg = str(e).strip().splitlines()
    return (msg[0] if msg else "")[:400], "\n".join(msg[:12])[:3000]


def run_program(p, rank, pg, timeout):
    env = RankEnv(rank, p["world"], pg, timeout)
    status = {"exc": None, "wait_exc": []}
    for i, c in enumerate(p["calls"]):
        try:
            exec_call(i, c, env)
        except Exception as e:  # noqa: BLE001
            head, full = _short(e)
            status["exc"] = {"idx": i, "op": c["op"], "type": type(e).__name__, "msg": head, "full": full}
            break
    for w in list(env.works.values()) + env.pending:
        if w is None:
            continue
        try:
            w.wait()
        except Exception as e:  # noqa: BLE001
            status["wait_exc"].append({"type": type(e).__name__, "msg": _short(e)[0]})
    outputs = {v: g.view.detach().clone().contiguous() for v, g in env.t.items()}
    lists = {
        v: [x.view.detach().clone() if isinstance(x, T.Guarded) else x for x in lst] for v, lst in env.lists.items()
    }
    status.update(outputs=outputs, lists=lists, guard_bad=sum(not g.guards_ok() for g in env.all_guarded))
    return status, env
