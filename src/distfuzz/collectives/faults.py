from __future__ import annotations

import gc
import os
import signal
import time
from datetime import timedelta

import torch
import torch.distributed as dist

from .interp import RO, FuzzerUndefinedVar, RankEnv


def _work(env, var):
    w = env.works.get(var, "undef")
    if w == "undef":
        raise FuzzerUndefinedVar(var)
    return w


def _member_group(env, g):
    grp, mem = env.G(g)
    if grp is None or (mem is not None and env.rank not in mem):
        return None, mem
    return grp, mem


def exec_fault(op, a, env: RankEnv):
    rank = env.rank
    if op == "drop_work":
        _work(env, a["w"])
        # drop the only reference while the op may still be in flight
        env.works[a["w"]] = None
        if a.get("gc"):
            gc.collect()
    elif op in ("wait_twice", "wait_late"):
        w = _work(env, a["w"])
        if w is not None:
            w.wait()
            if op == "wait_twice":
                w.wait()
        env.works[a["w"]] = None
    elif op == "destroy_group":
        grp, _ = env.G(a["group"])
        if grp is not None and grp != dist.GroupMember.NON_GROUP_MEMBER:
            dist.destroy_process_group(grp)
            if grp in env.created_groups:
                env.created_groups.remove(grp)
        env.groups.pop(a["group"], None)
    elif op == "abort_group":
        grp, _ = env.G(a["group"])
        if grp is not None and grp != dist.GroupMember.NON_GROUP_MEMBER:
            try:
                be = grp._get_backend(torch.device("cpu"))
            except Exception:  # noqa: BLE001
                be = None
            for tgt, meth in ((grp, "abort"), (grp, "shutdown"), (be, "abort"), (be, "shutdown")):
                if tgt is not None and hasattr(tgt, meth):
                    getattr(tgt, meth)()
                    break
    elif op in ("async_resize", "async_free"):
        grp, _ = _member_group(env, a["group"])
        if grp is None:
            return
        t = env.tensor(a["t"])
        work = dist.all_reduce(t, op=RO[a["op"]], async_op=True, group=grp)
        # Reallocate or free the buffer while the collective is still in flight. A Python error is
        # an acceptable outcome; the oracle only cares about SIGSEGV/SIGABRT and hangs.
        try:
            how = "free" if op == "async_free" else a.get("how")
            if how == "free":
                t.set_(torch.empty(0, dtype=t.dtype))
            elif how == "grow":
                t.resize_(t.numel() * 4 + 8)
            elif how == "shrink":
                t.resize_(0)
            elif how == "zero":
                t.zero_()
            else:
                t.set_(torch.empty(t.numel() + 4, dtype=t.dtype))
        except Exception:  # noqa: BLE001
            pass
        try:
            work.wait()
        except Exception:  # noqa: BLE001
            pass
    elif op == "batch_mismatch":
        _batch_mismatch(env, env.tensor(a["t"]).contiguous(), a["kind"])
    elif op == "crash_rank":
        grp, mem = _member_group(env, a["group"])
        if grp is None:
            return
        victim, when = a["victim"], a.get("when")
        is_victim = rank == victim and (mem is None or victim in mem)
        if is_victim and when in ("before", "during_barrier"):
            os.kill(os.getpid(), signal.SIGKILL)
        if when == "during_barrier":
            dist.barrier(group=grp)
            return
        w = dist.all_reduce(torch.ones(8), async_op=True, group=grp)
        if is_victim:
            os.kill(os.getpid(), signal.SIGKILL)
        w.wait()
    elif op == "short_timeout_probe":
        grp, mem = _member_group(env, a["group"])
        if grp is None:
            return
        # one member arrives late: monitored_barrier must raise a timeout, not hang or crash
        timeout = min(env.timeout, 1.0)
        if rank == (mem[0] if mem else 0):
            time.sleep(timeout + 1.5)
        dist.monitored_barrier(group=grp, timeout=timedelta(seconds=timeout), wait_all_ranks=True)
    else:
        raise KeyError(op)


def _batch_mismatch(env, t, kind):
    rank, W, pg = env.rank, env.world, env.pg
    nxt, prv = (rank + 1) % W, (rank - 1) % W
    ops = []
    if kind == "send_norecv" and rank == 0:
        ops.append(dist.P2POp(dist.isend, t, 1, group=pg))
    elif kind == "recv_nosend" and rank == 0:
        ops.append(dist.P2POp(dist.irecv, t, 1, group=pg))
    elif kind == "selfloop":
        ops += [dist.P2POp(dist.isend, t, rank, group=pg), dist.P2POp(dist.irecv, t.clone(), rank, group=pg)]
    elif kind == "double_recv":
        ops += [
            dist.P2POp(dist.irecv, t, prv, group=pg),
            dist.P2POp(dist.irecv, t.clone(), prv, group=pg),
            dist.P2POp(dist.isend, t.clone(), nxt, group=pg),
        ]
    elif kind == "cross":
        ops += [dist.P2POp(dist.isend, t, nxt, group=pg), dist.P2POp(dist.irecv, t.clone(), nxt, group=pg)]
    if ops:
        for w in dist.batch_isend_irecv(ops):
            w.wait()
