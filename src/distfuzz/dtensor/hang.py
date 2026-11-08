import os
import sys
import threading
import time
from collections import Counter

DIST = os.sep + os.path.join("torch", "distributed") + os.sep
TORCH = os.sep + "torch" + os.sep
COLL_FILES = ("distributed_c10d.py", "_functional_collectives.py")
COLL_FUNCS = frozenset(
    {
        "all_gather",
        "all_gather_into_tensor",
        "all_gather_object",
        "all_gather_tensor",
        "all_reduce",
        "all_reduce_coalesced",
        "all_to_all",
        "all_to_all_single",
        "barrier",
        "batch_isend_irecv",
        "broadcast",
        "broadcast_object_list",
        "gather",
        "monitored_barrier",
        "recv",
        "reduce",
        "reduce_scatter",
        "reduce_scatter_tensor",
        "scatter",
        "send",
        "wait_tensor",
    }
)


def _view_groups_branch(loc):
    # f and t are the current source and target group sizes; they stay fixed while view_groups backtracks forever.
    # f == 1 means the source side ran out (the ([1], []) branch, to_idx -= 1), t == 1 the target side (from_idx -= 1)
    f, t = loc.get("f"), loc.get("t")
    if f is None or t is None:
        return None
    if f == 1 and t != 1:
        return "src_exhausted"
    if t == 1 and f != 1:
        return "dst_exhausted"
    return None


# loops whose stuck state can be named from the frame's locals, so different branches get different signatures
KNOWN_LOOPS = {("_view_ops.py", "view_groups"): _view_groups_branch}


def inspect_stack(frame, harness_groups=()):
    frames = []
    while frame is not None:
        frames.append(frame)
        frame = frame.f_back
    stack = [f"{os.path.basename(f.f_code.co_filename)}:{f.f_code.co_name}:{f.f_lineno}" for f in frames[:8]]
    for f in frames:
        base = os.path.basename(f.f_code.co_filename)
        if base in COLL_FILES and f.f_code.co_name in COLL_FUNCS:
            group = f.f_locals.get("group")
            name = f.f_code.co_name + ("@harness" if any(group is g for g in harness_groups) else "")
            return dict(
                state="collective",
                coll=name,
                frame=f,
                file=base,
                func=f.f_code.co_name,
                line=f.f_lineno,
                label=None,
                stack=stack,
            )
    pick = next((f for f in frames if DIST in f.f_code.co_filename), None)
    pick = pick or next((f for f in frames if TORCH in f.f_code.co_filename), None)
    pick = pick or (frames[0] if frames else None)
    if pick is None:
        return None
    base = os.path.basename(pick.f_code.co_filename)
    labeler = KNOWN_LOOPS.get((base, pick.f_code.co_name))
    label = None
    if labeler is not None:
        try:
            label = labeler(pick.f_locals)
        except Exception:
            label = None
    return dict(
        state="code",
        coll=None,
        frame=pick,
        file=base,
        func=pick.f_code.co_name,
        line=pick.f_lineno,
        label=label,
        known=labeler is not None,
        in_dist=DIST in pick.f_code.co_filename,
        stack=stack,
    )


class Watchdog:
    """Samples one thread's stack while a program runs and reports, once, where it is stuck.

    cpu: the same torch.distributed frame has been running for `loop` seconds over several lines (or is a known loop)
    collective: the same collective call has been waiting for `coll` seconds, below Gloo's own 20 s timeout so the
        call has not yet turned into an error
    deadline: the program has run for `deadline` seconds, just before the parent's program timeout
    """

    def __init__(self, emit, thread_id=None, period=0.2, harness_groups=()):
        self.emit = emit
        self.tid = thread_id if thread_id is not None else threading.main_thread().ident
        self.period = period
        self.harness = tuple(harness_groups)
        self.lock = threading.Lock()
        self.job = None
        self.fired = False
        self.step = (-1, "")
        self.err = None
        self.track = None
        threading.Thread(target=self._run, daemon=True).start()

    def arm(self, wd):
        with self.lock:
            self.fired = False
            self.track = None
            self.step = (-1, "")
            self.err = None
            self.job = dict(wd, t0=time.monotonic()) if wd else None

    def disarm(self):
        with self.lock:
            self.job = None
            self.track = None
            return self.fired

    def _run(self):
        while True:
            time.sleep(self.period)
            with self.lock:
                if self.job is None or self.fired:
                    continue
                rep = self.sample(time.monotonic())
                if rep is not None:
                    self.fired = True
                    self.track = None
                    self.emit(rep)

    def sample(self, now):
        job = self.job
        frame = sys._current_frames().get(self.tid)
        if frame is None or job is None:
            return None
        info = inspect_stack(frame, self.harness)
        if info is None:
            return None
        tr = self.track
        if tr is None or tr["frame"] is not info["frame"]:
            tr = self.track = dict(frame=info["frame"], t0=now, lines=set(), labels=Counter())
        tr["lines"].add(info["line"])
        if info["label"]:
            tr["labels"][info["label"]] += 1
        held, elapsed = now - tr["t0"], now - job["t0"]
        kind = None
        if info["state"] == "collective" and held >= job["coll"]:
            kind = "collective"
        elif (
            info["state"] == "code"
            and info["in_dist"]
            and held >= job["loop"]
            and (info["known"] or len(tr["lines"]) >= 2)
        ):
            kind = "cpu"
        elif elapsed >= job["deadline"]:
            kind = "deadline"
        if kind is None:
            return None
        label = tr["labels"].most_common(1)[0][0] if tr["labels"] else None
        return dict(
            kind=kind,
            file=info["file"],
            func=info["func"],
            line=info["line"],
            lines=sorted(tr["lines"]),
            label=label,
            coll=info["coll"],
            step=self.step[0],
            expr=self.step[1],
            err=self.err,
            held=round(held, 2),
            elapsed=round(elapsed, 2),
            stack=info["stack"],
        )


def _loc(r):
    s = f"{r['file']}:{r['func']}"
    if r.get("label"):
        return s + ":" + r["label"]
    # a spinning frame's sampled line is random, so only a frame stuck in one call is keyed by its line
    return s if r["kind"] == "cpu" else s + f":L{r['line']}"


def hang_finding(reports):
    """One HANG finding from the per-rank watchdog reports (None for a rank that sent none)."""
    got = [(i, r) for i, r in enumerate(reports) if r]
    if not got:
        return dict(kind="HANG", step=-1, expr="", msg="program timeout")
    cpu = [(i, r) for i, r in got if r["kind"] == "cpu"]
    coll = [(i, r) for i, r in got if r["kind"] == "collective"]
    if cpu:
        loc = Counter(_loc(r) for _, r in cpu).most_common(1)[0][0]
        where = "cpu|" + loc
        lead = next(r for _, r in cpu if _loc(r) == loc)
    elif coll:
        name = Counter(r["coll"] for _, r in coll).most_common(1)[0][0]
        inside = {i for i, r in coll if r["coll"] == name}
        missing = [i for i in range(len(reports)) if i not in inside]
        where = f"collective|{name}|ranks-missing:{missing}"
        lead = next(r for _, r in coll if r["coll"] == name)
        errs = [reports[i]["err"] for i in missing if reports[i] and reports[i].get("err")]
        if errs:
            # the missing ranks raised in this step and wait in the harness's error-flag exchange while the others
            # wait in the step's own collective: a rank-divergent error that deadlocks instead of being exchanged
            return dict(
                kind="RANK_DIVERGENT_ERROR",
                step=lead["step"],
                expr=lead["expr"],
                msg=Counter(errs).most_common(1)[0][0],
                hang=where,
                ranks=[f"{r['kind']}:{r.get('coll') or _loc(r)}" if r else None for r in reports],
            )
    else:
        loc = Counter(_loc(r) for _, r in got).most_common(1)[0][0]
        where = "stuck|" + loc
        lead = next(r for _, r in got if _loc(r) == loc)
    states = [f"{r['kind']}:{r.get('coll') or _loc(r)}" if r else None for r in reports]
    return dict(
        kind="HANG",
        step=lead["step"],
        expr=lead["expr"],
        msg=where,
        where=where,
        secs=max(r["elapsed"] for _, r in got),
        ranks=states,
    )
