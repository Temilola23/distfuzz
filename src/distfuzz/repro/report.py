from __future__ import annotations

import re
from dataclasses import dataclass, field


def normalize_text(s: str) -> str:
    """dynamicTitleReplacement-style scrubbing of free text (messages, not titles)."""
    s = re.sub(r"0x[0-9a-fA-F]+", "ADDR", s)
    s = re.sub(r"/[\w./-]+\.py", "FILE", s)
    s = re.sub(r"\[rank \d+\]", "[rank]", s)
    s = re.sub(r"(?<![A-Za-z_])-?\d+(\.\d+)?(e[-+]?\d+)?", "NUM", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s[:120]


# Python frames that are plumbing, not the bug (like pkg/report skipPatterns)
SKIP_FUNCS = {
    "<module>",
    "main",
    "run",
    "_wrap",
    "spawn",
    "start_processes",
    "join",
    "_call_impl",
    "_wrapped_call_impl",
    "__torch_dispatch__",
    "__torch_function__",
    "dispatch",
    "wrapper",
    "_dispatch_get",
    "inner",
    "fn",
    "decorate_context",
    "wrapped",
    "__call__",
    "show",
    "program",
    "run_path",
    "_process_worker",
    "worker",
    "_bootstrap",
    "_bootstrap_inner",
    "wait",
    "_run_mod",
    "dump_traceback_later",
    "case",
    "handler",
    "_propagate",
    "propagate",
    "propagate_op_sharding",
    "propagate_op_sharding_non_cached",
    "_op_dispatcher",
    "_run",
    "_handle",
    "full_tensor",
    "redistribute",
    "apply",
    "forward",
    "_try_wait",
    "_wait",
    "select",
    "poll",
    "_poll",
    "__exit__",
    "__enter__",
    "timeout",
    "_recv",
    "recv",
    "recv_bytes",
    "_recv_bytes",
    "get",
    "result",
    "_invoke_run",
    "run_once",
}
SKIP_FILES = ("multiprocessing/", "threading.py", "selectors.py", "/spawn.py", "faulthandler")

OP_ALIASES = {
    "SETITEM": "__setitem__",
    "R": "redistribute",
    "BW": "backward",
    "mk": "distribute_tensor",
    "CMP": "torch.compile",
}


def op_of(text: str) -> str:
    """Best-effort torch API name from a description/expression.

    'v7.fill_(3)' -> fill_ ; 'torch.linalg.vector_norm(v1, ord=inf)' -> vector_norm ;
    'sharded[6] = 100' -> __setitem__ ; 'mse_loss [Replicate()]x[Shard(2)]' -> mse_loss ;
    'scatter(out)' -> scatter ; 'max()' -> max
    """
    t = text.strip()
    if re.search(r"\w\s*\[[^\]]*\]\s*=[^=]", t):
        return "__setitem__"
    m = re.match(r"^(?:[\w.]+\.)?(\w+)\s*\(", t)  # outermost call at the start
    if m:
        name = m.group(1)
        if name in OP_ALIASES:
            return OP_ALIASES[name]
        if name.startswith("v") and name[1:].isdigit():
            pass
        else:
            return name
    m = re.match(r"^v\d+\.(\w+)\s*\(", t)  # method on a value
    if m:
        return m.group(1)
    m = re.search(r"\.(\w+)\s*\(", t)  # first method call anywhere
    if m:
        return m.group(1)
    m = re.match(r"^(\w+)", t)
    return OP_ALIASES.get(m.group(1), m.group(1)) if m else "unknown"


FRAME_RE = re.compile(r'^\s*File "(?P<file>[^"]+)", line (?P<line>\d+)(?:, in (?P<func>\S+)| in (?P<func2>\S+))?')
EXC_LINE_RE = re.compile(
    r"^(?P<type>(?:[A-Za-z_][\w.]*\.)?[A-Z]\w*(?:Error|Exception|Exit|Interrupt|Warning|Failure))(?::\s?(?P<msg>.*))?$"
)
WRAPPER_EXC = ("ProcessRaisedException", "ProcessExitedException", "ChildFailedError")


@dataclass
class Frame:
    file: str
    line: int
    func: str


def parse_tracebacks(text: str):
    """All 'Traceback (most recent call last):' blocks -> list of (frames, exc_type, msg)."""
    out = []
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        if lines[i].lstrip().startswith("Traceback (most recent call last):"):
            frames = []
            j = i + 1
            while j < len(lines):
                m = FRAME_RE.match(lines[j])
                if m:
                    frames.append(
                        Frame(m.group("file"), int(m.group("line")), m.group("func") or m.group("func2") or "?")
                    )
                    j += 1
                    continue
                s = lines[j].strip()
                em = EXC_LINE_RE.match(s)
                if em and not lines[j].startswith(" "):
                    out.append((frames, em.group("type").split(".")[-1], (em.group("msg") or "").strip()))
                    break
                j += 1
            i = j + 1
        else:
            i += 1
    return out


def guilty_frame(frames, prefer=("torch/distributed", "torch/")):
    """Innermost frame worth naming in a title (syzkaller: first non-skipped frame)."""
    for pref in prefer:
        for f in reversed(frames):
            if pref in f.file and f.func not in SKIP_FUNCS and not any(s in f.file for s in SKIP_FILES):
                return f.func
    for f in reversed(frames):
        if f.func not in SKIP_FUNCS and not any(s in f.file for s in SKIP_FILES):
            return f.func
    return frames[-1].func if frames else "unknown"


def parse_faulthandler(text: str):
    """faulthandler dump ('Timeout (0:00:20)!' + 'File ..., line N in func', most recent first)."""
    m = re.search(r"Timeout \(\d+:\d\d:\d\d(?:\.\d+)?\)!", text)
    if not m:
        return None
    frames = []
    for line in text[m.end() :].splitlines()[1:60]:
        fm = FRAME_RE.match(line)
        if fm:
            frames.append(Frame(fm.group("file"), int(fm.group("line")), fm.group("func") or fm.group("func2") or "?"))
        elif frames and not line.startswith(" "):
            break
    frames.reverse()  # most recent call last, like a traceback
    return frames


@dataclass
class Report:
    title: str | None
    kind: str | None  # WRONG_RESULT HANG CRASH EXCEPTION DIVERGENT OOB_WRITE
    alt_titles: list = field(default_factory=list)
    evidence: list = field(default_factory=list)  # output lines that justify the title
    setup_error: str | None = None  # import/API missing => cannot test this version
    completed: bool = False  # script reached its end marker / exit 0
    corrupted: bool = False  # interleaved multi-rank dump: title unreliable (pkg/report Corrupted)

    def titles(self):
        return ([self.title] if self.title else []) + [t for t in self.alt_titles if t != self.title]


GLOO_COLLATERAL = re.compile(
    r"(Connection closed by peer|Connection reset by peer|Timed out|timed out|"
    r"ProcessGroupGloo.*(Timeout|timeout)|Broken pipe|SIGTERM|EnforceNotMet.*read)"
)
SETUP_ERR = re.compile(
    r"^(ModuleNotFoundError|ImportError)|cannot import name|has no attribute '(Partial|DTensor|"
    r"distribute_tensor|init_device_mesh|full_tensor)'"
)


def subsystem_of(text: str, script: str = "") -> str:
    if (
        "DTensor" in script
        or "torch/distributed/tensor" in text
        or "torch/distributed/_tensor" in text
        or "DTensor" in text
    ):
        return "DTensor"
    if "gloo" in (script + text).lower():
        return "c10d/gloo"
    return "torch"


def parse(output: str, rc: int = 0, timed_out: bool = False, world: int = 4, script: str = "") -> Report:
    sub = subsystem_of(output, script)
    rep = Report(None, None)
    lines = output.splitlines()
    rep.completed = (rc == 0 and not timed_out) or "[distfuzz-repro] DONE" in output

    titles = []  # (priority, kind, title, evidence)

    # setup errors (API not present in this version) -> not a verdict about the bug
    for _frames, etype, msg in parse_tracebacks(output):
        if SETUP_ERR.search(f"{etype}: {msg}") or etype in ("ModuleNotFoundError", "ImportError"):
            rep.setup_error = f"{etype}: {msg}"[:200]
    m = re.search(r"\[distfuzz-repro\] SETUP-ERROR (.*)", output)
    if m:
        rep.setup_error = m.group(1)[:200]

    # wrong results: '<name>: MISMATCH ...' (cases.py / generated) and '<name>  WRONG ...' (sweep)
    for i, line in enumerate(lines):
        m = re.match(r"^\s*(?P<name>[^\[][^:]{0,120}?):\s+MISMATCH\b(?P<rest>.*)$", line)
        m2 = re.match(r"^(?P<name>[\w()./]+)\s+WRONG\b(?P<rest>.*)$", line)
        mm = m or m2
        if mm and ("print(" in line or 'f"' in line or line.lstrip().startswith(("File ", "#"))):
            mm = None  # source line quoted in a traceback
        if mm:
            op = op_of(mm.group("name"))
            ev = [line] + [x for x in lines[i + 1 : i + 3] if re.match(r"^\s+(ref|dt|expected|actual|got)\b", x)]
            titles.append((1, "WRONG_RESULT", f"{sub}: wrong result in {op}", ev))
            if m2 and "out-of-view writes ranks" in line and not line.rstrip().endswith("ranks []"):
                titles.append((3, "OOB_WRITE", f"{sub}: out-of-bounds write in {op}", [line]))
    for line in lines:
        m = re.match(r"^\s*(?:\[rank \d+\] )?OUT-OF-BOUNDS WRITE in (\w+)", line)
        if m:
            titles.append((3, "OOB_WRITE", f"{sub}: out-of-bounds write in {m.group(1)}", [line]))

    # hang: faulthandler watchdog, or killed by the outer timeout
    fh = parse_faulthandler(output)
    if fh is not None and not any("torch/" in f.file for f in fh):
        rep.corrupted = True
        titles.append((0, "HANG", f"{sub}: hang (corrupted report)", ["stack dumps of several ranks interleaved"]))
    elif fh is not None:
        func = guilty_frame(fh)
        titles.append(
            (0, "HANG", f"{sub}: hang in {func}", [x for x in lines if "Timeout (" in x][:1] + [f"  top frame: {func}"])
        )
    elif timed_out:
        titles.append((5, "HANG", f"{sub}: hang (no stack)", ["killed by outer timeout"]))

    # native crashes / aborts
    m = re.search(r"terminate called after throwing an instance of '([^']+)'", output)
    if m:
        titles.append((0, "CRASH", f"{sub}: abort ({m.group(1)})", [m.group(0)]))
    m = re.search(r"terminated with signal (SIG\w+)|Fatal Python error: (Segmentation fault|Aborted)", output)
    if m or rc in (-11, 139):
        sig = (m.group(1) or ("SIGSEGV" if "Segmentation" in (m.group(2) or "") else "SIGABRT")) if m else "SIGSEGV"
        if sig not in ("SIGTERM", "SIGKILL"):
            titles.append((0, "CRASH", f"{sub}: crash ({sig})", [m.group(0) if m else f"exit code {rc}"]))

    # per-rank exceptions -> rank-divergent vs uniform
    raised: dict[int, tuple[str, str]] = {}
    for line in lines:
        m = re.match(r"^\[rank (\d+)\] (?:raised|EXC) (\w+): ?(.*)$", line)
        if m:
            raised.setdefault(int(m.group(1)), (m.group(2), m.group(3)))
    tbs = [t for t in parse_tracebacks(output) if t[1] not in WRAPPER_EXC]
    primary = [t for t in tbs if not GLOO_COLLATERAL.search(t[2]) and not SETUP_ERR.search(f"{t[1]}: {t[2]}")]
    real_raised = {r: v for r, v in raised.items() if not GLOO_COLLATERAL.search(v[1])}
    if real_raised and len(real_raised) < world:
        etype, msg = next(iter(sorted(real_raised.items())))[1]
        mo = re.match(r"^(\w+)\(\)", msg)
        where = mo.group(1) if mo else (guilty_frame(primary[0][0]) if primary else "unknown")
        titles.append(
            (
                0,
                "DIVERGENT",
                f"{sub}: rank-divergent {etype} in {where}",
                [f"[rank {r}] raised {e}: {m_[:150]}" for r, (e, m_) in sorted(real_raised.items())]
                + [f"ranks without error: {sorted(set(range(world)) - set(real_raised))}"],
            )
        )
    if primary and not rep.setup_error:
        frames, etype, msg = primary[0]
        titles.append(
            (
                2,
                "EXCEPTION",
                f"{sub}: {etype} in {guilty_frame(frames)}",
                [f"{etype}: {msg[:200]}"]
                + [f'  File "{f.file.split("site-packages/")[-1]}", line {f.line}, in {f.func}' for f in frames[-3:]],
            )
        )
    elif real_raised and len(real_raised) >= world and not rep.setup_error:
        etype, msg = next(iter(sorted(real_raised.items())))[1]
        mo = re.match(r"^(\w+)\(\)", msg)
        titles.append(
            (2, "EXCEPTION", f"{sub}: {etype} in {mo.group(1) if mo else 'unknown'}", [f"{etype}: {msg[:200]}"])
        )

    if not titles:
        return rep
    titles.sort(key=lambda t: t[0])
    seen = []
    for _pri, kind, title, ev in titles:
        title = re.sub(r"\s+", " ", title)[:120]
        if title in seen:
            continue
        if rep.title is None:
            rep.title, rep.kind, rep.evidence = title, kind, ev
        else:
            rep.alt_titles.append(title)
            rep.evidence += ev[:2]
        seen.append(title)
    return rep


def same_bug(target: str, rep: Report) -> bool:
    """A run reproduces `target` if the target is among its titles (pkg/report AltTitles semantics)."""
    return target in rep.titles()
