from __future__ import annotations

import re

import torch

from .reference import Val, run_reference

TOL = {
    torch.float16: (1e-2, 1e-2),
    torch.bfloat16: (2e-2, 2e-2),
    torch.float32: (1e-4, 1e-5),
    torch.float64: (1e-7, 1e-9),
    torch.complex64: (1e-4, 1e-5),
}

MEMCORRUPT = {-11: "SIGSEGV", -6: "SIGABRT", -4: "SIGILL", -8: "SIGFPE", -7: "SIGBUS"}


def norm(msg: str) -> str:
    msg = re.sub(r"\[[^\]]*/[^\]]*\]", "[..]", msg)
    msg = re.sub(r"0x[0-9a-fA-F]+", "#", msg)
    msg = re.sub(r"\d+", "#", msg)
    return msg[:160]


def touched_from(prog, at):
    out = set()
    for c in prog["calls"][at:]:
        out |= set(c.get("rets", {}).values())
        for src in [c["args"], *c.get("div", {}).values()]:
            out |= {v for v in src.values() if isinstance(v, str)}
    return out


def compare(ref, rs, skip):
    bad, n = [], 0
    for r, out in enumerate(rs):
        for var, val in ref.t[r].items():
            if val.v is None or var in skip or var not in out["outputs"]:
                continue
            n += 1
            if not close(out["outputs"][var], val.v):
                bad.append((r, var))
        for var, lst in ref.lists[r].items():
            got = out["lists"].get(var)
            if got is None or var in skip:
                continue
            n += 1
            if len(got) != len(lst) or not all(entry_ok(g, e) for g, e in zip(got, lst)):
                bad.append((r, var))
    return bad, n


def entry_ok(got, exp):
    if isinstance(exp, Val):  # value unspecified (e.g. a non-root reduce buffer), dtype and shape are not
        return got.dtype == exp.dtype and tuple(got.shape) == exp.shape
    if isinstance(exp, torch.Tensor):
        return close(got, exp)
    return got == exp


def close(a, b):
    if a is None or b is None:
        return True
    if a.dtype != b.dtype or tuple(a.shape) != tuple(b.shape):
        return False
    if a.numel() == 0:
        return True
    if a.dtype in TOL:
        rt, at = TOL[a.dtype]
        return bool(
            torch.allclose(
                a.to(torch.complex128 if a.is_complex() else torch.float64),
                b.to(torch.complex128 if b.is_complex() else torch.float64),
                rtol=rt,
                atol=at,
                equal_nan=True,
            )
        )
    return bool(torch.equal(a, b))


def is_timeout(msg):
    m = msg.lower()
    return "timed out" in m or "timeout" in m or "timing out" in m


NCCL_WATCHDOG = re.compile(r"(?i)watchdog caught collective operation timeout|timeout at nccl work")
# ProcessGroupNCCL's check_gpu_single_tensor / check_gpu_tensors_same_device: documented NCCL input rules that the
# Gloo-shaped reference does not model (desc.NCCL_SEMANTICS, "input layout"). The raising rank leaves the program,
# so whatever its peers do afterwards (time out, block, finish) is a consequence, not a finding.
NCCL_REJECT = re.compile(
    r"Tensors must be contiguous|Tensors must be non-overlapping and dense|Tensors for P2P must be non-overlapping"
)


def nccl_rejected(results):
    """True if a rank that reported raised one of NCCL's documented layout rejections."""
    return any(
        e and e.get("type") == "ValueError" and NCCL_REJECT.search(e.get("msg", ""))
        for r in results
        if r
        for e in [r.get("exc")]
    )


def effective_kind(res, backend="gloo"):
    """A rank killed by the NCCL watchdog after a collective timeout is a hang, not a crash."""
    if backend == "nccl" and res["kind"] == "crash" and NCCL_WATCHDOG.search(res.get("crash_log", "")):
        return "hang"
    return res["kind"]


def classify(prog, res, ref=None, backend="gloo"):
    """Findings for one run. Signatures from a backend other than gloo start with "<backend>:"."""
    ref = ref or run_reference(prog)
    findings, info = _classify(prog, res, ref, backend)
    if backend != "gloo":
        findings = [dict(f, sig=f"{backend}:{f['sig']}") for f in findings]
    return findings, info


def _classify(prog, res, ref, backend):
    kind = effective_kind(res, backend)
    if kind != res["kind"]:
        res = dict(res, kind=kind, nccl_watchdog=True)
    info = {
        "ref": ref.status,
        "ref_at": ref.at,
        "ref_why": ref.why,
        "racy": ref.racy,
        "exc": 0,
        "timeouts": 0,
        "detail_mismatch": 0,
        "checked": False,
        "oom": False,
        "nccl_unchecked": 0,
        "nccl_rejected": 0,
    }
    victims = {c["args"]["victim"] for c in prog["calls"] if c["op"] == "crash_rank"}
    if res["kind"] == "crash" and victims:
        # crash_rank SIGKILLs its victim on purpose; only a survivor dying of memory corruption counts
        bad = [
            (r, MEMCORRUPT[e]) for r, e in enumerate(res.get("exitcodes") or []) if r not in victims and e in MEMCORRUPT
        ]
        if not bad:
            return [], info
        detail = f"victims={sorted(victims)} survivor crash {bad}; log:\n{res.get('crash_log', '')[-1200:]}"
        return [dict(kind="KILL_SURVIVOR_CRASH", sig=f"KILL_SURVIVOR_CRASH|{bad[0][1]}", detail=detail)], info
    if res["kind"] == "crash":
        if -9 in (res.get("exitcodes") or []):  # SIGKILL from the OOM killer, not a torch crash
            info["oom"] = True
            return [], info
        log = res.get("crash_log", "")
        m = re.search(
            r"(libc\+\+abi[^|]*|Segmentation fault|terminate called[^|]*|EnforceNotMet[^|]*|Fatal Python error[^|]*"
            r"|Check failed[^|]*)",
            log,
        )
        # which ranks died varies from run to run for the same bug, so only the kinds of death go in the signature
        codes = sorted({MEMCORRUPT.get(e, f"exit {e}") for e in res.get("exitcodes") or [] if e})
        head = norm(m.group(1)) if m else "+".join(codes) or "no exit code"
        return [dict(kind="CRASH", sig=f"CRASH|{ref.status}|{head}", detail=log[-1500:])], info
    if res["kind"] == "hang" and backend == "nccl" and ref.status == "invalid" and not victims:
        info["nccl_unchecked"] = 1  # NCCL does not detect the mismatch; blocking until the timeout is documented
        return [], info
    if res["kind"] in ("hang", "ok") and backend == "nccl" and nccl_rejected(res["results"]):
        info["nccl_rejected"] = 1
        if res["kind"] == "hang":
            return [], info
    if res["kind"] == "hang":
        detail = f"stuck ranks {res.get('stuck_ranks')}"
        if res.get("hang_log"):
            detail += "\n" + res["hang_log"][-3000:]
        if victims:
            return [dict(kind="KILL_SURVIVOR_HANG", sig=f"KILL_SURVIVOR_HANG|{ref.status}", detail=detail)], info
        return [dict(kind="HANG", sig=f"HANG|{ref.status}", detail=detail)], info
    rs = res["results"]
    for r in rs:
        if r.get("executor_error"):
            return [
                dict(
                    kind="EXECUTOR_BUG",
                    sig="EXECUTOR_BUG|" + norm(r["executor_error"].splitlines()[-1]),
                    detail=r["executor_error"],
                )
            ], info
    findings = []
    excs = [r.get("exc") for r in rs]
    wexcs = [e for r in rs for e in r.get("wait_exc", [])]
    allmsgs = [e["msg"] for e in excs if e] + [e["msg"] for e in wexcs]
    info["exc"] = len(allmsgs)
    # NCCL words it "... before timing out" or "timed out in blocking wait", depending on who noticed
    timed_out = is_timeout if backend == "nccl" else (lambda m: "timed out" in m.lower())
    info["timeouts"] = sum(timed_out(m) for m in allmsgs)
    info["detail_mismatch"] = sum("mismatch" in m.lower() and "collective" in m.lower() for m in allmsgs)
    if any(r.get("guard_bad") for r in rs):
        layouts = sorted(
            {v["layout"] for c in prog["calls"] for v in c["args"].values() if isinstance(v, dict) and "layout" in v}
        )
        findings.append(
            dict(
                kind="GUARD",
                sig=f"GUARD|{ref.status}|{','.join(layouts)}",
                detail=f"guard corruption on ranks {[i for i, r in enumerate(rs) if r.get('guard_bad')]}",
            )
        )
    undefined = any(e and e["type"] == "FuzzerUndefinedVar" for e in excs)
    if undefined or info["nccl_rejected"]:
        return findings, info
    if ref.status == "valid" and info["exc"]:
        raised = [e for e in excs if e]
        cands = [e for e in raised if not is_timeout(e["msg"])] or raised
        first = min(cands, key=lambda e: e["idx"]) if cands else dict(op="wait", **wexcs[0])
        findings.append(
            dict(
                kind="VALID_ERROR",
                sig=f"VALID_ERROR|{first['op']}|{first['type']}|{norm(first['msg'])}",
                detail=first.get("full", first["msg"]),
            )
        )
    elif not ref.racy and (
        ref.status == "valid"
        or ref.status == "uncertain"
        and not wexcs
        and all(e is None or e["idx"] >= ref.at for e in excs)
    ):
        bad, n = compare(ref, rs, touched_from(prog, ref.at) if ref.at is not None else set())
        info["checked"] = n > 0
        if bad:
            # errors propagate forward, so the earliest call that produced a wrong value is the likeliest cause
            at, r, var = min((ref.origin(r, var), r, var) for r, var in bad)
            opname = prog["calls"][at]["op"]
            exp = ref.t[r].get(var)
            dt = str(exp.v.dtype) if exp is not None and exp.v is not None else "list"
            findings.append(
                dict(kind="WRONG_RESULT", sig=f"WRONG_RESULT|{opname}|{dt}", detail=f"mismatch on {bad[:6]}")
            )
    elif ref.status == "invalid" and info["exc"] == 0 and backend == "nccl":
        info["nccl_unchecked"] = 1  # every "invalid" in the reference is a cross-rank mismatch NCCL does not check
    elif ref.status == "invalid" and info["exc"] == 0:
        findings.append(
            dict(
                kind="SILENT_ACCEPT",
                sig=f"SILENT_ACCEPT|{prog['calls'][ref.at]['op']}|{norm(ref.why)}",
                detail=f"reference says invalid at call {ref.at}: {ref.why}; all ranks completed",
            )
        )
    return findings, info


def outcome(res, backend="gloo"):
    """Coarse outcome class of one run, for comparing backends: ok, raise, timeout, hang or crash."""
    kind = effective_kind(res, backend)
    if kind != "ok":
        return kind, None
    excs = [r["exc"] for r in res["results"] if r and r.get("exc")]
    if excs:
        e = min(excs, key=lambda e: e["idx"])
        return ("timeout" if all(is_timeout(x["msg"]) for x in excs) else "raise"), e
    wexcs = [e for r in res["results"] if r for e in r.get("wait_exc", [])]
    if wexcs:
        return ("timeout" if all(is_timeout(x["msg"]) for x in wexcs) else "raise"), dict(op="wait", **wexcs[0])
    return "ok", None


def _same_entry(x, y, exp):
    if isinstance(exp, Val):
        return isinstance(x, torch.Tensor) and isinstance(y, torch.Tensor) and x.dtype == y.dtype and x.shape == y.shape
    if isinstance(exp, torch.Tensor):
        return close(x, y)
    return x == y


def cross_compare(prog, ref, runs):
    """Compare two runs of one program on different backends: [(backend, res), (backend, res)].

    Only programs the reference calls valid and race-free are compared: for anything else the backends may legally
    differ (NCCL_SEMANTICS). Values are compared where the reference specifies them, against each other, with the
    single-run tolerances (integers exact).
    """
    if ref.status != "valid" or ref.racy:
        return []
    (na, ra), (nb, rb) = runs
    (ca, ea), (cb, eb) = outcome(ra, na), outcome(rb, nb)
    if {"crash", "hang"} & {ca, cb}:
        # each backend's own oracle reports these, and Gloo's crashes are not deterministic (gloo vs gloo differs)
        return []
    if ca != cb:
        e = ea or eb or {}
        op = e.get("op", "?")
        return [
            dict(
                kind="XBACKEND_OUTCOME",
                sig=f"XBACKEND_OUTCOME|{op}|{na}={ca}|{nb}={cb}",
                detail=f"{na}: {ca} {ea}\n{nb}: {cb} {eb}",
            )
        ]
    if ca != "ok":
        return []
    bad = []
    for r, (xa, xb) in enumerate(zip(ra["results"], rb["results"])):
        for var, val in ref.t[r].items():
            if val.v is not None and var in xa["outputs"] and var in xb["outputs"]:
                if not close(xa["outputs"][var], xb["outputs"][var]):
                    bad.append((r, var))
        for var, lst in ref.lists[r].items():
            ga, gb = xa["lists"].get(var), xb["lists"].get(var)
            if ga is None or gb is None:
                continue
            if len(ga) != len(gb) or not all(_same_entry(x, y, e) for x, y, e in zip(ga, gb, lst)):
                bad.append((r, var))
    if not bad:
        return []
    at, r, var = min((ref.origin(r, var), r, var) for r, var in bad)
    exp = ref.t[r].get(var)
    dt = str(exp.v.dtype) if exp is not None and exp.v is not None else "list"
    return [
        dict(
            kind="XBACKEND_VALUE",
            sig=f"XBACKEND_VALUE|{prog['calls'][at]['op']}|{dt}|{na}!={nb}",
            detail=f"{na} and {nb} disagree on {bad[:6]}",
        )
    ]
