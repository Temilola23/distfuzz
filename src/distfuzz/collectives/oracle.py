from __future__ import annotations

import re

import torch

from .reference import run_reference

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
            if len(got) != len(lst) or any(
                not (close(g, e) if isinstance(e, torch.Tensor) else g == e) for g, e in zip(got, lst)
            ):
                bad.append((r, var))
    return bad, n


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
    return "timed out" in m or "timeout" in m


def classify(prog, res, ref=None):
    ref = ref or run_reference(prog)
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
            r"(libc\+\+abi[^|]*|Segmentation fault|terminate called[^|]*|EnforceNotMet[^|]*|Fatal Python error[^|]*)",
            log,
        )
        head = norm(m.group(1)) if m else f"exitcodes={res.get('exitcodes')}"
        return [dict(kind="CRASH", sig=f"CRASH|{ref.status}|{head}", detail=log[-1500:])], info
    if res["kind"] == "hang":
        detail = f"stuck ranks {res.get('stuck_ranks')}"
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
    info["timeouts"] = sum("timed out" in m.lower() for m in allmsgs)
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
    if undefined:
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
            r, var = bad[0]
            opname = [c["op"] for c in prog["calls"] if var in c.get("rets", {}).values() or var == c["args"].get("t")][
                -1
            ]
            exp = ref.t[r].get(var)
            dt = str(exp.v.dtype) if exp is not None and exp.v is not None else "list"
            findings.append(
                dict(kind="WRONG_RESULT", sig=f"WRONG_RESULT|{opname}|{dt}", detail=f"mismatch on {bad[:6]}")
            )
    elif ref.status == "invalid" and info["exc"] == 0:
        findings.append(
            dict(
                kind="SILENT_ACCEPT",
                sig=f"SILENT_ACCEPT|{prog['calls'][ref.at]['op']}|{norm(ref.why)}",
                detail=f"reference says invalid at call {ref.at}: {ref.why}; all ranks completed",
            )
        )
    return findings, info
