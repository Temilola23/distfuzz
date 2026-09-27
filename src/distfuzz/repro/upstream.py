from __future__ import annotations

import json
import re
import subprocess
import time

REPO = "pytorch/pytorch"
STOP = {"in", "wrong", "result", "hang", "crash", "rank", "divergent", "c10d", "torch", "unknown", "program"}


def _gh(args, retries=3):
    for i in range(retries):
        p = subprocess.run(["gh", "api", "-X", "GET", *args], capture_output=True, text=True, timeout=60)
        if p.returncode == 0:
            return json.loads(p.stdout)
        if "rate limit" in (p.stderr + p.stdout).lower():
            time.sleep(30 * (i + 1))
            continue
        return {"error": p.stderr.strip()[:300]}
    return {"error": "rate limited"}


def auto_queries(title: str):
    """Keywords from a normalized title: subsystem word + the op/frame (+ exception type)."""
    sub, _, rest = title.partition(": ")
    subw = {"DTensor": "DTensor", "c10d/gloo": "gloo"}.get(sub, "distributed")
    words = [w.strip("()") for w in re.split(r"[\s]+", rest) if w and w.lower().strip("()") not in STOP]
    words = [w for w in words if not w.startswith("SIG") and "-" not in w and not w.endswith(("Error", "Exception"))]
    q = [f"repo:{REPO} {subw} {' '.join(words)}".strip()]
    return q


def search(query, n=5):
    r = _gh(["search/issues", "-f", f"q={query}", "-f", f"per_page={n}"])
    if "error" in r:
        return dict(query=query, error=r["error"], items=[])
    items = [
        dict(
            number=it["number"],
            title=it["title"],
            state=it["state"],
            url=it["html_url"],
            is_pr="pull_request" in it,
            updated=it["updated_at"][:10],
        )
        for it in r.get("items", [])
    ]
    return dict(query=query, total=r.get("total_count", 0), items=items)


def fetch(number):
    r = _gh([f"repos/{REPO}/issues/{number}"])
    if "error" in r:
        return dict(number=number, error=r["error"])
    return dict(
        number=number,
        title=r["title"],
        state=r["state"],
        url=r["html_url"],
        is_pr="pull_request" in r,
        updated=r["updated_at"][:10],
    )


def related(title, extra_queries=(), known=()):
    out = dict(searched_at=time.strftime("%Y-%m-%d"), searches=[], known=[])
    for q in list(auto_queries(title)) + [f"repo:{REPO} {q}" if "repo:" not in q else q for q in extra_queries]:
        out["searches"].append(search(q))
        time.sleep(2.5)  # search API: 30 req/min authenticated
    for n in known:
        out["known"].append(fetch(n))
    return out
