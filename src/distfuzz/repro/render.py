from __future__ import annotations

import html
import json
import os
import re

from .versions import order


def _vtable(bis):
    rows = ["| torch | verdict | bad | good | other failure | untestable (API missing) |", "|---|---|---|---|---|---|"]
    for r in bis["results"]:
        other = "; ".join(f"{t} x{c}" for t, c in r["other_titles"].items() if not t.startswith("setup")) or ""
        rows.append(
            f"| {r['version']} | **{r['verdict']}** | {r['bad']} | {r['good']} | {other} | {r['setup'] or ''} |"
        )
    return "\n".join(rows)


def affected_str(d):
    b = d.get("bisect")
    if not b:
        return "not bisected"
    if not b["affected"]:
        return "none of the tested versions"
    return ", ".join(b["affected"])


def markdown(d):
    rel = d.get("reliability", {})
    b = d.get("bisect")
    L = [f"# {d['title']}", ""]
    L += [
        f"- **Bundle:** `{d['name']}`  ",
        f"- **Found by:** {d['fuzzer']} (`{d['source']}`)"
        + (f", fuzzer signature `{d['fuzzer_sig']}`" if d.get("fuzzer_sig") else ""),
        f"- **Kind:** {d['kind']}"
        + (f"; alt titles: {', '.join('`' + t + '`' for t in d['alt_titles'])}" if d.get("alt_titles") else ""),
        f"- **Reliability (torch {rel.get('image')}):** {rel.get('k')}/{rel.get('n')} fresh containers -> **{rel.get('verdict')}**"
        + (f" (other outcomes: {rel['other_outcomes']})" if rel.get("other_outcomes") else ""),
        f"- **Versions affected:** {affected_str(d)}",
    ]
    if b:
        L.append(
            f"- **First bad:** {b['first_bad'] or '-'} (last good before it: {b['last_good_before'] or 'none tested'}); "
            f"**fixed in:** {b['fixed_in'] or 'not fixed in tested builds'}" + (f". {b['note']}" if b["note"] else "")
        )
    if d.get("dup_of"):
        L.append(f"- **Duplicate of:** `{d['dup_of']}` (same normalized title)")
    if d.get("dups"):
        L.append(f"- **Duplicates merged here:** {', '.join('`' + x + '`' for x in d['dups'])}")
    L += [
        f"- **Environment:** torch {d['env']['torch']} CPU wheel, Python 3.12, Gloo, world size {d['world']}, "
        f"container `--memory 3g --cpus 4 --network none`",
        "",
    ]
    L += [
        "## Reproduce",
        "",
        "```sh",
        "./run.sh        # builds the pinned image (Dockerfile + requirements.txt) and runs repro.py",
        "```",
        "",
    ]
    if d.get("minimize"):
        m = d["minimize"]
        if m.get("done"):
            L.append(
                f"Minimized from {m['units_before']} to {m['units_after']} program steps with ddmin "
                f"({m['tests']} container runs); the emitted standalone script was re-run and "
                + (
                    "still triggers the same title."
                    if m.get("verified")
                    else "**lost the bug, so the original program is kept**."
                )
            )
        else:
            L.append(f"Minimization: {m.get('why')}.")
        L.append("")
    L += ["```python", open(os.path.join(d["_dir"], "repro.py")).read().rstrip(), "```", ""]
    L += [
        "## Actual vs expected",
        "",
        f"**Expected:** {d.get('expected', '')}",
        "",
        "**Actual** (parsed from the run output):",
        "",
        "```",
    ]
    L += [str(x) for x in d.get("evidence", [])][:14]
    L += ["```", ""]
    if b:
        L += [
            f"## Version bisection ({b['n_per_version']} runs per version, pkg/bisect decision thresholds)",
            "",
            _vtable(b),
            "",
            f"nightly = `{b['nightly']}`. 'untestable' = the repro cannot run because the API it uses does not exist in that "
            "version (import error); 'skip' = results not decisive (unrelated failure or too few bad runs).",
            "",
        ]
    up = d.get("upstream", {})
    L += [f"## Related upstream issues (GitHub API, {up.get('searched_at', '')})", ""]
    if up.get("known"):
        L.append("Issues/PRs cited in the original triage notes (state fetched live):")
        for k in up["known"]:
            L.append(
                f"- [#{k['number']}]({k.get('url', '')}) {k.get('title', k.get('error', ''))} ({k.get('state', '?')}{', PR' if k.get('is_pr') else ''})"
            )
        L.append("")
    for s in up.get("searches", []):
        L.append(f"Search `{s['query']}` ({s.get('total', 0)} hits; automated, not verified as the same bug):")
        for it in s["items"]:
            L.append(
                f"- [#{it['number']}]({it['url']}) {it['title']} ({it['state']}{', PR' if it['is_pr'] else ''}, updated {it['updated']})"
            )
        if not s["items"]:
            L.append("- no results" + (f" ({s['error']})" if s.get("error") else ""))
        L.append("")
    if d.get("notes"):
        L += ["## Triage notes", "", d["notes"], ""]
    L += [
        "## Files",
        "",
        "`repro.py` standalone repro, `Dockerfile` + `requirements.txt` pinned env, `run.sh` one command, "
        "`bundle.json` all data, `logs/` raw outputs (reproduce, reliability sample, one log per bisected version)"
        + (", `program*.json` fuzzer program" if any(f.startswith("program") for f in os.listdir(d["_dir"])) else "")
        + ".",
        "",
    ]
    return "\n".join(L)


CSS = """
:root{--bg:#fbfbfa;--fg:#1d1d1f;--mut:#6b6b70;--line:#e3e3e0;--bad:#b3261e;--good:#1b7f3b;--warn:#9a6700;--code:#f2f2ef}
@media (prefers-color-scheme: dark){:root{--bg:#161618;--fg:#ececec;--mut:#9a9aa0;--line:#2c2c30;--bad:#ff6b61;--good:#4cc27a;--warn:#e0b341;--code:#1f1f22}}
body{background:var(--bg);color:var(--fg);font:15px/1.5 -apple-system,system-ui,sans-serif;max-width:1100px;margin:0 auto;padding:24px 16px}
table{border-collapse:collapse;width:100%;font-size:13px}td,th{border-bottom:1px solid var(--line);padding:6px 8px;text-align:left;vertical-align:top}
pre,code{background:var(--code);font:12.5px/1.45 ui-monospace,Menlo,monospace}pre{padding:12px;overflow:auto;border-radius:6px}
a{color:inherit}.bad{color:var(--bad);font-weight:600}.good{color:var(--good);font-weight:600}.skip,.untestable{color:var(--mut)}
.mut{color:var(--mut)}.v{display:inline-block;min-width:3.2em;text-align:center;border-radius:4px;padding:0 3px;margin:1px;font-size:11px;border:1px solid var(--line)}
.v.bad{background:color-mix(in srgb,var(--bad) 18%,transparent)}.v.good{background:color-mix(in srgb,var(--good) 15%,transparent)}
.wrap{overflow-x:auto}
"""


def md_to_html(md, title):
    """Tiny markdown renderer (headings, lists, tables, code fences, links, bold, inline code)."""
    out, lines, i = [], md.splitlines(), 0

    def inline(s):
        s = html.escape(s)
        s = re.sub(r"`([^`]+)`", r"<code>\1</code>", s)
        s = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", s)
        s = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r'<a href="\2">\1</a>', s)
        return s

    while i < len(lines):
        ln = lines[i]
        if ln.startswith("```"):
            j = i + 1
            buf = []
            while j < len(lines) and not lines[j].startswith("```"):
                buf.append(lines[j])
                j += 1
            out.append("<pre>" + html.escape("\n".join(buf)) + "</pre>")
            i = j + 1
            continue
        if ln.startswith("|"):
            rows = []
            while i < len(lines) and lines[i].startswith("|"):
                rows.append([c.strip() for c in lines[i].strip("|").split("|")])
                i += 1
            h = "".join(f"<th>{inline(c)}</th>" for c in rows[0])
            body = "".join("<tr>" + "".join(f"<td>{inline(c)}</td>" for c in r) + "</tr>" for r in rows[2:])
            out.append(f'<div class="wrap"><table><tr>{h}</tr>{body}</table></div>')
            continue
        m = re.match(r"^(#+) (.*)", ln)
        if m:
            out.append(f"<h{len(m.group(1))}>{inline(m.group(2))}</h{len(m.group(1))}>")
        elif ln.startswith("- "):
            buf = []
            while i < len(lines) and lines[i].startswith("- "):
                buf.append(f"<li>{inline(lines[i][2:])}</li>")
                i += 1
            out.append("<ul>" + "".join(buf) + "</ul>")
            continue
        elif ln.strip():
            out.append(f"<p>{inline(ln)}</p>")
        i += 1
    return (
        f"<!doctype html><html><head><meta charset=utf-8><meta name=viewport content='width=device-width,initial-scale=1'>"
        f"<title>{html.escape(title)}</title><style>{CSS}</style></head><body>"
        f"<p class=mut><a href='../index.html'>&larr; all bundles</a></p>{''.join(out)}</body></html>"
    )


def write_reports(bdir, data):
    d = dict(data, _dir=bdir)
    md = markdown(d)
    open(os.path.join(bdir, "REPORT.md"), "w").write(md)
    open(os.path.join(bdir, "report.html"), "w").write(md_to_html(md, data["title"]))


def dashboard(bundles_dir):
    rows = []
    for name in sorted(os.listdir(bundles_dir)):
        p = os.path.join(bundles_dir, name, "bundle.json")
        if os.path.exists(p):
            rows.append(json.load(open(p)))
    vers = []
    for d in rows:
        for r in (d.get("bisect") or {}).get("results", []):
            if r["version"] not in vers:
                vers.append(r["version"])
    vers.sort(key=order)
    trs = []
    for d in sorted(rows, key=lambda d: (d.get("dup_of") is not None, d.get("title") or "")):
        rel = d.get("reliability", {})
        cells = ""
        res = {r["version"]: r for r in (d.get("bisect") or {}).get("results", [])}
        for v in vers:
            r = res.get(v)
            if r is None:
                cells += "<span class='v mut'>·</span>"
            else:
                lab = {"bad": "BAD", "good": "ok", "skip": "?", "untestable": "n/a"}[r["verdict"]]
                cells += f"<span class='v {r['verdict']}' title='{v}: bad {r['bad']}/{r['runs']}'>{lab}</span>"
        b = d.get("bisect") or {}
        dup = f"<br><span class=mut>dup of {html.escape(d['dup_of'])}</span>" if d.get("dup_of") else ""
        trs.append(
            f"<tr><td><a href='{name_of(d)}/report.html'>{html.escape(d.get('title') or '(not reproduced)')}</a>{dup}"
            f"<br><span class=mut>{html.escape(d['name'])} · {d['fuzzer']}</span></td>"
            f"<td>{rel.get('k', '-')}/{rel.get('n', '-')}<br><span class=mut>{rel.get('verdict', '')}</span></td>"
            f"<td>{b.get('first_bad') or '-'}</td><td>{b.get('fixed_in') or '-'}</td><td>{cells}</td></tr>"
        )
    head = "".join(f"<span class='v'>{v.replace('nightly', 'ntly')}</span>" for v in vers)
    h = (
        f"<!doctype html><html><head><meta charset=utf-8><meta name=viewport content='width=device-width,initial-scale=1'>"
        f"<title>distfuzz.repro bundles</title><style>{CSS}</style></head><body><h1>distfuzz.repro bundles</h1>"
        f"<p class=mut>{len(rows)} bundles. Title = normalized dedup key. Reliability = runs reproducing the title in fresh containers "
        f"on the base version. Versions: BAD reproduces, ok = expected output, ? = inconclusive, n/a = API missing.</p>"
        f"<div class=wrap><table><tr><th>bug</th><th>reliability</th><th>first bad</th><th>fixed in</th><th>{head}</th></tr>"
        + "".join(trs)
        + "</table></div></body></html>"
    )
    open(os.path.join(bundles_dir, "index.html"), "w").write(h)


def name_of(d):
    return d["name"]
