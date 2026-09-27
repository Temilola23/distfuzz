from __future__ import annotations

import json
import os

from . import pipeline, render


def load_all(bundles_dir):
    out = []
    for name in sorted(os.listdir(bundles_dir)):
        p = os.path.join(bundles_dir, name, "bundle.json")
        if os.path.exists(p):
            out.append((p, json.load(open(p))))
    return out


def group(datas):
    groups = {}
    for d in datas:
        if d.get("title"):
            groups.setdefault(d["title"], []).append(d)
    return groups


def canonical(members):
    return sorted(members, key=lambda d: (not d.get("bisect"), d["name"]))[0]


def run(bundles_dir):
    items = load_all(bundles_dir)
    datas = [d for _, d in items]
    groups = group(datas)
    for mem in groups.values():
        c = canonical(mem)
        for d in mem:
            d.pop("dup_of", None)
            d.pop("dups", None)
        c["dups"] = [d["name"] for d in mem if d is not c]
        if not c["dups"]:
            c.pop("dups")
        for d in mem:
            if d is not c:
                d["dup_of"] = c["name"]
    for d in datas:
        d["related"] = sorted(
            {o["name"] for o in datas if o is not d and o.get("title") and o["title"] in d.get("alt_titles", [])}
        )
    for p, d in items:
        if d.get("bisect"):
            d["bisect"].update(pipeline.summarize(d["bisect"]["results"]))
        json.dump(d, open(p, "w"), indent=1)
        if d.get("title"):
            render.write_reports(os.path.dirname(p), d)
    render.dashboard(bundles_dir)
    return {t: [d["name"] for d in m] for t, m in groups.items()}
