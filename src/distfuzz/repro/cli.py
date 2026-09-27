from __future__ import annotations

import argparse
import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

from . import bundle, dedup, dock, render, upstream
from .bundle import make_bundle, rerun_reliability
from .report import parse
from .versions import matrix

_lock = threading.Lock()


def log(msg):
    with _lock:
        print(msg, flush=True)
        os.makedirs(dock.CACHE, exist_ok=True)
        with open(os.path.join(dock.CACHE, "repro.log"), "a") as f:
            f.write(msg + "\n")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m distfuzz.repro")
    ap.add_argument("--bundles", default=bundle.BUNDLES, help="output directory for bug bundles")
    ap.add_argument("--cache", default=dock.CACHE, help="image metadata, build logs and scratch work dirs")
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("images")
    a.add_argument("--only")
    a.add_argument("--force", action="store_true")
    b = sub.add_parser("bundle")
    b.add_argument("source")
    b.add_argument("--name", required=True)
    b.add_argument("--n", type=int, default=20)
    b.add_argument("--bisect-n", type=int, default=4)
    b.add_argument("--no-bisect", action="store_true")
    b.add_argument("--base", default="2.14.0")
    b.add_argument("--world", type=int)
    s = sub.add_parser("specs")
    s.add_argument("file")
    s.add_argument("--only")
    s.add_argument("--jobs", type=int, default=2)
    p = sub.add_parser("parse")
    p.add_argument("log")
    p.add_argument("--world", type=int, default=4)
    p.add_argument("--rc", type=int, default=1)
    r = sub.add_parser("reliability")
    r.add_argument("name")
    r.add_argument("--n", type=int, default=20)
    sub.add_parser("dedup")
    sub.add_parser("dashboard")
    u = sub.add_parser("upstream")
    u.add_argument("--specs", default=os.path.join("examples", "repro_specs.json"))
    o = ap.parse_args(argv)
    bundle.BUNDLES, dock.CACHE = o.bundles, o.cache
    BUNDLES = o.bundles
    if o.cmd == "images":
        keys = o.only.split(",") if o.only else [m["key"] for m in matrix()]
        for k in keys:
            m = dock.build_image(k, force=o.force, log=log)
            log(f"{k}: {m['torch_version']} {m['image_id'][:19]}")
    elif o.cmd == "bundle":
        spec = dict(
            name=o.name, source=o.source, n=o.n, bisect_n=o.bisect_n, bisect=not o.no_bisect, base=o.base, world=o.world
        )
        make_bundle(spec, log=log)
        dedup.run(BUNDLES)
    elif o.cmd == "specs":
        specs = json.load(open(o.file))
        if o.only:
            want = set(o.only.split(","))
            specs = [x for x in specs if x["name"] in want]

        def one(spec):
            try:
                return make_bundle(spec, log=log)
            except Exception as e:  # keep going; one broken bundle must not stop the batch
                import traceback

                log(f"[{spec['name']}] FAILED: {e}\n{traceback.format_exc()}")

        with ThreadPoolExecutor(o.jobs) as ex:
            list(ex.map(one, specs))
        log(json.dumps(dedup.run(BUNDLES), indent=1))
    elif o.cmd == "parse":
        rep = parse(open(o.log).read(), o.rc, False, o.world)
        print(
            json.dumps(
                dict(
                    title=rep.title,
                    alt=rep.alt_titles,
                    kind=rep.kind,
                    setup_error=rep.setup_error,
                    evidence=rep.evidence,
                ),
                indent=1,
            )
        )
    elif o.cmd == "reliability":
        rerun_reliability(o.name, o.n, log=log)
        dedup.run(BUNDLES)
    elif o.cmd == "upstream":
        specs = {s["name"]: s for s in json.load(open(o.specs))}
        for name in sorted(os.listdir(BUNDLES)):
            p = os.path.join(BUNDLES, name, "bundle.json")
            if os.path.exists(p) and name in specs:
                d = json.load(open(p))
                if d.get("title"):
                    s = specs[name]
                    d["upstream"] = upstream.related(d["title"], s.get("queries", []), s.get("known", []))
                    json.dump(d, open(p, "w"), indent=1)
        dedup.run(BUNDLES)
    elif o.cmd == "dedup":
        print(json.dumps(dedup.run(BUNDLES), indent=1))
    elif o.cmd == "dashboard":
        render.dashboard(BUNDLES)


if __name__ == "__main__":
    sys.exit(main())
