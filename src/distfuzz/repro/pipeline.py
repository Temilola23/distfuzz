from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor

from . import dock, extract, gen_distfuzz, gen_dtensor
from .report import parse
from .versions import matrix, order

LOG = print


class Runner:
    """Writes a finding's script into a scratch dir and runs it in containers."""

    def __init__(self, finding, workdir, timeout=150):
        self.f, self.timeout = finding, timeout
        self.dir = workdir
        os.makedirs(self.dir, exist_ok=True)
        self.path = os.path.join(self.dir, "repro.py")
        self.set_script(finding.script)

    def set_script(self, src):
        open(self.path, "w").write(src)

    def _parse(self, res):
        rep = parse(res["out"], res["rc"], res["timed_out"], world=self.f.world, script=self.f.script)
        infra = (
            res.get("infra", False)
            or rep.corrupted
            or "in abspath\nFileNotFoundError" in res["out"]
            or ("terminated with signal SIGKILL" in res["out"] and not res["timed_out"])
        )
        return dict(
            rc=res["rc"],
            secs=res["secs"],
            infra=infra,
            timed_out=res["timed_out"],
            title=rep.title,
            titles=rep.titles(),
            kind=rep.kind,
            setup_error=rep.setup_error,
            completed=rep.completed,
            evidence=rep.evidence,
            out=res["out"],
        )

    def _retry(self, run):
        for _ in range(3):  # like pkg/repro getVerdict: infra errors (docker error, OOM-killed rank) are retried
            r = self._parse(run())
            if not r["infra"]:
                break
        return r

    def fresh(self, key):
        return self._retry(lambda: dock.run_fresh(key, self.dir, self.argv_full(), timeout=self.timeout))

    def argv_full(self):
        return [*self.f.launcher, "repro.py", *self.f.args]

    def in_box(self, box):
        return self._retry(lambda: box.run_cmd(self.argv_full(), timeout=self.timeout))


def reproduce(runner, key, tries=3, target=None):
    """First run(s): establish the target title (pkg/repro: the crash title must reproduce)."""
    runs = []
    with dock.Box(key, runner.dir) as box:
        for _ in range(tries):
            r = runner.in_box(box)
            runs.append(r)
            if r["title"] and (target is None or target in r["titles"]):
                return (target or r["title"]), runs
    return None, runs


def ddmin(items, test):
    """Zeller's ddmin + 1-minimal pass (same algorithm as prog.Minimize's call removal)."""
    n = 2
    while len(items) >= 2:
        chunk = max(1, len(items) // n)
        subsets = [items[i : i + chunk] for i in range(0, len(items), chunk)]
        reduced = False
        for i in range(len(subsets)):
            comp = [x for j, s in enumerate(subsets) if j != i for x in s]
            if comp and test(comp):
                items, n, reduced = comp, max(n - 1, 2), True
                break
        if not reduced:
            if n >= len(items):
                break
            n = min(len(items), n * 2)
    i = 0
    while i < len(items) and len(items) > 1:
        cand = items[:i] + items[i + 1 :]
        if test(cand):
            items = cand
        else:
            i += 1
    return items


def minimize(finding, runner, key, target, tries=2):
    """ddmin over program units (dtensor steps / distfuzz calls). Strict title match, like getVerdict(strict)."""
    if finding.rec is None:
        return finding, dict(done=False, why="standalone script: no program to minimize (used verbatim)")
    gen = gen_dtensor if finding.fuzzer == "dtensor" else gen_distfuzz
    base = finding.rec
    all_units = gen.units(base)
    cache, stats = {}, dict(tests=0)

    with dock.Box(key, runner.dir) as box:

        def test(keep):
            k = tuple(keep)
            if k in cache:
                return cache[k]
            cand = gen.with_units(base, set(keep))
            if not gen.is_valid(cand):
                cache[k] = False
                return False
            try:
                src = gen.generate(cand)
            except Exception:
                cache[k] = False
                return False
            runner.set_script(src)
            ok = False
            for _ in range(tries):  # flaky bugs: any of `tries` runs counts (sigs_of in distfuzz)
                stats["tests"] += 1
                r = runner.in_box(box)
                if target in r["titles"]:
                    ok = True
                    break
            cache[k] = ok
            return ok

        keep = ddmin(all_units, test) if len(all_units) > 1 else all_units
    small = gen.with_units(base, set(keep))
    newf = extract.regenerate(finding, small)
    runner.set_script(newf.script)
    return newf, dict(done=True, units_before=len(all_units), units_after=len(keep), tests=stats["tests"])


def classify_rate(k, n):
    if k == n:
        return "deterministic"
    if k == 0:
        return "not reproducible"
    return "flaky"


def reliability(runner, key, target, n=20, parallel=3):
    """N runs, each in a FRESH container. (pkg/repro calculateReliability does <=10 runs, stop at 3 hits;
    we always do all N so the rate is an estimate, not a lower bound.)"""
    runs, excluded = [], 0
    while len(runs) < n and len(runs) + excluded < 3 * n:
        with ThreadPoolExecutor(parallel) as ex:
            batch = list(ex.map(lambda _: runner.fresh(key), range(n - len(runs))))
        runs += [r for r in batch if not r["infra"]]
        excluded += sum(r["infra"] for r in batch)
    n = len(runs)
    k = sum(1 for r in runs if target in r["titles"])
    other = {}
    for r in runs:
        if target not in r["titles"]:
            t = r["title"] or (
                "setup error: " + r["setup_error"]
                if r["setup_error"]
                else "no failure (expected output)"
                if r["completed"]
                else f"no title (rc={r['rc']})"
            )
            other[t] = other.get(t, 0) + 1
    return dict(
        image=key,
        n=n,
        k=k,
        infra_excluded=excluded,
        rate=round(k / n, 3),
        verdict=classify_rate(k, n),
        other_outcomes=other,
        secs=[r["secs"] for r in runs],
        sample_bad=next((r["out"] for r in runs if target in r["titles"]), None),
        sample_other=next((r["out"] for r in runs if target not in r["titles"]), None),
    )


def decide(n, bad, good, infra=0, flaky=False):
    """pkg/bisect bisectionDecision (bisect.go:828-856), verbatim thresholds."""
    want_bad = max(2, (n - infra) // 6)
    want_good = n * 3 // 4 if flaky else n // 2
    want_total = n // 2
    if bad == 0 and good >= want_good:
        return "good"
    if bad >= want_bad and good + bad >= want_total:
        return "bad"
    return "skip"


def test_version(runner, key, target, n):
    counts = dict(bad=0, good=0, other=0, setup=0, infra=0)
    titles, sample = {}, None
    with dock.Box(key, runner.dir) as box:
        for i in range(n):
            r = runner.in_box(box)
            if r["infra"]:
                counts["infra"] += 1
            elif target in r["titles"]:
                counts["bad"] += 1
                sample = sample or r["out"]
            elif r["setup_error"]:
                counts["setup"] += 1
                titles["setup error: " + r["setup_error"]] = titles.get("setup error: " + r["setup_error"], 0) + 1
                sample = sample or r["out"]
                if i == 1 and counts["setup"] == 2:
                    break  # API missing in this version: no point repeating
            elif r["title"]:
                counts["other"] += 1
                titles[r["title"]] = titles.get(r["title"], 0) + 1
                sample = sample or r["out"]
            elif r["completed"]:
                counts["good"] += 1
            else:
                counts["other"] += 1
                t = f"no title (rc={r['rc']})"
                titles[t] = titles.get(t, 0) + 1
                sample = sample or r["out"]
    runs = sum(counts.values())
    if counts["setup"] and not counts["bad"] and not counts["good"]:
        verdict = "untestable"
    else:
        # an unrelated failure is neither good nor bad for *this* bug (bisect treats it as skip)
        verdict = decide(runs, counts["bad"], counts["good"], counts["infra"])
    return dict(version=key, runs=runs, verdict=verdict, **counts, other_titles=titles, sample=(sample or "")[-6000:])


def bisect_versions(runner, target, keys=None, n=4, log=LOG, parallel=2):
    """Test every version in the matrix (a release list is short enough to test exhaustively, which
    also exposes regressions that come back). Returns per-version results plus first-bad/fixed-in."""
    keys = keys or [m["key"] for m in matrix()]
    keys = sorted(keys, key=order)
    with ThreadPoolExecutor(parallel) as ex:
        res = list(ex.map(lambda k: test_version(runner, k, target, n), keys))
    for r in res:
        log(
            f"  [bisect] torch {r['version']:8s} {r['verdict']:10s} bad={r['bad']} good={r['good']} "
            f"other={r['other']} setup={r['setup']}"
        )
    return dict(n_per_version=n, results=res, **summarize(res))


def summarize(res):
    """first-bad: earliest version whose verdict is bad and every testable version after it up to
    the first good is bad. fixed-in: the first good version after the last bad one (if any)."""
    seq = [(r["version"], r["verdict"]) for r in res]
    bad = [v for v, d in seq if d == "bad"]
    tested = [(v, d) for v, d in seq if d in ("bad", "good")]
    if not bad:
        return dict(first_bad=None, fixed_in=None, last_good_before=None, affected=[], note="never reproduced")
    first_bad = bad[0]
    idx = [v for v, _ in tested].index(first_bad)
    last_good_before = next((v for v, d in reversed(tested[:idx]) if d == "good"), None)
    last_bad = bad[-1]
    j = [v for v, _ in tested].index(last_bad)
    fixed_in = next((v for v, d in tested[j + 1 :] if d == "good"), None)
    regress = [v for v, d in tested[idx:j] if d == "good"]
    before = [(v, d) for v, d in seq[: [v for v, _ in seq].index(first_bad)] if d in ("skip", "untestable")]
    note = []
    if last_good_before is None and before:
        skips = [v for v, d in before if d == "skip"]
        missing = [v for v, d in before if d == "untestable"]
        if skips:
            note.append(
                f"{', '.join(skips)} fail with a different error (see table), so the bug cannot be observed there"
            )
        if missing:
            note.append(f"{', '.join(missing)} lack the API")
        note.append(f"{first_bad} is the oldest version where it reproduces")
    elif last_good_before is None:
        note.append("bad on the oldest tested version (introduced earlier)")
    if regress:
        note.append(f"not monotonic: good again on {regress}")
    if fixed_in is None:
        note.append("still reproduces on the newest tested build")
    return dict(
        first_bad=first_bad, last_good_before=last_good_before, fixed_in=fixed_in, affected=bad, note="; ".join(note)
    )
