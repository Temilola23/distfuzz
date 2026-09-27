import argparse
import copy
import json
import os
import re

from distfuzz.dtensor.fuzz import signature
from distfuzz.dtensor.gen import NAME_RE, parse_R
from distfuzz.dtensor.world import World

HERE = os.path.dirname(os.path.abspath(__file__))

MK_RE = re.compile(r"\bmk\((\d+)\)")


class Tester:
    def __init__(self, world_size, logdir, timeout):
        self.world = World(world_size, logdir, cov=False)
        self.timeout = timeout
        self.runs = 0

    def sigs(self, prog):
        self.runs += 1
        status, res = self.world.run(prog, self.timeout)
        out = set()
        if status == "timeout":
            out.add("HANG||program timeout")
        elif status == "crash":
            out.add("CRASH")
        if res and status in ("ok", "aborted"):
            for r in res:
                for f in r.get("findings", []):
                    out.add(signature(f))
            hs = [tuple(r.get("hashes", [])) for r in res]
            if status == "ok" and len(set(hs)) > 1:
                out.add("RANK_HASH_DIVERGENT||full_tensor differs across ranks")
        return out

    def close(self):
        self.world.close()


def well_formed(steps):
    defined = set()
    for st in steps:
        if set(NAME_RE.findall(st["expr"])) - defined:
            return False
        defined.add(st["out"])
    return True


def sig_match(target, sigs):
    if target.startswith("CRASH"):
        return "CRASH" in sigs
    if target.startswith("RANK_DIVERGENT_ERROR"):
        k = target.split("|")[1]
        return any(s.startswith("RANK_DIVERGENT_ERROR|" + k) for s in sigs)
    return target in sigs


def ddmin(items, test):
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


def minimize(prog, target, tester, world_size):
    prog = copy.deepcopy(prog)
    prog.pop("mut", None)

    def with_steps(steps):
        p = copy.deepcopy(prog)
        p["steps"] = copy.deepcopy(steps)
        return p

    flaky = target.split("|")[0] in ("HANG", "CRASH", "RANK_DIVERGENT_ERROR", "FATAL", "RANK_HASH_DIVERGENT")

    def t_steps(steps):
        return well_formed(steps) and all(
            sig_match(target, tester.sigs(with_steps(steps))) for _ in range(2 if flaky else 1)
        )

    if not t_steps(prog["steps"]):
        return None, "not reproducible"
    prog["steps"] = ddmin(prog["steps"], t_steps)
    for spec in prog["inputs"]:
        for d, old in enumerate(spec["pl"]):
            if old != "R":
                spec["pl"][d] = "R"
                if not sig_match(target, tester.sigs(prog)):
                    spec["pl"][d] = old
        if spec.get("rg"):
            spec["rg"] = False
            if not sig_match(target, tester.sigs(prog)):
                spec["rg"] = True
    for st in prog["steps"]:
        pr = parse_R(st["expr"])
        if pr:
            src, pls = pr
            for d, old in enumerate(pls):
                if old != "R":
                    pls[d] = "R"
                    oldexpr, st["expr"] = st["expr"], f"R({src}, {pls!r})"
                    if not sig_match(target, tester.sigs(prog)):
                        st["expr"], pls[d] = oldexpr, old
    used = sorted({int(i) for st in prog["steps"] for i in MK_RE.findall(st["expr"])})
    remap = {o: n for n, o in enumerate(used)}
    prog["inputs"] = [prog["inputs"][o] for o in used]
    for st in prog["steps"]:
        st["expr"] = MK_RE.sub(lambda m: f"mk({remap[int(m.group(1))]})", st["expr"])
    ok = sig_match(target, tester.sigs(prog)) and well_formed(prog["steps"])
    return prog, ("ok" if ok else "renumber-broke")


def emit_repro(prog, world, path):
    rt = open(os.path.join(HERE, "runtime.py")).read()
    body = "\n".join(f"    {st['out']} = {st['expr']}" for st in prog["steps"])
    mesh_shape = {"1d": f"({world},)", "2d": "(2, 2)" if world == 4 else "(2, 3)"}[prog["mesh"]]
    src = f"""#!/usr/bin/env python3
import datetime
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.device_mesh import init_device_mesh

{rt}

INPUTS = {prog["inputs"]!r}


def program(mk, R, BW, SETITEM, CMP, torch=torch, F=F):
{body}
    return {{k: v for k, v in locals().items() if k.startswith("v")}}


def run_path(ctx):
    env = make_env(ctx)
    try:
        return program(env["mk"], env["R"], env["BW"], env["SETITEM"], env["CMP"]), None
    except Exception as e:
        return None, e


def main(rank, world, port):
    dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{{port}}", rank=rank,
                            world_size=world, timeout=datetime.timedelta(seconds=30))
    mesh = init_device_mesh("cpu", {mesh_shape})
    torch.manual_seed(0)
    ref, rerr = run_path(Ctx("ref", None, INPUTS))
    torch.manual_seed(0)
    got, derr = run_path(Ctx("dist", mesh, INPUTS))
    if rerr or derr:
        print(f"[rank {{rank}}] ref error: {{rerr!r}}  dtensor error: {{derr!r}}", flush=True)
    if ref is not None and got is not None:
        for k in ref:
            r, d = ref[k], got.get(k)
            if not isinstance(r, torch.Tensor):
                continue
            full = d.full_tensor() if isinstance(d, DTensor) else d
            diff = compare(r, full)
            if rank == 0:
                pl = getattr(d, "placements", None)
                print(f"{{k}}: {{'OK' if diff is None else 'MISMATCH ' + diff}}  placements={{pl}}")
                if diff is not None:
                    print("   ref :", r.detach().flatten()[:12].tolist())
                    print("   dt  :", full.detach().flatten()[:12].tolist())
            if isinstance(d, DTensor):
                import torch.distributed.tensor._utils as U
                exp, _ = U.compute_local_shape_and_global_offset(d.shape, d.device_mesh, d.placements)
                local = tuple(d.to_local().shape)
                if tuple(exp) != local:
                    print(f"[rank {{rank}}] {{k}}: LOCAL SHAPE {{local}} != expected {{tuple(exp)}}", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    import socket
    s = socket.socket(); s.bind(("127.0.0.1", 0)); port = s.getsockname()[1]; s.close()
    mp.spawn(main, args=({world}, port), nprocs={world}, join=True)
"""
    with open(path, "w") as f:
        f.write(src)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("findings")
    ap.add_argument("--out", default=os.path.join("runs", "dtensor-repros"))
    ap.add_argument("--sig-substr", action="append", default=[])
    ap.add_argument("--index", type=int, action="append", default=[])
    ap.add_argument("--max", type=int, default=50)
    ap.add_argument("--timeout", type=float, default=30)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rows = [json.loads(line) for line in open(a.findings)]
    sel = []
    for i, r in enumerate(rows):
        if (a.index and i in a.index) or (a.sig_substr and any(s in r["sig"] for s in a.sig_substr)):
            sel.append((i, r))
    sel = sel[: a.max]
    testers = {}
    results = []
    for i, r in sel:
        w = r.get("world", 4)
        if w not in testers:
            testers[w] = Tester(w, os.path.join(a.out, f"logs_w{w}"), a.timeout)
        t = testers[w]
        before = t.runs
        mp, status = minimize(r["prog"], r["sig"], t, w)
        rec = dict(index=i, sig=r["sig"], status=status, runs=t.runs - before)
        if mp:
            name = re.sub(r"[^A-Za-z0-9]+", "_", r["sig"])[:60] + f"_{i}"
            path = os.path.join(a.out, name + ".py")
            emit_repro(mp, w, path)
            json.dump(dict(sig=r["sig"], prog=mp, world=w), open(os.path.join(a.out, name + ".json"), "w"), indent=1)
            rec.update(
                path=path,
                steps=[s["expr"] for s in mp["steps"]],
                mesh=mp["mesh"],
                inputs=mp["inputs"],
                orig_steps=len(r["prog"]["steps"]),
            )
        print(json.dumps(rec), flush=True)
        results.append(rec)
    for t in testers.values():
        t.close()
    with open(os.path.join(a.out, "minimized.jsonl"), "a") as f:
        for rec in results:
            f.write(json.dumps(rec) + "\n")


if __name__ == "__main__":
    main()
