import datetime
import json
import os
import shutil
import socket
import sys
import tempfile
import traceback

# minimize.emit disables this import: emitted repros define run_segment inline.
if "run_segment" not in globals():
    from distfuzz.dcp.runtime import run_segment


def _free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def _rank_main(rank, W, port, scn, si, workdir, q):
    import torch
    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh

    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=W, timeout=datetime.timedelta(seconds=30)
    )
    meshes = {"1d": init_device_mesh("cpu", (W,))}
    for a in range(1, W + 1):
        if W % a == 0:
            meshes[f"2d:{a},{W // a}"] = init_device_mesh("cpu", (a, W // a), mesh_dim_names=("dp", "tp"))
    rt = dict(rank=rank, world=W, meshes=meshes, ckpt_pg=dist.new_group(backend="gloo"))
    try:
        res = run_segment(rt, scn, si, workdir)  # noqa: F821 (provided by runtime)
    except Exception:
        res = dict(findings=[dict(kind="FATAL", msg=traceback.format_exc()[-1500:])], hashes=[])
    q.put((rank, res))
    try:
        dist.destroy_process_group()
    except Exception:
        pass


def run_standalone(scn, timeout=180, verbose=True):
    import torch.multiprocessing as mp

    ctx = mp.get_context("spawn")
    workdir = tempfile.mkdtemp(prefix="dcpfz_", dir=os.environ.get("CK_ROOT") or None)
    allf = []
    try:
        for si, seg in enumerate(scn["segs"]):
            W = seg["w"]
            q = ctx.Queue()
            port = _free_port()
            procs = [ctx.Process(target=_rank_main, args=(r, W, port, scn, si, workdir, q)) for r in range(W)]
            for p in procs:
                p.start()
            res = {}
            import queue
            import time

            deadline = time.time() + timeout
            while len(res) < W:
                try:
                    r, v = q.get(timeout=1)
                    res[r] = v
                    continue
                except queue.Empty:
                    pass
                dead = [p.exitcode for p in procs if p.exitcode not in (None, 0)]
                if dead:
                    allf.append(dict(kind="CRASH", seg=si, msg=f"exitcodes {dead}"))
                    break
                if time.time() > deadline:
                    allf.append(dict(kind="HANG", seg=si, msg="segment timeout"))
                    break
            for p in procs:
                p.join(5)
                if p.is_alive():
                    p.kill()
            fs = []
            for r in sorted(res):
                for f in res[r]["findings"]:
                    if (f["kind"], f.get("msg")) not in {(g["kind"], g.get("msg")) for g in fs}:
                        fs.append(dict(f, rank=r))
            hs = {tuple(v["hashes"]) for v in res.values()}
            if not fs and len(hs) > 1:
                fs.append(dict(kind="RANK_HASH_DIVERGENT", seg=si, msg=str(hs)))
            if verbose:
                print(
                    f"segment {si}: w={W} par={seg['par']} steps={seg['steps']} -> "
                    f"{[(f['kind'], f.get('phase'), f.get('msg', '')[:300]) for f in fs] or 'ok'}",
                    flush=True,
                )
                for f in fs:
                    if f.get("tb") and os.environ.get("TB"):
                        print(f["tb"])
            allf.extend(fs)
            if fs or len(res) < W:
                break
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
    return allf


def main(argv):
    with open(argv[0]) as f:
        scn = json.load(f)
    sys.exit(1 if run_standalone(scn) else 0)
