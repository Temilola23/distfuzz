# Everything runs under the __main__ guard: async_save(PROCESS) spawns children that re-import this module.
import argparse
import datetime
import faulthandler
import json
import os
import signal
import sys
import time
import traceback
import warnings

TARGETS = [
    os.sep + os.path.join("torch", "distributed", d)
    for d in ("checkpoint" + os.sep, "fsdp" + os.sep, "tensor" + os.sep, "_composable" + os.sep, "_state_dict_utils.py")
]
TARGETS.append(os.sep + os.path.join("torch", "nn", "parallel") + os.sep)


def main():
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser()
    for a in ("--rank", "--world", "--port", "--cov"):
        ap.add_argument(a, type=int, default=1)
    args = ap.parse_args()
    faulthandler.register(signal.SIGUSR1, all_threads=True)
    proto = os.fdopen(os.dup(1), "w", buffering=1)
    os.dup2(2, 1)

    import torch
    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh

    from distfuzz.dcp.runtime import run_segment

    torch.set_num_threads(1)

    cov_new = set()
    if args.cov:
        mon = sys.monitoring
        mon.use_tool_id(4, "dcpfuzz")

        def line(code, ln):
            fn = code.co_filename
            if any(t in fn for t in TARGETS):
                cov_new.add(fn[fn.find(os.sep + "torch" + os.sep) + 1 :] + ":" + str(ln))
            return mon.DISABLE

        mon.register_callback(4, mon.events.LINE, line)
        mon.set_events(4, mon.events.LINE)

    W = args.world
    dist.init_process_group(
        "gloo",
        init_method=f"tcp://127.0.0.1:{args.port}",
        rank=args.rank,
        world_size=W,
        timeout=datetime.timedelta(seconds=30),
    )
    meshes = {"1d": init_device_mesh("cpu", (W,))}
    for a in range(1, W + 1):
        if W % a == 0:
            meshes[f"2d:{a},{W // a}"] = init_device_mesh("cpu", (a, W // a), mesh_dim_names=("dp", "tp"))
    rt = dict(rank=args.rank, world=W, meshes=meshes, ckpt_pg=dist.new_group(backend="gloo"))

    proto.write(json.dumps({"ready": True, "rank": args.rank}) + "\n")
    for ln in sys.stdin:
        cmd = json.loads(ln)
        if cmd.get("cmd") == "exit":
            break
        t0, fatal = time.time(), None
        try:
            res = run_segment(rt, cmd["scn"], cmd["si"], cmd["dir"])
        except Exception:
            res, fatal = dict(findings=[], hashes=[]), traceback.format_exc()[-2000:]
        aborted = fatal is not None or any(
            f["kind"] == "RANK_DIVERGENT_ERR" or (f["kind"] == "ERR" and f["phase"] not in ("build", "read_ckpt"))
            for f in res["findings"]
        )
        proto.write(
            json.dumps(
                dict(rank=args.rank, aborted=aborted, fatal=fatal, cov=sorted(cov_new), dt=time.time() - t0, **res)
            )
            + "\n"
        )
        cov_new.clear()
        if aborted:
            break
    os._exit(0)


if __name__ == "__main__":
    main()
