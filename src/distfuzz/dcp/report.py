import json


def bucket(r):
    f, s = r["finding"], r["scn"]
    k, sig = f["kind"], r["sig"]
    i = f.get("seg", 0)
    prev = s["segs"][i - 1] if i else {}
    flat = prev.get("save", {}).get("planner", {}).get("flat", True)
    if k == "FATAL" and s["segs"][i].get("load", {}).get("mode") == "full_bcast":
        return "A4 set_state_dict(broadcast_from_rank0) rank-divergent collectives"
    if k in ("CRASH", "FATAL"):
        return "env/cascade: OOM kill (-9) or gloo timeout after a peer died"
    if k in ("LOAD_MISMATCH", "LOAD_EXTRA", "ROUNDTRIP_MISMATCH") or "read_ckpt" in sig:
        if not flat:
            return "A1 flatten_state_dict=False drops non-tensor/nested values on load"
        if prev.get("steps") == 0:
            return "B1 get_state_dict phantom optimizer step (#164929)"
    if k in ("STATE_MISMATCH", "LOSS_MISMATCH"):
        return "E fuzzer bug: mixed-precision segment earlier in chain (fixed mid-run)"
    if "Expected device" in sig:
        return "A3 full_state_dict load into stateless optimizer"
    if "param_groups" in sig or "'state'" in sig:
        return "A2 flatten_optimizer_state_dict + full_state_dict"
    if "Error(s) in loading" in sig or "Missing key" in sig:
        return "C ignore_frozen_params round-trip needs strict=False"
    if "_fused_sgd_.default does not have" in sig:
        return "A5 fused SGD on DTensor"
    if "mixed torch.Tensor and DTensor" in sig and "read_ckpt" not in sig:
        return "B foreach/fused optimizer or clip over mixed Tensor/DTensor params (#183207 class)"
    if "aten.stack" in sig:
        return "B clip_grad_norm_ across meshes (#180346)"
    if "reduction dim" in sig or "remove or reshape sharded" in sig:
        return "B TP uneven hidden dim (#150336)"
    if "in-place operations that require placement" in sig:
        return "A6 TP empty shard -> grad placement Replicate"
    if k == "RANK_DIVERGENT_ERR":
        return "C async_save thread on default PG racing main-thread collectives (#123447)"
    return "?"


def main(argv):
    run_dir = argv[0]
    rows = [json.loads(line) for line in open(run_dir + "/findings.jsonl")]
    S = json.load(open(run_dir + "/summary.json"))["sigs"]
    agg = {}
    for r in rows:
        b = bucket(r)
        a = agg.setdefault(b, [0, 0])
        a[0] += 1
        a[1] += S.get(r["sig"], 0)
    for b, (n, hits) in sorted(agg.items()):
        print(f"{n:3d} sigs {hits:4d} hits  {b}")
    for r in rows:
        if bucket(r) == "?":
            print("UNBUCKETED", r["sig"], r["finding"].get("msg", "")[:200])
