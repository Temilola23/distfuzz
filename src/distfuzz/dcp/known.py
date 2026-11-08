"""Known bugs of the training profile: a static predictor (config -> known bug ids, before running) and a
post-hoc classifier (signature + finding -> known bug id).

The campaign skips most scenarios the predictor flags (`fuzz --known-rate R` runs a flagged scenario with
probability R, so regressions still show) and counts executed scenarios whose findings are all known as not useful.
Ids follow research/rca-2026-10/VERIFIED.md sections and upstream issue numbers.
"""

FULL_LOADS = ("full", "full_bcast", "full_bcast_split", "model_bcast")
FSDP_LIKE = ("fsdp", "hsdp", "fsdp_tp", "ep", "pp_fsdp")


def _clip(scn):
    c = scn.get("clip")
    if not c:
        return None
    return {"max": c} if isinstance(c, (int, float)) else c


def tp_degree(seg):
    if seg["par"] == "tp":
        return seg["w"]
    if seg["par"] == "fsdp_tp":
        return seg["mesh2d"][1]
    return 1


def tp_shards(scn, seg):
    """(size along the TP shard dim, TP shard dim) of every TP-sharded parameter of the segment's plan."""
    m = scn["model"]
    out = []
    d = m["d0"]
    dims = []  # input width of every block
    for b in m["blocks"]:
        dims.append(d)
        if b["t"] == "lin":
            d = b["out"]
    for name, style in seg.get("tp", {}).items():
        if name == "emb":
            out.append((m["vocab"], 0) if style in ("row", "row_sp") else (m["d0"], 1))
        elif name == "head":
            if style.startswith("col"):
                out.append((m["out"], 0))
            else:
                out.append((d, 1))
        else:
            i = int(name)
            b = m["blocks"][i]
            if b["t"] == "lin":
                out.append((b["out"], 0) if style == "col" else (dims[i], 1))
            elif b["t"] == "mlp":
                out += [(b["h"], 0), (b["h"], 1)]
            elif b["t"] == "attn":
                out += [(dims[i], 0), (dims[i], 1)]
    return out


def uneven_tp(scn, seg):
    k = tp_degree(seg)
    return k > 1 and any(n % k for n, _ in tp_shards(scn, seg))


def strided_full_load(scn, seg):
    """§32: FSDP2 + TP parameter whose TP shard dim is also FSDP's dim 0, divisible by tp but not by dp x tp."""
    if seg["par"] != "fsdp_tp":
        return False
    dp, tp = seg["mesh2d"]
    return any(dim == 0 and n % tp == 0 and n % (dp * tp) for n, dim in tp_shards(scn, seg))


ZB = ("InterleavedZeroBubble", "ZBVZeroBubble", "DualPipeV")


def uneven_fsdp(scn, seg):
    """Some FSDP-sharded parameter whose dim 0 does not divide by the shard degree (padded shards)."""
    from distfuzz.dcp.runtime import build_model

    deg = seg["w"] if seg["par"] == "fsdp" else seg.get("mesh2d", [1, seg["w"]])[1]
    if deg <= 1:
        return False
    return any(p.dim() and p.shape[0] % deg for p in build_model(dict(scn, f8=None), 0).parameters())


def zb_view_boundary(scn, seg):
    """#200199 (fixed after 2.14.0): a split-backward schedule where a stage that is not the last ends in a module
    whose output is a view (nn.Embedding, or a bare nn.Linear on 3-D activations)."""
    from distfuzz.dcp.runtime import pp_split

    pc = seg.get("pp") or {}
    if pc.get("sched") not in ZB:
        return False
    pp = seg.get("ppmesh", [seg["w"], 1])[0]
    blocks = scn["model"]["blocks"]
    for st in pp_split(scn, pp * pc.get("vps", 1))[:-1]:
        u = st[-1]
        if u == "emb" or (u not in ("head",) and blocks[int(u)]["t"] == "lin" and blocks[int(u)]["act"] == "none"):
            return True
    return False


def predict(scn):
    """Known bug ids this scenario is expected to hit, from its config alone."""
    from distfuzz.dcp.fuzz import static_empty

    ids = []
    o = scn["optim"]
    cc = _clip(scn)
    inf = cc is not None and str(cc.get("norm", 2.0)) == "inf"
    for si, seg in enumerate(scn["segs"]):
        par = seg["par"]
        ld = seg.get("load", {}) if si else {}
        if inf and seg["steps"] > 0 and par in FSDP_LIKE:
            try:
                if static_empty(scn, seg):
                    ids.append("K30-inf-clip-empty-shard")
            except Exception:  # noqa: BLE001 (a predictor must never stop the campaign)
                pass
        if ld.get("api") == "sd" and ld.get("mode") == "full_bcast":
            ids.append("K6-joint-bcast-load")
        if ld.get("mode") in FULL_LOADS[:3] and o["name"] == "sgd" and not o.get("momentum"):
            ids.append("K7-stateless-optim-full-load")
        if par not in ("none", "ddp") and o["name"] == "sgd" and o.get("fused"):
            ids.append("K8-fused-sgd-dtensor")
        if par in ("tp", "fsdp_tp") and uneven_tp(scn, seg):
            ids.append("K150336-uneven-tp")
        if ld.get("mode") in FULL_LOADS and strided_full_load(scn, seg):
            ids.append("K32-strided-full-load")
        f8 = scn.get("f8") or {}
        if f8.get("ag") and par in ("fsdp", "hsdp") and uneven_fsdp(scn, seg):
            ids.append("Kao3982-float8-allgather-uneven")
        if par == "pp_fsdp" and seg["steps"] > 0 and (seg.get("pp") or {}).get("sched") in ZB:
            ids.append("N33-zb-fsdp-grads")
        if seg.get("sp") and o["name"] == "sgd" and o.get("momentum") and seg.get("save"):
            ids.append("N34-sp-partial-momentum")
        if scn.get("compile") and par in ("tp", "fsdp_tp") and seg["steps"] > 0 and "emb" in seg.get("tp", {}):
            ids.append("K201700-compile-async-tp-input")
        if par in ("pp", "pp_fsdp") and seg["steps"] > 0 and zb_view_boundary(scn, seg):
            ids.append("K200199-zb-view-detach")
    return sorted(set(ids))


# post-hoc: (predicate over signature, finding, scenario) -> id; mirrors research/train-fuzz-2026-11/tools/triage.py
def _tb(f):
    return f.get("tb") or ""


RULES = [
    (lambda s, f, c: "_fused_sgd_" in s, "K8-fused-sgd-dtensor"),
    (
        lambda s, f, c: "_StridedShard" in _tb(f) or ("_shard_tensor" in _tb(f) and "AssertionError" in s),
        "K32-strided-full-load",
    ),
    (lambda s, f, c: "clip:inf" in s and "empty tensor" in s, "K30-inf-clip-empty-shard"),
    (
        lambda s, f, c: "full_bcast|" in s and "set_state_dict > distributed_c10d.py:broadcast" in s,
        "K6-joint-bcast-load",
    ),
    (lambda s, f, c: "Expected device to be set" in s, "K7-stateless-optim-full-load"),
    (lambda s, f, c: "reduction dim" in s or "remove or reshape sharded" in s, "K150336-uneven-tp"),
    (lambda s, f, c: "in-place operations that require placement" in s, "K9-empty-tp-shard"),
    (lambda s, f, c: "index_select(): self indexing axis dim should be positive" in s, "K15-embedding-empty-row-shard"),
    (lambda s, f, c: "Can't detach views in-place" in _tb(f) or "stage backward input" in s, "K200199-zb-view-detach"),
    (lambda s, f, c: "unhashable type: non-nested SymInt" in _tb(f), "K159635-dtensor-dynamic-symint-hash"),
    # FSDP2 all-gathers in the inputs' dtype when they all match; Gloo has no float8 types (backend limitation)
    (lambda s, f, c: "Invalid scalar type" in s and (c.get("f8") or {}).get("ag"), "Lgloo-float8-allgather"),
    (
        lambda s, f, c: (
            (f.get("kind") == "CRASH" or "Received data size doesn't match" in _tb(f)) and (c.get("f8") or {}).get("ag")
        ),
        "Kao3982-float8-allgather-uneven",
    ),
    (
        lambda s, f, c: (
            f.get("kind") in ("LOSS_MISMATCH", "GNORM_MISMATCH", "STATE_MISMATCH")
            and "pp_fsdp" in s.split("|")[3]
            and any((g.get("pp") or {}).get("sched") in ZB for g in c.get("segs", []))
        ),
        "N33-zb-fsdp-grads",
    ),
    # #201700: the TP embedding's output (redistribute(async_op=True).to_local(), an AsyncCollectiveTensor that
    # requires grad) enters a compiled block; AOT autograd resolves it with trigger_wait() and drops requires_grad,
    # so the gradient never reaches the embedding (zero grad, or a plain Tensor in RedistributeBackward)
    (
        lambda s, f, c: (
            c.get("compile")
            and (
                "has no attribute '_local_tensor'" in s
                or (
                    f.get("kind") in ("LOSS_MISMATCH", "GNORM_MISMATCH", "STATE_MISMATCH")
                    and any(g["par"] in ("tp", "fsdp_tp") and "emb" in g.get("tp", {}) for g in c.get("segs", []))
                )
            )
        ),
        "K201700-compile-async-tp-input",
    ),
    # VERIFIED §34: SGD momentum clones a Partial(sum) SP norm grad, DCP refuses to save the Partial buffer
    (lambda s, f, c: "partial placements" in _tb(f) or "partial placements" in s, "N34-sp-partial-momentum"),
    (lambda s, f, c: "mixed torch.Tensor and DTensor" in s, "K183207-mixed-tensor-dtensor"),
    (lambda s, f, c: "aten.stack" in s and "mesh" in s, "K180346-clip-across-meshes"),
    (lambda s, f, c: "extra.step=-1" in f.get("msg", ""), "K4-flatten-false"),
    (
        lambda s, f, c: "full_tensor > _functional_collectives.py:wait_tensor" in s and "fsdp_tp" in s,
        "K31-fsdp-zero-size-non-shard-dim",
    ),
]


def classify(sig, finding, scn):
    for fn, name in RULES:
        try:
            if fn(sig, finding, scn):
                return name
        except Exception:  # noqa: BLE001
            continue
    return None
