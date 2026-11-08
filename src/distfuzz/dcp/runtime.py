# Embedded verbatim into emitted repros: imports must stay stdlib + torch only.
import hashlib
import math
import os
import re
import sys
import traceback

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

# minimize.emit inlines blocks.py ahead of this file, so emitted repros skip this import
if "MoE" not in globals():
    from distfuzz.dcp.blocks import (
        MOE_STATS,
        LoRAMLP,
        MoE,
        TitanModel,
        TitanOptim,
        apply_ac,
        apply_ep,
        clip_grads,
        expert_params,
        has_empty_local,
        replicate_leftovers,
    )

DT = {"f64": torch.float64, "f32": torch.float32, "bf16": torch.bfloat16, "f16": torch.float16}
# device of the distributed model ("cuda" under NCCL; the single-process reference always runs on CPU). The harness's
# own control collectives (error flags, loss sums) run on the default group with tensors on that device; "ctl" can
# name a separate group for them. Under NCCL no Gloo group is created (Gloo segfaulted at init on Kaggle's image).
DEV = {"device": "cpu", "ctl": None}


def ctl_t(t):
    return t.to(dev()) if DEV["device"] != "cpu" and DEV["ctl"] is None else t


def dev():
    if DEV["device"] == "cuda":
        return torch.device("cuda", torch.cuda.current_device())
    return torch.device("cpu")


class Lin(nn.Module):
    def __init__(self, din, dout, bias, act):
        super().__init__()
        self.lin = nn.Linear(din, dout, bias=bias)
        self.act = act

    def forward(self, x):
        return _act(self.lin(x), self.act)


class MLP(nn.Module):
    def __init__(self, d, h, bias, act):
        super().__init__()
        self.fc1 = nn.Linear(d, h, bias=bias)
        self.fc2 = nn.Linear(h, d, bias=bias)
        self.act = act

    def forward(self, x):
        return x + self.fc2(_act(self.fc1(x), self.act))


class Attn(nn.Module):
    def __init__(self, d, nh, bias):
        super().__init__()
        self.hd = d // nh
        self.q = nn.Linear(d, d, bias=bias)
        self.k = nn.Linear(d, d, bias=False)  # k bias has an analytically zero grad; Adam amplifies its noise
        self.v = nn.Linear(d, d, bias=bias)
        self.o = nn.Linear(d, d, bias=bias)

    def forward(self, x):
        q, k, v = self.q(x), self.k(x), self.v(x)
        B, L = q.shape[:2]  # under sequence parallelism x is a sequence shard and q the gathered sequence
        q = q.reshape(B, L, -1, self.hd).transpose(1, 2)
        k = k.reshape(B, L, -1, self.hd).transpose(1, 2)
        v = v.reshape(B, L, -1, self.hd).transpose(1, 2)
        a = torch.softmax(q @ k.transpose(-1, -2) / math.sqrt(self.hd), dim=-1)
        y = (a @ v).transpose(1, 2).reshape(B, L, -1)
        return x + self.o(y)


class LN(nn.Module):
    def __init__(self, d, affine, bias, rms=False):
        super().__init__()
        if rms:  # torchtitan / HF llama norms
            self.ln = nn.RMSNorm(d, eps=1e-6, elementwise_affine=affine)
        else:
            self.ln = nn.LayerNorm(d, elementwise_affine=affine, bias=bias)

    def forward(self, x):
        return self.ln(x)


class Buf(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.register_buffer("count", torch.zeros(()))
        self.register_buffer("scale", torch.arange(d) * 0.01, persistent=False)

    def forward(self, x):
        if self.training:
            with torch.no_grad():
                self.count.add_(1)
        return x * (1 + 0.01 * self.count.to(x.dtype)) + self.scale.to(x.dtype)


def _act(x, a):
    if a == "relu":
        return F.relu(x)
    if a == "tanh":
        return torch.tanh(x)
    if a == "gelu":
        return F.gelu(x)
    return x


class Net(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.kind = m["kind"]
        d = m["d0"]
        if self.kind == "tok":
            self.emb = nn.Embedding(m["vocab"], d)
        blocks = []
        for b in m["blocks"]:
            t = b["t"]
            if t == "lin":
                blocks.append(Lin(d, b["out"], b["bias"], b["act"]))
                d = b["out"]
            elif t == "mlp":
                blocks.append(MLP(d, b["h"], b["bias"], b["act"]))
            elif t == "attn":
                blocks.append(Attn(d, b["nh"], b["bias"]))
            elif t == "ln":
                blocks.append(LN(d, b["affine"], b["bias"], b.get("rms", False)))
            elif t == "lora":
                blocks.append(LoRAMLP(d, b["h"], b["r"], b["alpha"], b["bias"]))
            elif t == "moe":
                blocks.append(MoE(d, b))
            elif t == "buf":
                blocks.append(Buf(d))
        self.blocks = nn.ModuleList(blocks)
        self.head = nn.Linear(d, m["out"], bias=m["head_bias"])
        if m.get("tie"):
            self.head.weight = self.emb.weight
        for i in m.get("frozen", []):
            for p in self.blocks[i].parameters():
                p.requires_grad_(False)
        if m.get("lora_only"):  # torchtune LoRA recipes train the adapters only
            for n, p in self.named_parameters():
                if "lora_" not in n:
                    p.requires_grad_(False)

    def forward(self, x):
        h = self.emb(x) if self.kind == "tok" else x
        for b in self.blocks:
            h = b(h)
        return self.head(h)


def model_dims_ok(m):
    d = m["d0"]
    for b in m["blocks"]:
        if b["t"] == "lin":
            d = b["out"]
        elif b["t"] == "attn" and d % b["nh"]:
            return False
    if m.get("tie") and (m["kind"] != "tok" or d != m["d0"] or m["out"] != m["vocab"]):
        return False
    if any(i >= len(m["blocks"]) for i in m.get("frozen", [])):
        return False
    return True


def build_model(scn, seed):
    torch.manual_seed(seed)
    net = Net(scn["model"]).to(DT[scn["dtype"]])
    if scn.get("f8"):
        apply_float8(net, scn["f8"])
    return net


def apply_float8(net, f8):
    """torchao float8 training as accelerate applies it (accelerate@f47439c utils/ao.py:49-143,
    accelerator.py:2040-2058): Float8Linear for nn.Linear layers with both dims divisible by 16 except the first and
    last linear, pad_inner_dim, optional float8 all-gather under FSDP2."""
    from torchao.float8 import Float8LinearConfig, convert_to_float8_training

    lins = [n for n, m in net.named_modules() if isinstance(m, nn.Linear)]
    edge = {lins[0], lins[-1]} if lins else set()

    def keep(m, fqn):
        return isinstance(m, nn.Linear) and fqn not in edge and m.in_features % 16 == 0 and m.out_features % 16 == 0

    cfg = Float8LinearConfig(pad_inner_dim=True, enable_fsdp_float8_all_gather=bool(f8.get("ag")))
    convert_to_float8_training(net, module_filter_fn=keep, config=cfg)


def batch(scn, step):
    m = scn["model"]
    B, L = scn["batch"], m["seq"]
    g = torch.Generator().manual_seed(scn["seed"] * 1009 + step)
    dt = DT[scn["dtype"]]
    if m["kind"] == "tok":
        x = torch.randint(0, m["vocab"], (B, L), generator=g)
        y = torch.randint(0, m["out"], (B, L), generator=g)
    else:
        x = torch.randn(B, L, m["d0"], generator=g).to(dt)
        y = torch.randn(B, L, m["out"], generator=g).to(dt)
    return x, y


def loss_fn(scn, out, y):
    out = out.to(DT[scn["dtype"]])
    if scn["model"]["kind"] == "tok":
        return F.cross_entropy(out.reshape(-1, out.shape[-1]), y.reshape(-1))
    return F.mse_loss(out, y)


def build_optim(scn, params):
    o = scn["optim"]
    kw = {}
    if o.get("foreach") is not None:
        kw["foreach"] = o["foreach"]
    if o.get("fused"):
        kw["fused"] = True
        kw.pop("foreach", None)
    if o["name"] == "sgd":
        return torch.optim.SGD(
            params, lr=o["lr"], momentum=o["momentum"], nesterov=o["nesterov"], weight_decay=o["wd"], **kw
        )
    cls = torch.optim.Adam if o["name"] == "adam" else torch.optim.AdamW
    betas = tuple(o.get("betas", (0.9, 0.95)))
    return cls(params, lr=o["lr"], betas=betas, weight_decay=o["wd"], amsgrad=o["amsgrad"], **kw)


def dp_info(seg, rank, meshes):
    par = seg["par"]
    if par in ("none", "tp"):
        return 1, 0
    if par == "fsdp_tp":
        m = meshes[mesh_key(seg)]
        return m.size(0), m.get_coordinate()[0]
    return seg["w"], rank  # ddp / fsdp / hsdp


def mesh_key(seg):
    if seg["par"] in ("fsdp_tp", "hsdp"):
        return "2d:{},{}".format(*seg["mesh2d"])
    return "1d"


def pp_key(seg):
    """(pp, dp) mesh of a pipeline segment; pure PP is pp = world, dp = 1."""
    return "pp:{},{}".format(*seg.get("ppmesh", [seg["w"], 1]))


# pipeline parallelism -------------------------------------------------------------------------------------------
class PPStage(nn.Module):
    """One pipeline model chunk: the units of Net it owns, under the same FQNs (blocks keyed by their index, as
    torchtitan's _split_module keeps a ModuleDict of layers), so checkpoints reshard between PP and non-PP."""

    def __init__(self, net, units):
        super().__init__()
        self.kind = net.kind
        self.first = "emb" in units or units[0] == "0"
        if "emb" in units:
            self.emb = net.emb
        self.blocks = nn.ModuleDict({u: net.blocks[int(u)] for u in units if u not in ("emb", "head")})
        if "head" in units:
            self.head = net.head

    def forward(self, x):
        h = self.emb(x) if hasattr(self, "emb") else x
        for b in self.blocks.values():
            h = b(h)
        return self.head(h) if hasattr(self, "head") else h


def pp_units(scn):
    m = scn["model"]
    return (["emb"] if m["kind"] == "tok" else []) + [str(i) for i in range(len(m["blocks"]))] + ["head"]


def pp_split(scn, nstages):
    """Contiguous, balanced split of emb / blocks / head over the stages (torchtitan
    _generate_llm_fqn_per_model_part: embedding on the first stage, output on the last)."""
    u = pp_units(scn)
    q, r = divmod(len(u), nstages)
    out, i = [], 0
    for s in range(nstages):
        n = q + (1 if s < r else 0)
        out.append(u[i : i + n])
        i += n
    return out


def pp_sched_class(name):
    from torch.distributed.pipelining.schedules import get_schedule_class

    return get_schedule_class(name)


def pp_stage_ids(pp_rank, pp, sched, nstages):
    # torchtitan _get_pp_rank_to_stage_indices_mapping: "v" for ZBV / DualPipeV, else "loop"
    vps = nstages // pp
    if sched in ("ZBVZeroBubble", "DualPipeV"):
        return (pp_rank, nstages - 1 - pp_rank)
    return tuple(pp_rank + s * pp for s in range(vps))


def build_pp(scn, seg, meshes, rank, poison=None):
    """Returns (parts, opts, schedule, stage ids, pp group, dp mesh or None)."""
    from torch.distributed.pipelining import PipelineStage
    from torch.distributed.pipelining.schedules import PipelineScheduleMulti

    pc = seg["pp"]
    pm = meshes[pp_key(seg)]
    pp, dp = seg.get("ppmesh", [seg["w"], 1])
    pp_rank = pm.get_coordinate()[0]
    nstages = pp * pc.get("vps", 1)
    split = pp_split(scn, nstages)
    ids = pp_stage_ids(pp_rank, pp, pc["sched"], nstages)
    group = pm.get_group("pp")
    parts, stages = [], []
    for sid in ids:
        net = build_model(scn, scn["seed"])
        if poison:
            poison(net)
        net = net.to(dev())
        part = PPStage(net, split[sid])
        if dp > 1:
            part = parallelize(scn, dict(seg, par="pp_fsdp"), part, meshes)
        else:
            apply_ac(part, scn.get("ac"))
            part = apply_compile(scn, part, "pre")
            part = apply_compile(scn, part, "post")
        parts.append(part)
        stages.append(PipelineStage(part, sid, nstages, dev(), group=group))
    opts = [build_optim(scn, named_params(scn, p)) for p in parts]
    cls = pp_sched_class(pc["sched"])
    n = len(micro_batches(scn, 0, dp, 0)[0][0]) // pc["mb"]
    assert n > 0
    titan = pc.get("titan_loss", False)
    gdf = dp if (seg.get("fsdp_knobs") or {}).get("gdf1") else 1  # FSDP sums over dp: the loss carries the 1/dp

    def lf(out, y):
        loss = loss_fn(scn, out, y) / gdf
        # torchtitan: scale_grads=False and the loss divided by the global token count (here: per-microbatch mean
        # / n_microbatches); torch default: scale_grads=True with per-microbatch means
        return loss / pc["mb"] if titan else loss

    if issubclass(cls, PipelineScheduleMulti):
        sched = cls(stages, n_microbatches=pc["mb"], loss_fn=lf, scale_grads=not titan)
    else:
        sched = cls(stages[0], n_microbatches=pc["mb"], loss_fn=lf, scale_grads=not titan)
    return parts, opts, sched, ids, nstages, pm


def tp_plan(scn, seg):
    from torch.distributed.tensor import Replicate, Shard
    from torch.distributed.tensor.parallel import ColwiseParallel, RowwiseParallel, SequenceParallel

    # sequence parallelism (torchtitan decoder_sharding / apply_tp with enable_sp): the residual stream between
    # blocks is a local Shard(1) (sequence) tensor; norms run SequenceParallel, colwise inputs all-gather the
    # sequence, rowwise outputs reduce-scatter back to Shard(1)
    sp = bool(seg.get("sp"))
    S1 = Shard(1)
    plan = {}
    for name, style in seg.get("tp", {}).items():
        if name == "emb":
            if style in ("row", "row_sp"):
                plan["emb"] = RowwiseParallel(input_layouts=Replicate(), output_layouts=S1 if sp else Replicate())
            else:
                plan["emb"] = ColwiseParallel(output_layouts=S1 if sp else Replicate())
        elif name == "head":
            kw = dict(input_layouts=S1) if sp else {}
            if style == "col_lp":  # torchtitan lm_head under loss_parallel: vocab-sharded logits stay DTensors
                plan["head"] = ColwiseParallel(output_layouts=Shard(-1), use_local_output=False, **kw)
            elif style == "col":
                plan["head"] = ColwiseParallel(output_layouts=Replicate(), **kw)
            else:
                plan["head"] = RowwiseParallel(input_layouts=Replicate())
        else:
            i = int(name)
            t = scn["model"]["blocks"][i]["t"]
            p = f"blocks.{i}"
            if t == "ln" and style == "sp":
                plan[p + ".ln"] = SequenceParallel(use_local_output=True)
            elif t == "lin":
                if sp:
                    plan[p + ".lin"] = (
                        ColwiseParallel(input_layouts=S1, output_layouts=S1)
                        if style == "col"
                        else RowwiseParallel(input_layouts=S1, output_layouts=S1)
                    )
                else:
                    plan[p + ".lin"] = (
                        ColwiseParallel(output_layouts=Replicate())
                        if style == "col"
                        else RowwiseParallel(input_layouts=Replicate())
                    )
            elif t == "mlp":
                plan[p + ".fc1"] = ColwiseParallel(input_layouts=S1) if sp else ColwiseParallel()
                plan[p + ".fc2"] = RowwiseParallel(output_layouts=S1) if sp else RowwiseParallel()
            elif t == "attn":
                for q in ("q", "k", "v"):
                    plan[f"{p}.{q}"] = ColwiseParallel(input_layouts=S1) if sp else ColwiseParallel()
                plan[p + ".o"] = RowwiseParallel(output_layouts=S1) if sp else RowwiseParallel()
    return plan


def apply_compile(scn, model, when):
    """torch.compile per block or for the whole model. "pre" (before FSDP, after TP and AC): wrap each block in an
    OptimizedModule (accelerate compile_regions, older torchtitan apply_compile); "post" (after FSDP and AC):
    in-place Module.compile() per block (accelerate compile_regions_fsdp2), or torch.compile of the whole model."""
    c = scn.get("compile")
    if not c:
        return model
    from torch import _dynamo

    _dynamo.config.capture_scalar_outputs = True  # accelerate sets it for regional compile (MoE counts)
    kw = dict(backend=c["backend"], fullgraph=bool(c.get("fullgraph")))
    how = c["how"]
    base = model.module if hasattr(model, "module") and not hasattr(model, "blocks") else model  # DDP wrapper
    blocks = base.blocks
    keys = list(blocks.keys()) if isinstance(blocks, nn.ModuleDict) else range(len(blocks))
    if when == "pre" and how == "wrap":
        for k in keys:
            blocks[k] = torch.compile(blocks[k], **kw)
    elif when == "post" and how == "inplace":
        for k in keys:
            blocks[k].compile(**kw)
    elif when == "post" and how == "whole":
        return torch.compile(model, **kw)
    return model


def fsdp_unit(model, u):
    if u == "emb":
        return model.emb
    if u == "head":
        return model.head
    return model.blocks[u] if isinstance(model.blocks, nn.ModuleDict) else model.blocks[int(u)]


def fsdp_knobs(seg, model, units):
    """FSDPModule settings production stacks use: torchtitan disables FSDP's gradient division (loss is scaled by
    the global token count instead) and skips the embedding's backward unshard; Megatron-Core sets an explicit
    backward prefetch chain when activation recomputation is on; torchtitan prefetches the next block in forward."""
    kn = seg.get("fsdp_knobs") or {}
    if not kn:
        return
    from torch.distributed.fsdp import FSDPModule

    mods = [m for m in units if isinstance(m, FSDPModule)]
    if kn.get("gdf1"):
        for m in list(mods) + ([model] if isinstance(model, FSDPModule) else []):
            m.set_gradient_divide_factor(1.0)
            if DEV["device"] == "cpu":
                # a custom factor makes FSDP2 reduce-scatter with NCCL's PREMUL_SUM for f32/bf16 (documented in
                # set_gradient_divide_factor), which Gloo rejects; sum-only reductions are the supported route
                m.set_force_sum_reduction_for_comms(True)
    if kn.get("emb_no_bwd_unshard") and hasattr(model, "emb") and isinstance(model.emb, FSDPModule):
        model.emb.set_unshard_in_backward(False)
    if kn.get("bwd_prefetch"):
        for a, b in zip(mods[1:], mods[:-1]):
            a.set_modules_to_backward_prefetch([b])
    if kn.get("fwd_prefetch"):
        for a, b in zip(mods[:-1], mods[1:]):
            a.set_modules_to_forward_prefetch([b])


def parallelize(scn, seg, model, meshes):
    from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
    from torch.distributed.tensor.parallel import parallelize_module

    par = seg["par"]
    if par == "none":
        apply_ac(model, scn.get("ac"))
        model = apply_compile(scn, model, "pre")
        return apply_compile(scn, model, "post")
    if par == "ddp":
        from torch.nn.parallel import DistributedDataParallel as DDP

        apply_ac(model, scn.get("ac"))
        model = apply_compile(scn, model, "pre")
        model = DDP(
            model, broadcast_buffers=seg.get("ddp_bcast", True), gradient_as_bucket_view=seg.get("ddp_gabv", False)
        )
        return apply_compile(scn, model, "post")
    if par == "tp":
        parallelize_module(model, meshes["1d"], tp_plan(scn, seg))
        if scn.get("profile") == "train":
            replicate_leftovers(model, meshes["1d"])
        apply_ac(model, scn.get("ac"))
        model = apply_compile(scn, model, "pre")
        return apply_compile(scn, model, "post")
    mp = seg.get("mp")
    mpp = MixedPrecisionPolicy(
        param_dtype=DT[mp["param"]] if mp else None, reduce_dtype=DT[mp["reduce"]] if mp and mp.get("reduce") else None
    )
    if par == "fsdp_tp":
        m2 = meshes[mesh_key(seg)]
        parallelize_module(model, m2["tp"], tp_plan(scn, seg))
        if scn.get("profile") == "train":
            replicate_leftovers(model, m2["tp"])
        fmesh = m2["dp"]
    elif par == "hsdp":
        fmesh = meshes[mesh_key(seg)]
    elif par in ("pp_fsdp", "pp"):
        fmesh = meshes[pp_key(seg)]["dp"]
    elif seg.get("mesh_from_group"):  # Megatron-Core: DeviceMesh.from_group(data-parallel group)
        fmesh = meshes["_from_group"]
    else:
        fmesh = meshes["1d"]
    ign = set()
    if par == "ep":  # experts Shard(0) over the 1-D mesh, FSDP2 for the rest (torchtitan: experts outside dp_shard)
        apply_ep(model, meshes["1d"], seg.get("a2a", "functional"), meshes.get("_ep_pg"))
        ign = set(expert_params(model))
    apply_ac(model, scn.get("ac"))
    model = apply_compile(scn, model, "pre")
    raf = seg.get("raf", True)
    units = []
    groups = seg.get("groups") or []  # torchtitan: fully_shard([norm, lm_head]) as one unit
    grouped = {u for g in groups for u in g}
    for u in seg.get("units", []):
        if u in grouped:
            continue
        try:
            mod = fsdp_unit(model, u)
        except (KeyError, IndexError, AttributeError):
            continue  # unit not on this pipeline stage
        r = raf
        if isinstance(r, int) and not isinstance(r, bool):
            r = r if fmesh.ndim == 1 and fmesh.size() % r == 0 and fmesh.size() > r else True
        units.append(
            fully_shard(
                mod,
                mesh=fmesh,
                reshard_after_forward=r,
                mp_policy=mpp,
                ignored_params=(ign & set(mod.parameters())) or None,
            )
        )
    for g in groups:
        try:
            mods = [fsdp_unit(model, u) for u in g]
        except (KeyError, IndexError, AttributeError):
            continue
        fully_shard(mods, mesh=fmesh, reshard_after_forward=bool(raf), mp_policy=mpp)
        units.append(mods[0])
    fully_shard(
        model,
        mesh=fmesh,
        reshard_after_forward=bool(seg.get("raf_root", True)),
        mp_policy=mpp,
        ignored_params=ign or None,
    )
    fsdp_knobs(seg, model, units)
    return apply_compile(scn, model, "post")


def canon(name):
    name = name.replace("_checkpoint_wrapped_module.", "").replace("_orig_mod.", "")
    return name[len("module.") :] if name.startswith("module.") else name


def full_of(t):
    from torch.distributed.tensor import DTensor

    if isinstance(t, DTensor):
        t = t.full_tensor()
    return t.detach().cpu().clone()


def gather(model, opt):
    # deliberately independent of get_state_dict: full_tensor() of every param, persistent buffer, optim state.
    # model / opt may be lists (pipeline model chunks and their optimizers)
    models = model if isinstance(model, (list, tuple)) else [model]
    opts = opt if isinstance(opt, (list, tuple)) else [opt]
    out = {}
    pname = {}
    for mdl in models:
        for n, p in mdl.named_parameters():
            out["p:" + canon(n)] = full_of(p)
            pname[p] = canon(n)
        for mn, mod in mdl.named_modules():
            for bn, b in mod.named_buffers(recurse=False):
                if bn in mod._non_persistent_buffers_set:
                    continue
                out["b:" + canon(mn + "." + bn if mn else bn)] = full_of(b)
    for o in opts:
        for p, st in o.state.items():
            n = pname.get(p, "?")
            for k, v in st.items():
                if isinstance(v, torch.Tensor):
                    out[f"o:{n}:{k}"] = full_of(v)
                else:
                    out[f"o:{n}:{k}"] = torch.tensor(float(v))
    return out


def gather_all(parts, opts, pp):
    """gather(); under PP every rank holds some chunks, so merge every rank's view (dp replicas agree)."""
    g = gather(parts, opts)
    if not pp:
        return g
    allg = [None] * dist.get_world_size()
    dist.all_gather_object(allg, g, group=DEV["ctl"])
    out = {}
    for d in allg:
        out.update(d)
    return out


def named_params(scn, model):
    """Trainable parameters; (name, param) pairs when the optimizer must know FQNs (torchtitan builds its optimizers
    from named parameters so Optimizer.state_dict() carries param_names)."""
    ps = [(canon(n), p) for n, p in model.named_parameters() if p.requires_grad]
    return ps if scn.get("named_opt") else [p for _, p in ps]


def thash(d):
    h = hashlib.md5()
    for k in sorted(d):
        t = d[k].contiguous().reshape(-1)
        h.update(k.encode())
        h.update(t.view(torch.uint8).numpy().tobytes() if t.numel() else b"")
    return h.hexdigest()[:12]


def tol(scn, seg):
    if seg.get("mp_any"):
        return 5e-2, 5e-2
    if scn["dtype"] == "f64":
        return 1e-7, 1e-9
    return 2e-4, 2e-5


def cat_of(k):
    parts = k.split(":")
    if parts[0] == "o":
        return "o:" + parts[-1]
    return parts[0] + ":" + parts[1].split(".")[-1]


def cmp_states(ref, got, rtol, atol, exact=False):
    diffs = []
    for k in sorted(set(ref) | set(got)):
        if k not in got:
            diffs.append((k, "missing"))
            continue
        if k not in ref:
            diffs.append((k, "unexpected"))
            continue
        r, g = ref[k], got[k]
        if tuple(r.shape) != tuple(g.shape):
            diffs.append((k, f"shape {tuple(r.shape)} vs {tuple(g.shape)}"))
            continue
        if r.dtype != g.dtype and exact:
            diffs.append((k, f"dtype {r.dtype} vs {g.dtype}"))
            continue
        if r.numel() == 0:
            continue
        if exact:
            ok = torch.equal(r, g) or bool(torch.isclose(r.double(), g.double(), 0, 0, equal_nan=True).all())
        else:
            ok = bool(torch.isclose(g.double(), r.double(), rtol=rtol, atol=atol, equal_nan=True).all())
        if not ok:
            md = (g.double() - r.double()).abs().nan_to_num(float("inf")).max().item()
            diffs.append((k, f"maxdiff={md:.3g}"))
    return diffs


def clip_cfg(scn):
    c = scn.get("clip")
    if not c:
        return None
    return {"max": c} if isinstance(c, (int, float)) else c


def micro_batches(scn, step, dp=1, dpr=0, device=None):
    """The step's batch split into `accum` micro-batches, then this rank's 1/dp of each."""
    x, y = batch(scn, step)
    if device is not None:
        x, y = x.to(device), y.to(device)
    k = scn.get("accum", 1)
    m = x.shape[0] // k
    n = m // dp
    return [(x[i * m + dpr * n : i * m + (dpr + 1) * n], y[i * m + dpr * n : i * m + (dpr + 1) * n]) for i in range(k)]


def ref_run(scn, nsteps, reset_opt_at=()):
    model = build_model(scn, scn["seed"])
    opt = build_optim(scn, [p for p in model.parameters() if p.requires_grad])
    losses, gnorms, states = [], [], {0: gather(model, opt)}
    cc = clip_cfg(scn)
    for s in range(nsteps):
        if s in reset_opt_at:  # a model-only checkpoint restarts the optimizer from scratch
            opt = build_optim(scn, [p for p in model.parameters() if p.requires_grad])
        model.train()
        tot = 0.0
        mbs = micro_batches(scn, s)
        for x, y in mbs:
            loss = loss_fn(scn, model(x), y) / len(mbs)
            loss.backward()
            tot += loss.item()
        if cc:
            gnorms.append(clip_grads([p for p in model.parameters() if p.requires_grad], dict(cc, impl="torch")))
        opt.step()
        opt.zero_grad(set_to_none=scn.get("zg_none", True))
        losses.append(tot)
        states[s + 1] = gather(model, opt)
    return losses, gnorms, states


def sdo(**kw):
    from torch.distributed.checkpoint.state_dict import StateDictOptions

    return StateDictOptions(**kw)


class AppState:
    def __init__(self, model, opt, opts):
        self.model, self.opt, self.opts = model, opt, opts

    def state_dict(self):
        from torch.distributed.checkpoint.state_dict import get_state_dict

        m, o = get_state_dict(self.model, self.opt, options=self.opts)
        return {"model": m, "optim": o}

    def load_state_dict(self, sd):
        from torch.distributed.checkpoint.state_dict import set_state_dict

        set_state_dict(
            self.model, self.opt, model_state_dict=sd["model"], optim_state_dict=sd["optim"], options=self.opts
        )


def _stateful(cls):
    from torch.distributed.checkpoint.stateful import Stateful

    return type("AppStateS", (cls, Stateful), {})


def save_ckpt(scn, sv, model, opt, path, step, rt):
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.state_dict import get_optimizer_state_dict, get_state_dict

    part_list = list(model) if isinstance(model, (list, tuple)) else [model]
    opt_list = list(opt) if isinstance(opt, (list, tuple)) else [opt]
    model, opt = part_list[0], opt_list[0]

    ign = bool(sv.get("ign_frozen"))
    fos = bool(scn.get("flat_osd"))
    mode = sv.get("mode", "sharded")
    opts = sdo(
        full_state_dict=mode != "sharded",
        cpu_offload=mode == "full_offload",
        ignore_frozen_params=ign,
        flatten_optimizer_state_dict=fos,
    )
    api = sv.get("api", "sd")
    if api == "titan":  # torchtitan checkpointer: dcp.save of Stateful ModelWrapper / OptimizersContainer
        state = {"model": _stateful(TitanModel)(part_list), "optimizer": _stateful(TitanOptim)(opt_list)}
    elif api == "stateful":
        state = {
            "app": _stateful(AppState)(model, opt, sdo(ignore_frozen_params=ign, flatten_optimizer_state_dict=fos))
        }
    elif api == "raw":
        state = {"model": model.state_dict(), "optim": get_optimizer_state_dict(model, opt, options=opts)}
    else:
        m, o = get_state_dict(model, opt, options=opts)
        state = {"model": m, "optim": o}
    state["extra"] = {"step": torch.tensor(step), "tag": f"s{step}"}
    pl = sv.get("planner", {})
    planner = dcp.DefaultSavePlanner(
        flatten_state_dict=pl.get("flat", True),
        flatten_sharded_tensors=pl.get("flat_sh", True),
        dedup_save_to_lowest_rank=pl.get("dedup_low", False),
        enable_plan_caching=pl.get("cache", False),
    )
    wr = sv.get("writer", {})
    fmt_kw = {}
    if wr.get("fmt") == "safetensors":
        from torch.distributed.checkpoint.filesystem import SerializationFormat

        fmt_kw["serialization_format"] = SerializationFormat.SAFETENSORS
    writer = dcp.FileSystemWriter(
        path,
        single_file_per_rank=wr.get("sfpr", True),
        sync_files=False,
        thread_count=wr.get("threads", 1),
        per_thread_copy_ahead=wr.get("copy_ahead", 10_000_000),
        **fmt_kw,
    )
    a = sv.get("async", "none")
    if a == "none":
        dcp.save(state, storage_writer=writer, planner=planner)
        return None
    from torch.distributed.checkpoint.state_dict_saver import AsyncCheckpointerType

    pg = rt["ckpt_pg"] if sv.get("async_pg") == "new" else None
    fut = dcp.async_save(
        state,
        storage_writer=writer,
        planner=planner,
        process_group=pg,
        async_checkpointer_type=AsyncCheckpointerType.PROCESS if a == "process" else AsyncCheckpointerType.THREAD,
    )
    return lambda: fut.result() if hasattr(fut, "result") else fut.upload_completion.result()


def load_ckpt(scn, ld, sv_prev, model, opt, path, rank):
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.state_dict import (
        get_optimizer_state_dict,
        get_state_dict,
        set_optimizer_state_dict,
        set_state_dict,
    )

    part_list = list(model) if isinstance(model, (list, tuple)) else [model]
    opt_list = list(opt) if isinstance(opt, (list, tuple)) else [opt]
    model, opt = part_list[0], opt_list[0]

    ign = bool(sv_prev.get("ign_frozen"))
    fos = bool(scn.get("flat_osd"))
    mode = ld.get("mode", "sharded")
    api = ld.get("api", "sd")
    pl = ld.get("planner", {})
    planner = dcp.DefaultLoadPlanner(
        flatten_state_dict=pl.get("flat", True), flatten_sharded_tensors=pl.get("flat_sh", True)
    )
    reader = dcp.FileSystemReader(path)
    extra = {"step": torch.tensor(-1), "tag": ""}
    if api == "titan":
        state = {
            "model": _stateful(TitanModel)(part_list),
            "optimizer": _stateful(TitanOptim)(opt_list),
            "extra": extra,
        }
        dcp.load(state, storage_reader=reader, planner=planner)
        return state["extra"]
    if api == "stateful":
        app = _stateful(AppState)(model, opt, sdo(ignore_frozen_params=ign, flatten_optimizer_state_dict=fos))
        state = {"app": app, "extra": extra}
        dcp.load(state, storage_reader=reader, planner=planner)
        return state["extra"]
    if mode in ("full_bcast", "full_bcast_split", "model_bcast"):
        opts = sdo(full_state_dict=True, cpu_offload=True, ignore_frozen_params=ign, flatten_optimizer_state_dict=fos)
        if mode == "model_bcast":
            # model only (torchtune / accelerate): get_state_dict would also run DSD's zero-lr optimizer init step
            from torch.distributed.checkpoint.state_dict import get_model_state_dict

            state = {"model": get_model_state_dict(model, options=opts), "extra": extra}
        else:
            m, o = get_state_dict(model, opt, options=opts)  # rank0 full, others empty
            state = {"model": m, "optim": o, "extra": extra}
        if rank == 0:
            dcp.load(state, storage_reader=reader, planner=planner, no_dist=True)
        bopts = sdo(
            full_state_dict=True,
            broadcast_from_rank0=True,
            cpu_offload=ld.get("bcast_offload", False),
            ignore_frozen_params=ign,
            flatten_optimizer_state_dict=fos,
        )
        if mode == "full_bcast":
            set_state_dict(model, opt, model_state_dict=state["model"], optim_state_dict=state["optim"], options=bopts)
        else:
            # torchtune load_from_full_model_state_dict / accelerate cpu_ram_efficient_loading: split API
            from torch.distributed.checkpoint.state_dict import set_model_state_dict

            set_model_state_dict(model, state["model"], options=bopts)
            if mode == "full_bcast_split":
                set_optimizer_state_dict(model, opt, state["optim"], options=bopts)
        t = ctl_t(torch.tensor([state["extra"]["step"].item() if rank == 0 else 0]))
        dist.broadcast(t, 0, group=DEV["ctl"])
        t = t.cpu()
        return {"step": t, "tag": ""}
    opts = sdo(full_state_dict=mode == "full", ignore_frozen_params=ign, flatten_optimizer_state_dict=fos)
    if api == "raw":
        state = {
            "model": model.state_dict(),
            "optim": get_optimizer_state_dict(model, opt, options=opts),
            "extra": extra,
        }
        if ld.get("legacy"):  # Megatron-Core torch_dcp: the deprecated dcp.load_state_dict entry point
            dcp.load_state_dict(state, storage_reader=reader, planner=planner)
        else:
            dcp.load(state, storage_reader=reader, planner=planner)
        model.load_state_dict(state["model"], strict=not ign)
        set_optimizer_state_dict(model, opt, state["optim"], options=opts)
    else:
        m, o = get_state_dict(model, opt, options=opts)
        state = {"model": m, "optim": o, "extra": extra}
        dcp.load(state, storage_reader=reader, planner=planner)
        set_state_dict(model, opt, model_state_dict=state["model"], optim_state_dict=state["optim"], options=opts)
    return state["extra"]


_READER = r"""
import sys, torch
try:  # float8 checkpoints hold torchao tensor subclasses
    import torchao.float8  # noqa: F401
except ImportError:
    pass
from torch.distributed.checkpoint.format_utils import dcp_to_torch_save
dcp_to_torch_save(sys.argv[1], sys.argv[2])
"""


def read_ckpt(path):
    # fresh process: flatten_state_dict=False checkpoints contain pickled DTensors that unpickle onto live meshes
    import subprocess
    import sys

    tmp = path.rstrip("/") + ".flat.pt"
    r = subprocess.run(
        [sys.executable, "-c", _READER, path, tmp],
        capture_output=True,
        text=True,
        # the harness reads its own checkpoints: allow the tensor subclasses weights_only rejects (torchao float8)
        env=dict(os.environ, OMP_NUM_THREADS="1", TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD="1"),
        timeout=120,
    )
    if r.returncode:
        raise RuntimeError("dcp_to_torch_save failed: " + (r.stderr.strip().splitlines() or ["?"])[-1])
    sd = torch.load(tmp, weights_only=False)
    os.remove(tmp)
    flat = {}

    def rec(pfx, v):
        if isinstance(v, dict):
            for k, x in v.items():
                rec(f"{pfx}.{k}" if pfx else str(k), x)
        elif isinstance(v, (list, tuple)):
            for i, x in enumerate(v):
                rec(f"{pfx}.{i}", x)
        else:
            flat[pfx] = v

    rec("", sd)
    out = {}
    for k, v in flat.items():
        k = k[4:] if k.startswith("app.") else k
        out[k] = v
    return out


def cmp_ckpt(a, b):
    diffs = []
    for k in sorted(set(a) | set(b)):
        if k.startswith("extra."):
            continue
        if k not in a or k not in b:
            diffs.append((k, "only in " + ("resave" if k in b else "orig")))
            continue
        x, y = a[k], b[k]
        if isinstance(x, torch.Tensor) and isinstance(y, torch.Tensor):
            if (
                x.shape != y.shape
                or x.dtype != y.dtype
                or not (torch.equal(x, y) or bool(torch.isclose(x.double(), y.double(), 0, 0, equal_nan=True).all()))
            ):
                diffs.append((k, "tensor differs"))
        elif isinstance(x, torch.Tensor) or isinstance(y, torch.Tensor):
            diffs.append((k, f"type {type(x).__name__} vs {type(y).__name__}"))
        elif x != y:
            diffs.append((k, f"{x!r} vs {y!r}"))
    return diffs


def norm_msg(e):
    s = (str(e).strip().splitlines() or [""])[0][:160]
    s = re.sub(r"0x[0-9a-f]+", "ADDR", s)
    s = re.sub(r"/[\w/.\-]+", "PATH", s)
    s = re.sub(r"\d+", "N", s)
    return f"{type(e).__name__}: {s}"


class Stop(Exception):
    pass


def run_segment(rt, scn, si, workdir):
    rank, W = rt["rank"], rt["world"]
    meshes = rt["meshes"]
    segs = scn["segs"]
    seg = segs[si]
    assert seg["w"] == W
    F_ = []
    hashes = []
    start = sum(s["steps"] for s in segs[:si])
    end = start + seg["steps"]
    sv = seg.get("save") if si < len(segs) - 1 else None
    extra_steps = sv.get("mutate", 0) if sv and sv.get("async", "none") != "none" else 0
    ref = {}
    # reference is only approximate once any bf16/f32-cast step ran, or under float8 (casts of differently
    # reduced activations): loss and grad norm get a loose tolerance, states are not compared
    mp_any = any(s.get("mp") for s in segs[: si + 1]) or bool(scn.get("f8"))
    rtol, atol = tol(scn, dict(seg, mp_any=mp_any))
    pair = (segs[si - 1]["par"] + "->" if si else "") + seg["par"]

    cur = {"sub": None, "phase": None}

    def mark(phase=None, sub=None):
        # the harness reads the last marker of a rank that hangs (World.hang_report)
        if phase is not None:
            cur["phase"] = phase
        cur["sub"] = sub
        full = cur["phase"] + (":" + sub if sub else "")
        print(f"DISTFUZZ_PHASE rank={rank} phase={full}", file=sys.stderr, flush=True)

    tags = {}
    for k in MOE_STATS:
        MOE_STATS[k] = False

    def sync(phase, fn):
        err, tb = None, None
        mark(phase)
        try:
            val = fn()
        except BaseException as e:  # noqa: B036 (DCP's CheckpointException derives from BaseException)
            if isinstance(e, (KeyboardInterrupt, SystemExit)):
                raise
            err, tb, val = e, traceback.format_exc(), None
            if cur["sub"]:
                phase = f"{phase}:{cur['sub']}"
            # the harness reads this line if the other ranks then hang (error on some ranks, hang on the rest)
            print(f"DISTFUZZ_ERR rank={rank} phase={phase} msg={norm_msg(err)}", file=sys.stderr, flush=True)
            print("DISTFUZZ_TB " + tb.replace("\n", "\nDISTFUZZ_TB "), file=sys.stderr, flush=True)
        t = ctl_t(torch.tensor([1 if err is not None else 0]))
        flags = [ctl_t(torch.zeros(1, dtype=torch.long)) for _ in range(W)]
        dist.all_gather(flags, t, group=DEV["ctl"])
        flags = [int(f.item()) for f in flags]
        if any(flags):
            if all(flags):
                F_.append(dict(kind="ERR", phase=phase, msg=norm_msg(err), pair=pair, seg=si, tb=tb[-1500:]))
            else:
                F_.append(
                    dict(
                        kind="RANK_DIVERGENT_ERR",
                        phase=phase,
                        msg=norm_msg(err) if err else "other rank raised",
                        pair=pair,
                        seg=si,
                        tb=(tb or "")[-1500:],
                        flags=flags,
                    )
                )
            raise Stop()
        return val

    def report(kind, diffs, extra=""):
        if diffs:
            cats = sorted({cat_of(k) if ":" in k else re.sub(r"\d+", "N", k) for k, _ in diffs})
            F_.append(
                dict(
                    kind=kind,
                    phase="check",
                    pair=pair,
                    seg=si,
                    cats=cats[:6],
                    msg=f"{len(diffs)} diffs {extra}; first: " + "; ".join(f"{k} {m}" for k, m in diffs[:4]),
                )
            )

    try:
        if rank == 0:
            resets = [
                sum(x["steps"] for x in segs[:j])
                for j in range(1, si + 1)
                if segs[j].get("load", {}).get("mode") == "model_bcast"
            ]
            rl, rg, rs = ref_run(scn, end + extra_steps, reset_opt_at=resets)
            ref["losses"], ref["gnorms"], ref["states"] = rl, rg, rs

        # 1. build
        pp = seg["par"] in ("pp", "pp_fsdp")
        pinfo = {}
        if scn.get("compile"):
            from torch import _dynamo

            _dynamo.reset()  # fresh caches per segment (recompile limits, stale guards)

        def poison(net):
            if si > 0:  # poison everything the checkpoint must restore
                ign = segs[si - 1]["save"].get("ign_frozen")
                with torch.no_grad():
                    for p in net.parameters():
                        if p.requires_grad or not ign:
                            p.add_(7.0)
                    for _, mod in net.named_modules():
                        for bn, b in mod.named_buffers(recurse=False):
                            if bn not in mod._non_persistent_buffers_set:
                                b.add_(5.0)

        def build():
            if pp:
                parts, opts, sched, ids, nst, pm = build_pp(scn, seg, meshes, rank, poison)
                pinfo.update(sched=sched, ids=ids, nst=nst, pm=pm)
                return parts, opts
            model = build_model(scn, scn["seed"])
            poison(model)
            model = parallelize(scn, seg, model.to(dev()), meshes)
            opt = build_optim(scn, named_params(scn, model))
            return [model], [opt]

        parts, opts = sync("build", build)
        model, opt = parts[0], opts[0]
        if pp:
            ppd, dp = seg.get("ppmesh", [W, 1])
            dpr = pinfo["pm"].get_coordinate()[1]
        else:
            dp, dpr = dp_info(seg, rank, meshes)
        B = scn["batch"]
        assert B % (dp * scn.get("accum", 1)) == 0
        tags["empty_shard"] = any(has_empty_local(m) for m in parts)
        mobj = parts if pp else model  # what save / load / gather take
        oobj = opts if pp else opt

        # 2. load
        if si > 0:
            prev = segs[si - 1]
            ppath = os.path.join(workdir, f"ck{si - 1}")
            ex = sync("load", lambda: load_ckpt(scn, seg.get("load", {}), prev["save"], mobj, oobj, ppath, rank))
            g = sync("gather_after_load", lambda: gather_all(parts, opts, pp))
            hashes.append(thash(g))
            if rank == 0:
                saved = torch.load(ppath + ".gathered.pt", weights_only=False)
                if int(ex["step"]) != start:
                    F_.append(
                        dict(
                            kind="LOAD_EXTRA",
                            phase="check",
                            pair=pair,
                            seg=si,
                            msg=f"extra.step={int(ex['step'])} expected {start}",
                        )
                    )
                if seg.get("load", {}).get("mode") == "model_bcast":  # optimizer restarts: compare the model only
                    saved = {k: v for k, v in saved.items() if not k.startswith("o:")}
                    g = {k: v for k, v in g.items() if not k.startswith("o:")}
                d = cmp_states(saved, g, 0, 0, exact=True)
                # get_state_dict lazily creates zero optimizer state; that is equivalent to "no state yet"
                d = [(k, m) for k, m in d if not (m == "unexpected" and k.startswith("o:") and not g[k].any())]
                report("LOAD_MISMATCH", d)
            if seg.get("resave"):
                rpath = os.path.join(workdir, f"rs{si}")
                sync(
                    "resave",
                    lambda: save_ckpt(scn, dict(prev["save"], **{"async": "none"}), mobj, oobj, rpath, start, rt),
                )
                if rank == 0:
                    try:
                        a, b = read_ckpt(ppath), read_ckpt(rpath)
                        if seg.get("load", {}).get("mode") == "model_bcast":  # optimizer restarted on purpose
                            a = {k: v for k, v in a.items() if not k.startswith("optim.")}
                            b = {k: v for k, v in b.items() if not k.startswith("optim.")}
                        report("ROUNDTRIP_MISMATCH", cmp_ckpt(a, b))
                    except Exception as e:  # noqa
                        F_.append(
                            dict(
                                kind="ERR",
                                phase="read_ckpt",
                                msg=norm_msg(e),
                                pair=pair,
                                seg=si,
                                tb=traceback.format_exc()[-1500:],
                            )
                        )
            t = ctl_t(torch.tensor([1 if F_ else 0]))
            dist.broadcast(t, 0, group=DEV["ctl"])
            if int(t.item()):
                raise Stop()

        # 3. train
        losses, gnorms = [], []
        cc = clip_cfg(scn)
        lp = seg.get("tp", {}).get("head") == "col_lp"
        eparams = expert_params(model) if seg["par"] == "ep" else []
        gdf1 = bool((seg.get("fsdp_knobs") or {}).get("gdf1"))
        defer = bool((seg.get("fsdp_knobs") or {}).get("defer_reduce"))

        def all_params():
            return [p for m in parts for p in m.parameters() if p.requires_grad]

        def train_pp(s0, s1):
            sched, ids, nst = pinfo["sched"], pinfo["ids"], pinfo["nst"]
            first, last = 0 in ids, nst - 1 in ids
            for s in range(s0, s1):
                for m in parts:
                    m.train()
                ((x, y),) = micro_batches(scn, s, dp, dpr, dev())
                mb_losses = []
                mark(sub="fwd")
                sched.step(*((x,) if first else ()), target=y if last else None, losses=mb_losses if last else None)
                if cc:
                    mark(sub="clip")
                    gnorms.append(clip_grads(all_params(), cc, pinfo["pm"].get_group("pp")))
                mark(sub="opt")
                for o in opts:
                    o.step()
                    o.zero_grad(set_to_none=scn.get("zg_none", True))
                mark(sub="log")
                if last:
                    tot = sum(float(v) for v in mb_losses) * (dp if gdf1 else 1)
                    tot = tot if seg["pp"].get("titan_loss") else tot / len(mb_losses)
                else:
                    tot = 0.0
                lt = ctl_t(torch.tensor([tot], dtype=torch.float64))
                dist.all_reduce(lt, group=DEV["ctl"])
                losses.append(lt.item() / dp)
            cur["sub"] = None

        def train(s0, s1):
            import contextlib

            for s in range(s0, s1):
                model.train()
                mbs = micro_batches(scn, s, dp, dpr, dev())
                tot = torch.zeros((), dtype=torch.float64)
                for i, (x, y) in enumerate(mbs):
                    last = i == len(mbs) - 1
                    ctx = contextlib.nullcontext()
                    if defer and len(mbs) > 1 and hasattr(model, "set_is_last_backward"):
                        # torchtitan defer_fsdp_gradient_reduction: reduce and reshard only on the last micro-batch
                        model.set_is_last_backward(last)
                        model.set_reshard_after_backward(last)
                        model.set_requires_gradient_sync(last)
                    elif scn.get("no_sync") and len(mbs) > 1:
                        if seg["par"] == "ddp" and not last:
                            ctx = model.no_sync()
                        elif hasattr(model, "set_requires_gradient_sync"):  # accelerate's FSDP2 no_sync
                            model.set_requires_gradient_sync(last)
                    if lp:
                        from torch.distributed.tensor.parallel import loss_parallel

                        ctx = loss_parallel()
                    with ctx:
                        mark(sub="fwd")
                        loss = loss_fn(scn, model(x), y) / len(mbs)
                        mark(sub="bwd")
                        # gradient_divide_factor 1.0: FSDP sums over dp, the loss carries the 1/dp (torchtitan)
                        (loss / dp if gdf1 else loss).backward()
                    lv = loss.detach()
                    tot += (lv.full_tensor() if hasattr(lv, "full_tensor") else lv).double().cpu()
                for p in eparams if not gdf1 else []:  # experts see every rank's tokens; FSDP averages the rest
                    if p.grad is not None:
                        p.grad.div_(dp)
                if cc:
                    mark(sub="clip")
                    gnorms.append(clip_grads([p for p in model.parameters() if p.requires_grad], cc))
                mark(sub="opt")
                opt.step()
                opt.zero_grad(set_to_none=scn.get("zg_none", True))
                if (scn.get("f8") or {}).get("precompute"):
                    # accelerate registers precompute_float8_dynamic_scale_for_fsdp as an optimizer step post hook
                    from torchao.float8 import precompute_float8_dynamic_scale_for_fsdp

                    mark(sub="f8_scale")
                    precompute_float8_dynamic_scale_for_fsdp(model)
                mark(sub="log")
                lt = ctl_t(tot.reshape(1).clone())
                dist.all_reduce(lt, group=DEV["ctl"])
                losses.append(lt.item() / W)
            cur["sub"] = None

        trainf = train_pp if pp else train
        sync("train", lambda: trainf(start, end))
        if rank == 0:
            rl = ref["losses"][start:end]
            bad = [
                (start + i, a, b)
                for i, (a, b) in enumerate(zip(rl, losses))
                if not math.isclose(a, b, rel_tol=rtol, abs_tol=atol)
            ]
            if bad:
                F_.append(
                    dict(
                        kind="LOSS_MISMATCH",
                        phase="check",
                        pair=pair,
                        seg=si,
                        msg="first bad step {:d} ref={:.10g} got={:.10g} ({:d} bad)".format(*bad[0], len(bad)),
                    )
                )
            rg = ref["gnorms"][start:end]
            bad = [
                (start + i, a, b)
                for i, (a, b) in enumerate(zip(rg, gnorms))
                if not math.isclose(a, b, rel_tol=rtol, abs_tol=atol)
            ]
            if bad and not mp_any:
                F_.append(
                    dict(
                        kind="GNORM_MISMATCH",
                        phase="check",
                        pair=pair,
                        seg=si,
                        msg="first bad step {:d} ref={:.10g} got={:.10g} ({:d} bad)".format(*bad[0], len(bad)),
                    )
                )
        g = sync("gather_end", lambda: gather_all(parts, opts, pp))
        hashes.append(thash(g))
        if rank == 0 and not mp_any:
            report("STATE_MISMATCH", cmp_states(ref["states"][end], g, rtol, atol))

        # 4. save
        if sv is not None:
            path = os.path.join(workdir, f"ck{si}")
            if rank == 0:
                torch.save(g, path + ".gathered.pt")
            if sv.get("presave"):
                sync(
                    "presave",
                    lambda: save_ckpt(
                        scn, dict(sv, **{"async": "none"}), mobj, oobj, os.path.join(workdir, f"pre{si}"), end, rt
                    ),
                )
            waiter = sync("save", lambda: save_ckpt(scn, sv, mobj, oobj, path, end, rt))
            if waiter is not None:
                if extra_steps:
                    losses.clear()
                    sync("train_during_async", lambda: trainf(end, end + extra_steps))
                sync("async_wait", waiter)
            sync("post_save_barrier", lambda: dist.barrier())
    except Stop:
        pass
    tags.update(MOE_STATS)
    return dict(findings=F_, hashes=hashes, tags=tags)
