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
        apply_ac,
        apply_ep,
        clip_grads,
        expert_params,
        has_empty_local,
        replicate_leftovers,
    )

DT = {"f64": torch.float64, "f32": torch.float32, "bf16": torch.bfloat16, "f16": torch.float16}


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
        B, L, _ = x.shape
        q = self.q(x).reshape(B, L, -1, self.hd).transpose(1, 2)
        k = self.k(x).reshape(B, L, -1, self.hd).transpose(1, 2)
        v = self.v(x).reshape(B, L, -1, self.hd).transpose(1, 2)
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
    return net


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
    return cls(params, lr=o["lr"], betas=(0.9, 0.95), weight_decay=o["wd"], amsgrad=o["amsgrad"], **kw)


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


def tp_plan(scn, seg):
    from torch.distributed.tensor import Replicate
    from torch.distributed.tensor.parallel import ColwiseParallel, RowwiseParallel

    plan = {}
    for name, style in seg.get("tp", {}).items():
        if name == "emb":
            plan["emb"] = (
                RowwiseParallel(input_layouts=Replicate())
                if style == "row"
                else ColwiseParallel(output_layouts=Replicate())
            )
        elif name == "head":
            if style == "col_lp":  # torchtitan lm_head under loss_parallel: vocab-sharded logits stay DTensors
                from torch.distributed.tensor import Shard

                plan["head"] = ColwiseParallel(output_layouts=Shard(-1), use_local_output=False)
            else:
                plan["head"] = (
                    ColwiseParallel(output_layouts=Replicate())
                    if style == "col"
                    else RowwiseParallel(input_layouts=Replicate())
                )
        else:
            i = int(name)
            t = scn["model"]["blocks"][i]["t"]
            p = f"blocks.{i}"
            if t == "lin":
                plan[p + ".lin"] = (
                    ColwiseParallel(output_layouts=Replicate())
                    if style == "col"
                    else RowwiseParallel(input_layouts=Replicate())
                )
            elif t == "mlp":
                plan[p + ".fc1"] = ColwiseParallel()
                plan[p + ".fc2"] = RowwiseParallel()
            elif t == "attn":
                for q in ("q", "k", "v"):
                    plan[f"{p}.{q}"] = ColwiseParallel()
                plan[p + ".o"] = RowwiseParallel()
    return plan


def parallelize(scn, seg, model, meshes):
    from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
    from torch.distributed.tensor.parallel import parallelize_module

    par = seg["par"]
    if par == "none":
        return model
    if par == "ddp":
        from torch.nn.parallel import DistributedDataParallel as DDP

        apply_ac(model, scn.get("ac"))
        return DDP(
            model, broadcast_buffers=seg.get("ddp_bcast", True), gradient_as_bucket_view=seg.get("ddp_gabv", False)
        )
    if par == "tp":
        parallelize_module(model, meshes["1d"], tp_plan(scn, seg))
        if scn.get("profile") == "train":
            replicate_leftovers(model, meshes["1d"])
        apply_ac(model, scn.get("ac"))
        return model
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
    else:
        fmesh = meshes["1d"]
    ign = set()
    if par == "ep":  # experts Shard(0) over the 1-D mesh, FSDP2 for the rest (torchtitan: experts outside dp_shard)
        apply_ep(model, meshes["1d"], seg.get("a2a", "functional"), meshes.get("_ep_pg"))
        ign = set(expert_params(model))
    apply_ac(model, scn.get("ac"))
    raf = seg.get("raf", True)
    for u in seg.get("units", []):
        mod = model.emb if u == "emb" else model.head if u == "head" else model.blocks[int(u)]
        r = raf
        if isinstance(r, int) and not isinstance(r, bool):
            r = r if fmesh.ndim == 1 and fmesh.size() % r == 0 and fmesh.size() > r else True
        fully_shard(
            mod,
            mesh=fmesh,
            reshard_after_forward=r,
            mp_policy=mpp,
            ignored_params=(ign & set(mod.parameters())) or None,
        )
    fully_shard(
        model,
        mesh=fmesh,
        reshard_after_forward=bool(seg.get("raf_root", True)),
        mp_policy=mpp,
        ignored_params=ign or None,
    )
    return model


def canon(name):
    name = name.replace("_checkpoint_wrapped_module.", "")
    return name[len("module.") :] if name.startswith("module.") else name


def full_of(t):
    from torch.distributed.tensor import DTensor

    if isinstance(t, DTensor):
        t = t.full_tensor()
    return t.detach().cpu().clone()


def gather(model, opt):
    # deliberately independent of get_state_dict: full_tensor() of every param, persistent buffer, optim state
    out = {}
    pname = {}
    for n, p in model.named_parameters():
        out["p:" + canon(n)] = full_of(p)
        pname[p] = canon(n)
    for mn, mod in model.named_modules():
        for bn, b in mod.named_buffers(recurse=False):
            if bn in mod._non_persistent_buffers_set:
                continue
            out["b:" + canon(mn + "." + bn if mn else bn)] = full_of(b)
    for p, st in opt.state.items():
        n = pname.get(p, "?")
        for k, v in st.items():
            if isinstance(v, torch.Tensor):
                out[f"o:{n}:{k}"] = full_of(v)
            else:
                out[f"o:{n}:{k}"] = torch.tensor(float(v))
    return out


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


def micro_batches(scn, step, dp=1, dpr=0):
    """The step's batch split into `accum` micro-batches, then this rank's 1/dp of each."""
    x, y = batch(scn, step)
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
    if api == "stateful":
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
        t = torch.tensor([state["extra"]["step"].item() if rank == 0 else 0])
        dist.broadcast(t, 0)
        return {"step": t, "tag": ""}
    opts = sdo(full_state_dict=mode == "full", ignore_frozen_params=ign, flatten_optimizer_state_dict=fos)
    if api == "raw":
        state = {
            "model": model.state_dict(),
            "optim": get_optimizer_state_dict(model, opt, options=opts),
            "extra": extra,
        }
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
        env=dict(os.environ, OMP_NUM_THREADS="1"),
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
    mp_any = any(s.get("mp") for s in segs[: si + 1])  # reference is meaningless once any bf16/f32-cast step ran
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
        t = torch.tensor([1 if err is not None else 0])
        flags = [torch.zeros(1, dtype=torch.long) for _ in range(W)]
        dist.all_gather(flags, t)
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
        def build():
            model = build_model(scn, scn["seed"])
            if si > 0:  # poison everything the checkpoint must restore
                ign = segs[si - 1]["save"].get("ign_frozen")
                with torch.no_grad():
                    for p in model.parameters():
                        if p.requires_grad or not ign:
                            p.add_(7.0)
                    for _, mod in model.named_modules():
                        for bn, b in mod.named_buffers(recurse=False):
                            if bn not in mod._non_persistent_buffers_set:
                                b.add_(5.0)
            model = parallelize(scn, seg, model, meshes)
            opt = build_optim(scn, [p for p in model.parameters() if p.requires_grad])
            return model, opt

        model, opt = sync("build", build)
        dp, dpr = dp_info(seg, rank, meshes)
        B = scn["batch"]
        assert B % (dp * scn.get("accum", 1)) == 0
        tags["empty_shard"] = has_empty_local(model)

        # 2. load
        if si > 0:
            prev = segs[si - 1]
            ppath = os.path.join(workdir, f"ck{si - 1}")
            ex = sync("load", lambda: load_ckpt(scn, seg.get("load", {}), prev["save"], model, opt, ppath, rank))
            g = sync("gather_after_load", lambda: gather(model, opt))
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
                    lambda: save_ckpt(scn, dict(prev["save"], **{"async": "none"}), model, opt, rpath, start, rt),
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
            t = torch.tensor([1 if F_ else 0])
            dist.broadcast(t, 0)
            if int(t.item()):
                raise Stop()

        # 3. train
        losses, gnorms = [], []
        cc = clip_cfg(scn)
        lp = seg.get("tp", {}).get("head") == "col_lp"
        eparams = expert_params(model) if seg["par"] == "ep" else []

        def train(s0, s1):
            import contextlib

            for s in range(s0, s1):
                model.train()
                mbs = micro_batches(scn, s, dp, dpr)
                tot = torch.zeros((), dtype=torch.float64)
                for i, (x, y) in enumerate(mbs):
                    last = i == len(mbs) - 1
                    ctx = contextlib.nullcontext()
                    if scn.get("no_sync") and len(mbs) > 1:
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
                        loss.backward()
                    lv = loss.detach()
                    tot += (lv.full_tensor() if hasattr(lv, "full_tensor") else lv).double()
                for p in eparams:  # experts see every rank's tokens; FSDP averages the rest over dp
                    if p.grad is not None:
                        p.grad.div_(dp)
                if cc:
                    mark(sub="clip")
                    gnorms.append(clip_grads([p for p in model.parameters() if p.requires_grad], cc))
                mark(sub="opt")
                opt.step()
                opt.zero_grad(set_to_none=scn.get("zg_none", True))
                mark(sub="log")
                lt = tot.reshape(1).clone()
                dist.all_reduce(lt)
                losses.append(lt.item() / W)
            cur["sub"] = None

        sync("train", lambda: train(start, end))
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
        g = sync("gather_end", lambda: gather(model, opt))
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
                        scn, dict(sv, **{"async": "none"}), model, opt, os.path.join(workdir, f"pre{si}"), end, rt
                    ),
                )
            waiter = sync("save", lambda: save_ckpt(scn, sv, model, opt, path, end, rt))
            if waiter is not None:
                if extra_steps:
                    losses.clear()
                    sync("train_during_async", lambda: train(end, end + extra_steps))
                sync("async_wait", waiter)
            sync("post_save_barrier", lambda: dist.barrier())
    except Stop:
        pass
    tags.update(MOE_STATS)
    return dict(findings=F_, hashes=hashes, tags=tags)
