# Training-profile building blocks: LoRA, MoE with expert parallelism, activation checkpointing, clip variants.
# Embedded verbatim into emitted repros (before runtime.py): imports must stay stdlib + torch only.
#
# Shapes and names follow the stacks they come from (see templates.py for file:line provenance):
# LoRALinear = torchtune/modules/peft/lora.py; Router/Experts/token exchange = torchtitan models/common/moe;
# clip_titan = torchtitan distributed/utils.py clip_grad_norm_; clip_accel = accelerate
# Accelerator._clip_grad_norm_dtensor_aware.
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# what the MoE layers saw in the current segment (zero-token experts, ranks sending zero rows)
MOE_STATS = {"zero_tok": False, "zero_send": False, "zero_recv": False}


def chunk_sizes(n, w):
    """Local sizes of torch.chunk semantics (Shard placement): ceil(n / w) per rank, trailing shards may be empty."""
    c = -(-n // w) if n else 0
    out, left = [], n
    for _ in range(w):
        k = min(c, max(left, 0))
        out.append(k)
        left -= k
    return out


class LoRALinear(nn.Module):
    # torchtune LoRALinear: frozen base weight, lora_a Linear(in, rank), lora_b Linear(rank, out) zero-init,
    # output = base(x) + (alpha / rank) * lora_b(lora_a(x))
    def __init__(self, din, dout, rank, alpha, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(dout, din), requires_grad=False)
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        self.bias = nn.Parameter(torch.zeros(dout), requires_grad=False) if bias else None
        self.lora_a = nn.Linear(din, rank, bias=False)
        self.lora_b = nn.Linear(rank, dout, bias=False)
        nn.init.zeros_(self.lora_b.weight)
        self.scaling = alpha / rank

    def forward(self, x):
        return F.linear(x, self.weight, self.bias) + self.lora_b(self.lora_a(x)) * self.scaling


class LoRAMLP(nn.Module):
    # torchtune apply_lora_to_mlp: LoRA on the MLP projections, residual like the other blocks here
    def __init__(self, d, h, r, alpha, bias):
        super().__init__()
        self.fc1 = LoRALinear(d, h, r, alpha, bias)
        self.fc2 = LoRALinear(h, d, r, alpha, bias)

    def forward(self, x):
        return x + self.fc2(F.silu(self.fc1(x)))


class Router(nn.Module):
    # torchtitan TokenChoiceTopKRouter: gate Linear(dim, E, bias=False) in fp32, softmax or sigmoid, top-k
    def __init__(self, d, E, topk, score, route_norm):
        super().__init__()
        self.gate = nn.Linear(d, E, bias=False)
        self.topk, self.score, self.route_norm = topk, score, route_norm

    def forward(self, x):
        s = self.gate(x)
        s = torch.sigmoid(s) if self.score == "sigmoid" else F.softmax(s, dim=-1)
        top, idx = torch.topk(s, k=self.topk, dim=-1)
        if self.route_norm:
            top = top / (top.sum(dim=-1, keepdim=True) + 1e-20)
        return top, idx


class Experts(nn.Module):
    # torchtitan GroupedExperts: w1 (E, h, d), w2 (E, d, h), w3 (E, h, d); SwiGLU per expert
    def __init__(self, d, h, E):
        super().__init__()
        self.w1 = nn.Parameter(torch.empty(E, h, d))
        self.w2 = nn.Parameter(torch.empty(E, d, h))
        self.w3 = nn.Parameter(torch.empty(E, h, d))
        for w in (self.w1, self.w2, self.w3):
            nn.init.trunc_normal_(w, std=0.3)

    def forward(self, x, counts):
        """x: tokens grouped by local expert; counts: python ints, tokens per local expert."""
        ws = [w.to_local() if hasattr(w, "to_local") else w for w in (self.w1, self.w2, self.w3)]
        w1, w2, w3 = ws
        outs = [(F.silu(xe @ w1[e].T) * (xe @ w3[e].T)) @ w2[e].T for e, xe in enumerate(torch.split(x, counts))]
        if outs:
            return torch.cat(outs, dim=0)
        # no local experts (uneven E under EP): keep the output in the autograd graph so this rank still joins
        # the combine all_to_all's backward like the others
        return x.new_zeros(0, w2.shape[1]) + w2.sum() * 0


class _A2A(torch.autograd.Function):
    """all_to_all_single with autograd through plain c10d; the backward swaps the splits."""

    @staticmethod
    def forward(ctx, x, out_splits, in_splits, group):
        import torch.distributed as dist

        ctx.splits, ctx.group = (out_splits, in_splits), group
        out = x.new_empty((sum(out_splits),) + tuple(x.shape[1:]))
        dist.all_to_all_single(out, x.contiguous(), out_splits, in_splits, group=group)
        return out

    @staticmethod
    def backward(ctx, g):
        import torch.distributed as dist

        out_splits, in_splits = ctx.splits
        gi = g.new_empty((sum(in_splits),) + tuple(g.shape[1:]))
        dist.all_to_all_single(gi, g.contiguous(), in_splits, out_splits, group=ctx.group)
        return gi, None, None, None


def all_to_all(x, out_splits, in_splits, group, impl):
    if impl == "functional":
        # torchtitan token_dispatcher uses _functional_collectives.all_to_all_single; this is its autograd form
        from torch.distributed._functional_collectives import all_to_all_single_autograd

        return all_to_all_single_autograd(x, out_splits, in_splits, group)
    return _A2A.apply(x, out_splits, in_splits, group)


class MoE(nn.Module):
    def __init__(self, d, b):
        super().__init__()
        self.E, self.topk = b["E"], b["topk"]
        self.router = Router(d, b["E"], b["topk"], b["score"], b["route_norm"])
        self.experts = Experts(d, b["eh"], b["E"])
        self.shared = None
        if b.get("shared"):
            self.shared = nn.Sequential(nn.Linear(d, b["eh"], bias=False), nn.SiLU(), nn.Linear(b["eh"], d, bias=False))
        self.ep = None  # (group, rank, world, a2a impl) under expert parallelism

    def forward(self, x):
        shp = x.shape
        flat = x.reshape(-1, shp[-1])
        top, idx = self.router(flat)
        fidx = idx.reshape(-1)
        order = torch.argsort(fidx, stable=True)
        counts = torch.bincount(fidx, minlength=self.E)
        if bool((counts == 0).any()):
            MOE_STATS["zero_tok"] = True
        tok = flat[order // self.topk]
        out = self.experts(tok, counts.tolist()) if self.ep is None else self._ep(tok, counts)
        out = out * top.reshape(-1)[order].unsqueeze(-1)
        y = torch.zeros_like(flat).index_add(0, order // self.topk, out)
        if self.shared is not None:
            y = y + self.shared(flat)
        return x + y.reshape(shp)

    def _ep(self, tok, counts):
        import torch.distributed as dist

        group, rank, W, impl = self.ep
        sizes = chunk_sizes(self.E, W)
        starts = [sum(sizes[:r]) for r in range(W)]
        in_splits = [int(counts[starts[r] : starts[r] + sizes[r]].sum()) for r in range(W)]
        if self.E % W == 0:
            # torchtitan _token_count_exchange: all_to_all of counts.view(ep, -1)
            recv = torch.empty_like(counts).view(W, -1)
            dist.all_to_all_single(recv, counts.view(W, -1).contiguous(), group=group)
        else:  # uneven experts per rank (Shard(0) chunk semantics): exchange every count
            allc = [torch.zeros_like(counts) for _ in range(W)]
            dist.all_gather(allc, counts, group=group)
            recv = torch.stack([c[starts[rank] : starts[rank] + sizes[rank]] for c in allc])
        out_splits = [int(v) for v in recv.sum(1)]
        if 0 in in_splits:
            MOE_STATS["zero_send"] = True
        if 0 in out_splits:
            MOE_STATS["zero_recv"] = True
        got = all_to_all(tok, out_splits, in_splits, group, impl)
        # received rows are ordered (source rank, local expert); experts want (local expert, source rank)
        nloc, off, spans = sizes[rank], 0, []
        for r in range(W):
            for e in range(nloc):
                n = int(recv[r, e])
                spans.append((e, off, n))
                off += n
        perm = [i for e in range(nloc) for (be, o, n) in spans if be == e for i in range(o, o + n)]
        p = torch.tensor(perm, dtype=torch.long)
        y = self.experts(got[p], [int(v) for v in recv.sum(0)] if nloc else [])
        inv = torch.empty_like(p)
        inv[p] = torch.arange(len(perm))
        return all_to_all(y[inv], in_splits, out_splits, group, impl)


def expert_params(model):
    return [p for mod in model.modules() if isinstance(mod, MoE) for p in mod.experts.parameters()]


def apply_ep(model, mesh, impl, group=None):
    """Experts Shard(0) on `mesh`; token exchange on `group` (torchtitan's EP group is its own process group)."""
    from torch.distributed.tensor import Shard, distribute_tensor

    for mod in model.modules():
        if isinstance(mod, MoE):
            for n in ("w1", "w2", "w3"):
                w = getattr(mod.experts, n)
                setattr(mod.experts, n, nn.Parameter(distribute_tensor(w.data, mesh, [Shard(0)])))
            mod.ep = (group or mesh.get_group(), mesh.get_local_rank(), mesh.size(), impl)


def apply_ac(model, mode):
    """Wrap every block like torchtitan's apply_ac (ptd_checkpoint_wrapper, preserve_rng_state=False)."""
    if mode in (None, "none"):
        return
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper

    kw = {}
    if mode == "sac":
        from torch.utils.checkpoint import CheckpointPolicy, create_selective_checkpoint_contexts

        save = {torch.ops.aten.mm.default, torch.ops.aten.addmm.default, torch.ops.aten.bmm.default}

        def policy(ctx, op, *a, **k):
            return CheckpointPolicy.MUST_SAVE if op in save else CheckpointPolicy.PREFER_RECOMPUTE

        kw["context_fn"] = lambda: create_selective_checkpoint_contexts(policy)
    for i in range(len(model.blocks)):
        model.blocks[i] = checkpoint_wrapper(model.blocks[i], preserve_rng_state=False, **kw)


def clip_titan(params, max_norm, norm_type, foreach):
    # torchtitan distributed/utils.py clip_grad_norm_ without the pipeline branch: group grads by mesh,
    # get_total_norm per group, full_tensor() of DTensor norms, combine, clip_grads_with_norm_ per group
    from torch.distributed.tensor import DTensor

    by_mesh = {}
    for p in params:
        if p.grad is not None:
            by_mesh.setdefault(p.grad.device_mesh if isinstance(p.grad, DTensor) else None, []).append(p)
    norms = []
    for ps in list(by_mesh.values()) or [[]]:
        n = torch.nn.utils.get_total_norm([p.grad for p in ps if p.grad is not None], norm_type, False, foreach)
        norms.append(n.full_tensor() if isinstance(n, DTensor) else n)
    if len(norms) == 1:
        total = norms[0]
    elif math.isinf(norm_type):
        total = torch.stack(norms).amax()
    else:
        total = sum(n**norm_type for n in norms) ** (1.0 / norm_type)
    for ps in by_mesh.values():
        torch.nn.utils.clip_grads_with_norm_(ps, max_norm, total, foreach)
    return total


def clip_accel(params, max_norm, norm_type, foreach):
    # accelerate Accelerator._clip_grad_norm_dtensor_aware (the FSDP2 path of Accelerator.clip_grad_norm_)
    from torch.distributed.tensor import DTensor

    by_mesh = {}
    for p in params:
        if p.grad is not None:
            by_mesh.setdefault(p.grad.device_mesh if isinstance(p.grad, DTensor) else None, []).append(p)
    norms = [torch.nn.utils.get_total_norm([p.grad for p in ps], norm_type, foreach=foreach) for ps in by_mesh.values()]
    norms = torch.stack([n.full_tensor() if isinstance(n, DTensor) else n for n in norms]) if norms else torch.zeros(1)
    total = torch.linalg.vector_norm(norms, norm_type)
    if max_norm != float("inf"):
        for ps in by_mesh.values():
            torch.nn.utils.clip_grads_with_norm_(ps, max_norm, total, foreach)
    return total


def clip_grads(params, c):
    """c: {"max", "norm", "foreach", "impl"}; returns the total norm as a float."""
    nt = float(c.get("norm", 2.0))
    fe = c.get("foreach")
    impl = c.get("impl", "torch")
    if impl == "titan":
        n = clip_titan(params, c["max"], nt, fe)
    elif impl == "accel":
        n = clip_accel(params, c["max"], nt, fe)
    else:
        n = torch.nn.utils.clip_grad_norm_(params, c["max"], norm_type=nt, foreach=fe)
    if hasattr(n, "full_tensor"):
        n = n.full_tensor()
    return float(n.item())


def has_empty_local(model):
    """True if this rank holds an empty local shard of a parameter that is not empty globally."""
    for p in model.parameters():
        if hasattr(p, "to_local") and p.numel() > 0 and p.to_local().numel() == 0:
            return True
    return False


def no_parallel(module, mesh):
    """torchtitan's NoParallel: replicate a module's parameters as DTensors on the TP mesh, take and return local
    tensors, so a TP model has no plain parameters left (mixed Tensor/DTensor lists break foreach ops)."""
    from torch.distributed.tensor import DTensor, Replicate, distribute_module

    def inp(mod, args, mesh):
        return tuple(
            DTensor.from_local(a, mesh, [Replicate()], run_check=False)
            if isinstance(a, torch.Tensor) and not isinstance(a, DTensor)
            else a
            for a in args
        )

    def outp(mod, out, mesh):
        return out.to_local() if isinstance(out, DTensor) else out

    return distribute_module(module, mesh, None, input_fn=inp, output_fn=outp)


def replicate_leftovers(model, mesh):
    """Apply no_parallel to every top-level unit (block, emb, head) that still holds plain parameters."""
    from torch.distributed.tensor import DTensor

    units = list(model.blocks) + [m for m in (getattr(model, "emb", None), model.head) if m is not None]
    for u in units:
        ps = list(u.parameters())
        if ps and not any(isinstance(p, DTensor) for p in ps):
            no_parallel(u, mesh)
