"""Training-profile templates: what production stacks run, scaled down to tiny CPU models.

Each template fixes the choices a stack makes (block layout, parallelism, optimizer, clipping, accumulation,
checkpoint flow) and leaves sizes, world size and the remaining knobs to the generator. `SRC` records where each
choice comes from (repo@commit:path:line, read on Nov 8, 2026 from shallow clones). Values a user passes through a
stack's API (accelerate's and Lightning's norm_type, accelerate's no_sync accumulation) are drawn from what the API
accepts, not from the stack's default.
"""

SRC = {
    "titan.tp": "torchtitan@6062827 torchtitan/models/common/decoder_sharding.py:120-374 (colwise S(0) / rowwise S(1) "
    "projections, vocab-sharded embedding with Partial output, Replicate norms under SP, vocab-sharded lm_head "
    "output for the loss); mapped onto parallelize_module ColwiseParallel/RowwiseParallel",
    "titan.fsdp": "torchtitan@6062827 torchtitan/distributed/fsdp.py:205-235 (reshard_after_forward policy, "
    "MixedPrecisionPolicy param_dtype/reduce_dtype)",
    "titan.optim": "torchtitan@6062827 torchtitan/components/optim/optimizer.py:156-160 (AdamW lr 8e-4, betas "
    "(0.9, 0.95), eps 1e-8, weight_decay 0.1)",
    "titan.clip": "torchtitan@6062827 torchtitan/distributed/utils.py:615-712 (clip_grad_norm_: norm_type 2.0, "
    "grads grouped by mesh, get_total_norm per group)",
    "titan.accum": "torchtitan@6062827 torchtitan/trainer.py:219,355 (loop over gradient_accumulation_steps, "
    "no no_sync)",
    "titan.moe": "torchtitan@6062827 torchtitan/models/common/moe/token_dispatcher.py:244-353 (count exchange "
    "all_to_all of counts.view(ep, -1); input_splits / output_splits from per-rank sums; all_to_all_single for "
    "dispatch and combine with swapped splits)",
    "titan.ckpt": "torchtitan@6062827 torchtitan/components/checkpointer/dcp.py:318-334,409 (dcp.save / "
    "dcp.async_save / dcp.load of a Stateful app state)",
    "hf.tp": "transformers@a6f5c86 src/transformers/models/llama/configuration_llama.py:49-57 (q/k/v colwise, "
    "o rowwise, gate/up colwise, down rowwise; no sequence parallel)",
    "hf.moe": "transformers@a6f5c86 src/transformers/models/qwen3_moe/configuration_qwen3_moe.py:56-78 (attention "
    "colwise/rowwise; experts via ep_dispatch_experts)",
    "tune.lora": "torchtune@bd2a0fc torchtune/modules/peft/lora.py:99-100,146 (lora_a Linear(in, rank), lora_b "
    "Linear(rank, out), (alpha / rank) * lora_b(lora_a(x))); recipes/configs/llama3_1/8B_lora.yaml:31-35 (rank 8, "
    "alpha 16, LoRA on MLP)",
    "tune.fsdp": "torchtune@bd2a0fc torchtune/training/_distributed.py:661-714 (shard_model: fully_shard per layer "
    "with reshard_after_forward, root with reshard_after_forward=False)",
    "tune.ckpt": "torchtune@bd2a0fc torchtune/training/_distributed.py:336-388,532-567 "
    "(load_from_full_model_state_dict "
    "and load_from_full_optimizer_state_dict: set_*_state_dict(full_state_dict=True, broadcast_from_rank0=True); "
    "get_full_optimizer_state_dict full_state_dict=True, cpu_offload=True)",
    "tune.clip": "torchtune@bd2a0fc recipes/lora_finetune_distributed.py:115-117,792-795 (clip_grad_norm_ with "
    "max_norm from config, default norm_type 2; clip_grad_norm='inf' sets max_norm, not norm_type)",
    "accel.clip": "accelerate@f47439c src/accelerate/accelerator.py:2847-2905 (clip_grad_norm_(norm_type=2 "
    "default, user-passable) -> _clip_grad_norm_dtensor_aware for FSDP2)",
    "accel.accum": "accelerate@f47439c src/accelerate/accelerator.py:1093-1131 (no_sync: "
    "set_requires_gradient_sync(False) on FSDP2, DDP.no_sync)",
    "accel.load": "accelerate@f47439c cpu_ram_efficient_loading -> set_model_state_dict(full_state_dict=True, "
    "broadcast_from_rank0=True) (impact study, ../impact-2026-11/REPORT.md)",
    "lightning": "pytorch-lightning@84df182 src/lightning/fabric/strategies/model_parallel.py:74-100 "
    "(ModelParallelStrategy: user parallelize_fn, data_parallel_size x tensor_parallel_size, distributed or full "
    "checkpoint); Fabric.clip_gradients(norm_type) passes norm_type through",
}

# name -> fixed choices; the generator draws sizes, world size and the free knobs
TEMPLATES = {
    "titan-llama": dict(
        src=["titan.tp", "titan.fsdp", "titan.optim", "titan.clip", "titan.accum", "titan.ckpt"],
        kind="tok",
        layout="llama",
        pars=["fsdp", "fsdp", "fsdp_tp", "tp", "hsdp"],
        optim=dict(name="adamw", lr=8e-4, wd=0.1),
        fused=[True, None],
        clip=dict(max=[1.0], norm=[2.0], impl=["titan"]),
        accum=[1, 1, 2],
        no_sync=False,
        ac=["none", "full", "sac"],
        mp=0.25,
        lp=0.5,
        save=["stateful", "sd"],
        load=["sharded"],
        async_=0.4,
    ),
    "titan-moe": dict(
        src=["titan.moe", "titan.fsdp", "titan.optim", "titan.clip", "titan.ckpt"],
        kind="tok",
        layout="moe",
        pars=["ep", "ep", "fsdp"],
        optim=dict(name="adamw", lr=8e-4, wd=0.1),
        fused=[True, None],
        clip=dict(max=[1.0], norm=[2.0], impl=["titan"]),
        accum=[1, 2],
        no_sync=False,
        ac=["none", "full"],
        mp=0.0,
        lp=0.0,
        save=["stateful", "sd"],
        load=["sharded"],
        async_=0.3,
    ),
    "hf-tp": dict(
        src=["hf.tp", "accel.clip"],
        kind="tok",
        layout="llama",
        pars=["tp", "fsdp_tp"],
        optim=dict(name="adamw", lr=1e-3, wd=0.0),
        fused=[None, False],
        clip=dict(max=[1.0], norm=[2.0], impl=["accel", "torch"]),
        accum=[1, 2],
        no_sync=False,
        ac=["none"],
        mp=0.0,
        lp=0.0,
        save=["sd"],
        load=["sharded", "full"],
        async_=0.0,
    ),
    "tune-lora": dict(
        src=["tune.lora", "tune.fsdp", "tune.ckpt", "tune.clip"],
        kind="any",
        layout="lora",
        pars=["fsdp"],
        optim=dict(name="adamw", lr=3e-4, wd=0.01),
        fused=[None, True],
        clip=dict(max=[1.0, None], norm=[2.0], impl=["torch"]),
        accum=[1, 2],
        no_sync=False,
        ac=["none", "full"],
        mp=0.0,
        lp=0.0,
        save=["full_offload"],
        load=["model_bcast", "full_bcast_split"],
        async_=0.0,
    ),
    "accel-fsdp2": dict(
        src=["accel.clip", "accel.accum", "accel.load"],
        kind="any",
        layout="any",
        pars=["fsdp"],
        optim=dict(name="adamw", lr=1e-3, wd=0.0),
        fused=[None, False, True],
        clip=dict(max=[1.0, 0.1], norm=[2.0, 2.0, 1.0, float("inf")], impl=["accel"]),
        accum=[1, 2, 3],
        no_sync=True,
        ac=["none", "full"],
        mp=0.3,
        lp=0.0,
        save=["sd", "full_offload"],
        load=["sharded", "full_bcast_split", "model_bcast"],
        async_=0.0,
    ),
    "accel-ddp": dict(
        src=["accel.clip", "accel.accum"],
        kind="any",
        layout="any",
        pars=["ddp"],
        optim=dict(name="adamw", lr=1e-3, wd=0.0),
        fused=[None, False],
        clip=dict(max=[1.0, 0.1], norm=[2.0, 2.0, 1.0, float("inf")], impl=["torch"]),
        accum=[1, 2, 3],
        no_sync=True,
        ac=["none", "full"],
        mp=0.0,
        lp=0.0,
        save=["sd", "full_offload"],
        load=["sharded", "full"],
        async_=0.0,
    ),
    "lightning-mp": dict(
        src=["lightning"],
        kind="any",
        layout="any",
        pars=["fsdp", "fsdp_tp", "ddp"],
        optim=dict(name="adam", lr=1e-3, wd=0.0),
        fused=[None, False],
        clip=dict(max=[1.0, 0.5], norm=[2.0, 1.0, float("inf"), 3.0], impl=["torch"]),
        accum=[1, 2],
        no_sync=False,
        ac=["none"],
        mp=0.0,
        lp=0.0,
        save=["stateful", "sd", "full"],
        load=["sharded", "full"],
        async_=0.2,
    ),
}


def provenance(name):
    return [SRC[k] for k in TEMPLATES[name]["src"]]
