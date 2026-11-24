# Fuzzer details

This page describes what the training profile and the compile fuzzer generate, axis by axis. For how to run them, see the [Quickstart](quickstart.md) and the [command reference](cli.md).

## Training profile

`python -m distfuzz.dcp fuzz --profile train` generates training runs instead of checkpoint chains. It uses the DCP fuzzer's scenario format, worker ranks, reference and checkpoint oracles, and adds the parts of a training step that production stacks exercise and the classic profile does not.

Each scenario starts from a template in `src/distfuzz/dcp/templates.py`. A template fixes what one stack does (torchtitan dense and MoE, the Hugging Face TP plan, torchtune LoRA, accelerate with FSDP2 or DDP, Lightning's ModelParallelStrategy) and records where each choice comes from as `repo@commit:path:line`. The generator then draws sizes that are small or uneven relative to the world size (2 to 4 Gloo ranks), so FSDP2 leaves some ranks empty dim-0 shards and MoE routers send zero tokens to some experts.

| Axis | Values |
|---|---|
| Model | llama-style blocks (RMSNorm, attention, GELU MLP), MLP stacks, LoRA MLPs with a frozen base (torchtune `LoRALinear`), MoE with a top-k router and grouped experts (torchtitan), embedding plus head, tied or not |
| Parallelism | FSDP2 per block with `reshard_after_forward` and mixed precision, TP (`ColwiseParallel`/`RowwiseParallel`, NoParallel for the rest, a loss-parallel head), FSDP2 + TP on a 2-D mesh, HSDP, DDP, expert parallelism (`all_to_all_single` with uneven splits, functional or plain c10d) |
| Step | gradient accumulation with or without `no_sync` / `set_requires_gradient_sync`, activation checkpointing (full or selective), SGD/Adam/AdamW with foreach or fused |
| Clipping | `norm_type` 2, 1, 3 and inf, `foreach` on, off or default, through `torch.nn.utils.clip_grad_norm_`, torchtitan's per-mesh clip or accelerate's DTensor-aware clip |
| Checkpoint | `dcp.save`/`async_save` and `dcp.load` of sharded state dicts, full state dicts with `cpu_offload`, `set_state_dict` with `broadcast_from_rank0` (joint call, the split API torchtune and accelerate use, or model only), resumed under another world size or parallelism |

Signatures name the configuration axes that matter for the phase that failed, for example `TRAIN|PARTIAL_ERR|train:clip|fsdp2|clip:inf|clip-impl:torch|empty-shard|RuntimeError: _foreach_norm cannot compute the infinity norm on an empty tensor`. `PARTIAL_ERR` means some ranks raised and the others were stuck in a collective; `HANG` findings carry each rank's last phase and its stack location (`state_dict.py:set_state_dict > distributed_c10d.py:broadcast`), read from a faulthandler dump. `python -m distfuzz.dcp minimize` replays training scenarios on persistent worlds and shrinks them (fewer segments, steps and blocks, smaller layers, fewer axes) while the finding's kind, phase and message stay the same.

The second round of the profile added these axes. Each comes from a template with its provenance (`titan-sp`, `titan-pp`, `mcore-fsdp2`, `accel-fp8`, compile on the accelerate templates).

| Axis | Values |
|---|---|
| Sequence parallelism | `SequenceParallel` norms, colwise inputs gathered from `Shard(1)`, rowwise outputs reduce-scattered to `Shard(1)`, a sequence-sharded embedding output and lm_head input, optionally with the loss-parallel head (torchtitan's default TP plan) |
| Pipeline parallelism | `PipelineStage` model chunks that keep the model's FQNs, schedules GPipe, 1F1B, Interleaved1F1B, LoopedBFS, InterleavedZeroBubble, ZBVZeroBubble and DualPipeV, 2 to 4 stages or 2 x 2 PP x FSDP2, torchtitan's loss scaling and PP-aware clip; checkpoints through torchtitan's model and optimizer containers, so a run can resume between PP and non-PP layouts |
| `torch.compile` | per block before FSDP (`torch.compile(block)`), in place after FSDP (`Module.compile()`, accelerate's regional compile), or the whole model; `aot_eager`, and `inductor` when a C++ compiler is present (the `distfuzz:cpu-cc` image); each rank gets its own inductor cache directory |
| FSDP2 knobs | gradient divide factor 1 with the loss scaled by 1/dp, `set_unshard_in_backward(False)`, forward and backward prefetch chains, `fully_shard([last norm, head])`, deferred reduction across accumulation steps, `reshard_after_forward` as an int, `DeviceMesh.from_group`; 1 x 4 and 4 x 1 meshes |
| float8 | torchao `convert_to_float8_training` as accelerate applies it, with float8 all-gather and scale precompute; only when torchao is installed, and on GPUs only with compute capability 8.9 or newer |

`--known-rate R` runs a scenario that `dcp/known.py` predicts will hit a known bug (from its configuration alone) with probability R, so campaign time goes to new ground while regressions still show; `stats.jsonl` counts skipped, useful and known-only scenarios. `--device cuda` runs the workers on NCCL with one GPU per rank and a CPU reference; no Gloo group is created in that mode.

Set `DCP_PG_TIMEOUT` above `--timeout` so a stuck collective is reported by the harness with stack locations rather than as a Gloo timeout. The tests in `tests/dcp/test_train_multirank.py` check that the profile rediscovers the inf-norm clipping crash on FSDP2 and the `set_state_dict(broadcast_from_rank0)` hang, and that expert parallelism matches the single-process MoE.


## Compile-path fuzzing

`distfuzz.compile` looks for bugs that appear only when DTensor, FSDP2 or tensor parallel code runs under `torch.compile`. Inductor on CPU needs a C++ compiler, which `distfuzz:cpu` does not have; `docker/Dockerfile.cc` adds `g++` on top of any distfuzz image without changing it.

```bash
docker build -f docker/Dockerfile.cc --build-arg BASE=distfuzz:cpu -t distfuzz:cpu-cc docker
docker run --rm --cpus 3 --memory 3g --shm-size 1g -v "$PWD/runs:/src/runs" distfuzz:cpu-cc \
  python -m distfuzz.compile fuzz --minutes 45 --seed 1 --out runs/compile
python -m distfuzz.compile minimize runs/compile/findings.jsonl --index 3   # inside the container
python -m distfuzz.compile repro runs/compile/min_3.json --out repro.py      # plain PyTorch, run with torchrun
```

Two tracks share one loop:

| Track | Program | Legs |
|---|---|---|
| op (75%) | A DTensor generator program (same generator as `distfuzz.dtensor`, at most 14 steps) turned into one Python function: each `mk(j)` step becomes an argument, the other steps its body, the step outputs its return value. A config picks the backend (`aot_eager` or `inductor`), `fullgraph`, and `dynamic=False`, `dynamic=True` or `mark` (`maybe_mark_dynamic` on every sharded dim). | Plain-tensor reference; eager DTensor; compiled plain tensors (control); compiled DTensor, with backward when the program has one. Then a second call with fresh inputs of the same shape (recompiles, determinism) and a call on inputs where one sharded size is changed in every input that has it, without recompiling by hand. |
| module (25%) | A small model (embedding or linear input, 1-2 residual MLP blocks with LayerNorm, RMSNorm or none, a head) under FSDP2, TP (optionally sequence parallel) or FSDP2+TP on a 2x2 mesh, trained 2-4 steps with SGD or Adam. Compile per block before `fully_shard` (torchtitan's former layout) or the whole model after sharding. | Single process; eager parallel; compiled single process (control); compiled parallel. Losses per step and final parameters are compared. |

A difference counts only when eager DTensor matches the reference and the compiled plain-tensor control does not show the same difference, so inductor's own numerics and plain compile errors are filtered out. Outputs that backward does not start from are detached inside the function, because AOTAutograd guesses a DTensor tangent for every output that requires grad ([#172556](https://github.com/pytorch/pytorch/issues/172556)). Tensor-parallel dims in the module track are divisible by the TP degree, since uneven TP shards already fail in eager mode; FSDP2 shards stay uneven.

Signatures name the track, backend, dynamic mode, finding kind and op, for example `COMPILE|inductor|mark|MISMATCH|redistribute|dtype` or `MODULE|fsdp_tp|aot_eager|static|per_module|PARAM_MISMATCH|inp.weight|values`. `findings.jsonl` keeps the first program per signature and `failing.jsonl` keeps every failing program, so several hangs with one signature stay apart. Each rank writes the leg it is in to `logs/rank<r>.leg`, so a hang names its leg. Eager legs use a 30 s collective timeout and compiled legs 120 s, because ranks compile at different speeds. Each rank gets its own inductor cache directory: with one shared directory (the torchrun default) CPU ranks load each other's compiled DTensor graphs, which gives wrong values and crashes unrelated to the program; `--shared-cache` turns that back on to look for that class.

