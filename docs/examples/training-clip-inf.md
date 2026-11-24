# Check training configurations with the training profile

The training profile generates whole training runs the way production stacks set them up (torchtitan, torchtune, Hugging Face, accelerate, Lightning), runs each one sharded on 2 to 4 ranks, and compares every step with the same run on a single process: loss, gradient norm, parameters and optimizer state. Checkpoints are saved under one layout and loaded under another.

This example runs a 6-minute campaign, shows what it finds, shrinks the first finding to a two-rank configuration, and replays that configuration with one setting changed. The main bug is §30 in [Bugs found](../bugs.md): `clip_grad_norm_(norm_type=inf)` on an FSDP2 model fails on any rank that holds an empty shard of some parameter, and the other ranks wait in the all-reduce.

Time: about 25 minutes. Output below is from fresh runs on 24 November 2026 (torch 2.14.0 CPU, Docker with 4 CPUs), abridged where marked.

## 1. Run a campaign

```bash
docker run --rm --memory 3g --cpus 4 --shm-size 1g -e DCP_PG_TIMEOUT=600 -v "$PWD/runs:/src/runs" distfuzz:cpu \
  python -m distfuzz.dcp fuzz --profile train --minutes 6 --timeout 30 --seed 11 --out runs/train
```

`--timeout 30` is how long a training segment may run before the harness calls it hung and dumps every rank's stack. `DCP_PG_TIMEOUT=600` keeps the process group's own timeout out of the way, so a stuck collective is reported with its location.

```
[   45.9s] NEW TRAIN|PARTIAL_ERR|train:clip|fsdp2|clip:inf|clip-impl:accel|empty-shard|RuntimeError: _foreach_norm cannot compute the infinity norm on an empty tensor because th
[   81.2s] NEW TRAIN|STATE_MISMATCH|check|pp|attn|moe|p:weight
[  127.2s] NEW TRAIN|HANG|load|tp|load:sd:full_bcast|c10d_logger.py:wrapper > distributed_c10d.py:all_gather,state_dict.py:set_state_dict > distributed_c10d.py:broadcast
[  214.5s] NEW TRAIN|ERR|train:fwd|fsdp_tp|attn|accum|RuntimeError: a and b must have same reduction dim, but got [N, N] X [N, N].
[  244.1s] NEW TRAIN|ERR|load|tp->fsdp_tp|load:sd:full|empty-shard|AssertionError:
[  298.7s] NEW TRAIN|HANG|load|fsdp2|load:sd:full_bcast|c10d_logger.py:wrapper > distributed_c10d.py:all_gather,state_dict.py:set_state_dict > distributed_c10d.py:broadcast
[  357.4s] NEW TRAIN|HANG|load|hsdp|load:sd:full_bcast|c10d_logger.py:wrapper > distributed_c10d.py:all_gather,state_dict.py:set_state_dict > distributed_c10d.py:broadcast
{"t": 361.7, "execs": 113, "eps": 0.312, "segs": 209, "steps": 385, "cov": 20143, "corpus": 36, "sigs": 7, "restarts": 52, "kinds": {"PARTIAL_ERR": 1, "STATE_MISMATCH": 1, "HANG": 3, "ERR": 2}, "skipped": 0, "useful": 107, "known_execs": 6, "pred_run": 7, "pred_hit": 6, "unpred_known": 0}
```

113 training runs (209 segments, 385 steps) gave 7 signatures.

## 2. Read a training signature

A signature names the kind of failure, the phase it happened in, and the configuration axes that matter for that phase:

```
TRAIN | PARTIAL_ERR | train:clip | fsdp2 | clip:inf | clip-impl:accel | empty-shard | RuntimeError: _foreach_norm cannot ...
        kind          phase        parallelism  clip norm   clip implementation  a rank held an empty shard   message
```

| Kind | Meaning |
|---|---|
| `PARTIAL_ERR` | Some ranks raised and the others were stuck in a collective. In a real job this is a hang until the process group timeout, with the error only in one rank's log. |
| `ERR` | Every rank raised the same error. |
| `HANG` | The segment did not finish. The signature lists where each group of ranks is stuck, read from a stack dump: here rank 0 is in `set_state_dict > broadcast` and rank 1 in `all_gather`. |
| `LOSS_MISMATCH`, `GNORM_MISMATCH` | A step's loss or gradient norm differs from the single-process run. |
| `STATE_MISMATCH` | A parameter or optimizer state tensor differs from the single-process run after the segment. |
| `LOAD_MISMATCH`, `ROUNDTRIP_MISMATCH` | A checkpoint loaded into a fresh model does not restore the saved state. |

Each line of `runs/train/findings.jsonl` holds the signature, the finding, the full scenario that produced it, and the predictor's guess. Here is the triage of this run:

| Time | Signature (short) | Predicted before running | Verdict |
|---|---|---|---|
| 45.9 s | `PARTIAL_ERR` clip inf, FSDP2, empty shard | `K30-inf-clip-empty-shard` | §30 |
| 81.2 s | `STATE_MISMATCH` pipeline parallel, MoE | none | one parameter off by 4.6e-5; see below |
| 127.2 s, 298.7 s, 357.4 s | `HANG` in `set_state_dict` broadcast load, under TP, FSDP2 and HSDP | `K6-joint-bcast-load` | §6, see [the checkpoint example](checkpoint-broadcast-hang.md) |
| 214.5 s | `ERR` uneven tensor-parallel shards | `K150336-uneven-tp` | known upstream as #150336 |
| 244.1 s | `ERR` bare `AssertionError` loading a full state dict into FSDP2 + TP | `K32-strided-full-load` | §32 |

The `STATE_MISMATCH` is a single parameter off by 4.6e-5 in a run with an MoE block where some experts received no tokens. Small differences like this have been false positives before (Adam turns float rounding noise in a near-zero gradient into a full-size step), so replay it in float64 before calling it a bug.

## 3. Skip what is already known

`src/distfuzz/dcp/known.py` holds a predictor that looks at a scenario before it runs and names the known bugs it is likely to hit. Above, it named six of the seven findings in advance (`pred_hit: 6`). With `--known-rate 0.05`, a flagged scenario runs only 5 % of the time, so the campaign spends its budget elsewhere while regressions can still show up:

```bash
docker run --rm --memory 3g --cpus 4 --shm-size 1g -e DCP_PG_TIMEOUT=600 -v "$PWD/runs:/src/runs" distfuzz:cpu \
  python -m distfuzz.dcp fuzz --profile train --known-rate 0.05 --minutes 6 --timeout 30 --seed 11 --out runs/train-skip
```

```
[  165.5s] NEW TRAIN|STATE_MISMATCH|check|fsdp2|attn|moe|p:weight
{"t": 363.2, "execs": 239, "eps": 0.658, "segs": 459, "steps": 842, ..., "sigs": 1, "skipped": 20, "useful": 239, "known_execs": 0, ...}
```

Same seed and budget: 239 training runs instead of 113, and no time lost in known hangs. The only finding is the same small MoE parameter difference as before, now under FSDP2. Use the default `--known-rate 1` when you want to know whether a PyTorch build still has the known bugs, and `0.05` when you are looking for new ones.

## 4. Minimize the clipping finding

`minimize` replays each matching finding and shrinks the scenario (fewer segments, steps and blocks, smaller layers, fewer options) while the kind, phase and message stay the same:

```bash
docker run --rm --memory 3g --cpus 4 --shm-size 1g -e DCP_PG_TIMEOUT=600 -v "$PWD/runs:/src/runs" distfuzz:cpu \
  python -m distfuzz.dcp minimize runs/train/findings.jsonl runs/train-min "clip:inf"
```

It took 61 replays. The result in `runs/train-min/minimized.jsonl` is about as small as a training run gets:

```json
{"profile": "train", "tmpl": "accel-fsdp2", "dtype": "f64",
 "model": {"kind": "tok", "seq": 1, "d0": 1, "vocab": 1, "blocks": [], "out": 1, ...},
 "optim": {"name": "sgd", "lr": 0.001, "momentum": 0.9, ...},
 "clip": {"max": 1.0, "norm": "inf", "foreach": null, "impl": "torch"},
 "segs": [{"w": 2, "steps": 1, "par": "fsdp", "raf": true, ...}]}
```

An embedding of one token and a one-output head, FSDP2 on two ranks, one SGD step with inf-norm clipping through plain `torch.nn.utils.clip_grad_norm_`. With a single row, one of the two ranks holds an empty shard of each parameter.

`minimize` also writes `runs/train-min/TRAIN_PARTIAL_ERR_..._0.py`, a standalone script with the harness inlined and no distfuzz imports (exit code 1 means it reproduced):

```
DISTFUZZ_PHASE rank=0 phase=train:bwd
DISTFUZZ_PHASE rank=1 phase=train:bwd
DISTFUZZ_PHASE rank=0 phase=train:clip
DISTFUZZ_PHASE rank=1 phase=train:clip
DISTFUZZ_ERR rank=1 phase=train:clip msg=RuntimeError: _foreach_norm cannot compute the infinity norm on an empty tensor because the operation does not have an identity
DISTFUZZ_ERR rank=0 phase=train:clip msg=RuntimeError: [PATH:N] Timed out waiting Nms for recv operation to complete
```

Rank 1 raised inside the norm. Rank 0 had already entered the all-reduce of the norm and waited until Gloo's 30-second timeout.

## 5. Check one configuration

`python -m distfuzz.dcp run SCENARIO.json` replays a single scenario and exits with 1 if anything differs from the single-process run. Save the minimized scenario as `clip_inf.json`, copy it to `clip_2.json` with `"norm": 2`, and run both:

```bash
docker run --rm --memory 3g --cpus 4 --shm-size 1g -v "$PWD/runs:/src/runs" distfuzz:cpu \
  python -m distfuzz.dcp run runs/train-min/clip_2.json
```

```
DISTFUZZ_PHASE rank=0 phase=train:clip
DISTFUZZ_PHASE rank=1 phase=train:clip
DISTFUZZ_PHASE rank=0 phase=train:opt
...
segment 0: w=2 par=fsdp steps=1 -> ok
```

The 2-norm passes and matches the single-process run; `clip_inf.json` fails as above. This is the way to check your own setup: take a scenario close to it from `findings.jsonl` or `minimized.jsonl`, edit the fields below, and `run` it against the PyTorch build you use. [Install](../install.md#other-images) shows how to build the image for another release.

| Field | Values |
|---|---|
| `tmpl` | `titan-llama`, `titan-sp`, `titan-pp`, `titan-moe`, `mcore-fsdp2`, `hf-tp`, `tune-lora`, `accel-fsdp2`, `accel-ddp`, `accel-fp8`, `lightning-mp` (defined in `dcp/templates.py` with the source lines they copy) |
| `model.blocks` | list of `ln`, `attn`, `mlp`, `lin`, LoRA and MoE blocks with their sizes |
| `optim` | `sgd`, `adam`, `adamw`, with `foreach` or `fused` |
| `clip` | `max`, `norm` (2, 1, 3 or `"inf"`), `foreach`, `impl` (`torch`, `titan`, `accel`) |
| `accum`, `no_sync`, `ac` | gradient accumulation steps, whether to skip gradient sync between them, activation checkpointing (`none`, `full`, `sac` for selective) |
| `segs[i].w`, `segs[i].par` | world size and parallelism per segment: `ddp`, `fsdp`, `tp`, `fsdp_tp`, `hsdp`, `ep`, `pp`, `pp_fsdp` |
| `segs[i].save`, `segs[i+1].load` | how the checkpoint between two segments is written and read (sharded, full, `full_bcast`, async) |

The generator in `dcp/train_gen.py` is the authority on which combinations are valid; `TrainGen.valid()` rejects the rest.

## A shorter hand-written repro

`examples/repros/train/clip_inf_empty_shard.py` shows the bug without the harness: two `Linear` layers under `fully_shard` on 4 ranks, compared with a single process.

```bash
docker run --rm --memory 3g --cpus 4 --shm-size 1g distfuzz:cpu python examples/repros/train/clip_inf_empty_shard.py
```

```
out_features 8, norm_type inf, foreach True, rank 0 (local dim-0 rows [2, 2, 2, 2]): norm 4.0000, single process 4.0000
...
out_features 5, norm_type 2.0, foreach True, rank 3 (local dim-0 rows [0, 0, 2, 2]): norm 17.6241, single process 17.6241
...
out_features 5, norm_type inf, foreach True, rank 3 (local dim-0 rows [0, 0, 2, 2]): RuntimeError: _foreach_norm cannot compute the infinity norm on an empty tensor because t
out_features 5, norm_type inf, foreach True, rank 2 (local dim-0 rows [1, 1, 2, 2]): RuntimeError: Connection closed by peer [172.17.0.3]:11912. This is typically caused by a
...
out_features 5, norm_type inf, foreach False, rank 3 (local dim-0 rows [0, 0, 2, 2]): RuntimeError: linalg.vector_norm cannot compute the inf norm on an empty tensor because t
```

`Linear(8, 5)` on 4 ranks gives its weight 2, 2, 1 and 0 rows, because FSDP2 shards dim 0 with `torch.chunk` sizes. Rank 3 holds no rows and raises. The 2-norm of the same gradients is correct, and so is the inf-norm when every rank has rows.

## Why it matters

Inf-norm clipping is reachable through accelerate's and Lightning's `clip_grad_norm` options and through direct calls to `torch.nn.utils.clip_grad_norm_`. Small layers (a LoRA adapter, an MoE router gate with few experts, a small bias) leave empty FSDP2 shards as soon as the world size exceeds the layer's first dimension. The job does not crash at the bad line. One rank logs an error and the rest sit in an all-reduce until the process group times out.
