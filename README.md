# distfuzz

distfuzz checks whether PyTorch's distributed stack computes the same thing as a single process. You give it a PyTorch build; it generates distributed programs and whole training runs (FSDP2, tensor and pipeline parallel, HSDP, DDP, expert parallel, distributed checkpoints, `torch.compile`), runs each one on several ranks, and compares every value, gradient and checkpoint with the same computation on one process. When they differ, it shrinks the case to a few lines and writes a standalone PyTorch script you can run, share or attach to a bug report. So far it has found 44 verified bugs in PyTorch 2.14.0; for 34 of them a search of the PyTorch tracker found no earlier report.

Documentation: [temilola23.github.io/distfuzz](https://temilola23.github.io/distfuzz/) (the same pages are in [`docs/`](docs/)).

## Why

Bugs in distributed training rarely crash. A gradient comes back too large on every rank, or a checkpoint loads without an error on one rank and restores nothing there. Loss curves can still look plausible, so the framework is often the last thing anyone suspects.

Here is one. `F.mse_loss` on a sharded DTensor gives the right loss and twice the right gradient on 2 ranks:

```python
# torchrun --nproc-per-node 2 examples/repros/dtensor/mse_loss_grad.py
x, y = torch.tensor([2.0, 1.0]), torch.tensor([5.0, -1.0])
x2 = distribute_tensor(x, mesh, [Shard(0)]).requires_grad_()
F.mse_loss(x2, distribute_tensor(y, mesh, [Shard(0)])).backward()
```

```
loss   6.5 6.5
grad   [-3.0, 2.0] [-6.0, 4.0]
```

The first value on each line is from one process, the second from the sharded run. The gradient is multiplied by the world size, on torch 2.14.0 and on the November 2026 nightly, with no error or warning. distfuzz found it by comparing every gradient with a single-process run.

## Install

Multi-rank runs happen inside Docker, because the fuzzers make ranks crash and hang on purpose. You need Docker with 4 CPUs and 4 GB of memory, and Python 3.10 or newer for the host tools.

```bash
git clone https://github.com/Temilola23/distfuzz.git && cd distfuzz
make docker              # image distfuzz:cpu: Python 3.12, torch 2.14.0 CPU, distfuzz
pip install -e ".[dev]"  # host tools: repro bundles, tests, lint
```

[Install](docs/install.md) covers images for other PyTorch releases, nightlies, inductor and C++ coverage.

## Quickstart A: check your training setup

The training profile generates training runs from templates copied from torchtitan, torchtune, Hugging Face, accelerate and Lightning, and checks each step against a single process: loss, gradient norm, parameters, optimizer state, and checkpoints reloaded under a different layout.

```bash
docker build --build-arg TORCH_SPEC=torch==2.14.0 -t distfuzz:mine .   # the release you train on
docker run --rm --memory 3g --cpus 4 --shm-size 1g -e DCP_PG_TIMEOUT=600 -v "$PWD/runs:/src/runs" distfuzz:mine \
  python -m distfuzz.dcp fuzz --profile train --minutes 10 --timeout 30 --out runs/train
```

```
[   45.9s] NEW TRAIN|PARTIAL_ERR|train:clip|fsdp2|clip:inf|clip-impl:accel|empty-shard|RuntimeError: _foreach_norm cannot compute the infinity norm on an empty tensor ...
[  127.2s] NEW TRAIN|HANG|load|tp|load:sd:full_bcast|...,state_dict.py:set_state_dict > distributed_c10d.py:broadcast
```

The first line says: with FSDP2 and `clip_grad_norm_(norm_type=inf)`, a rank that holds an empty shard raises while the others wait. The second: a full-state-dict broadcast load under tensor parallel hangs, with rank 0 stuck in `set_state_dict`'s broadcast.

To check one configuration, take a scenario from `runs/train/findings.jsonl`, edit it to match your setup (world size, parallelism, optimizer, clip norm, checkpoint load mode), and replay it:

```bash
docker run --rm --memory 3g --cpus 4 --shm-size 1g -v "$PWD/runs:/src/runs" distfuzz:mine \
  python -m distfuzz.dcp run runs/mine.json      # exit 0: every step matched one process
```

[Worked example: training profile and clipping](docs/examples/training-clip-inf.md).

## Quickstart B: fuzz PyTorch's distributed layer

For PyTorch contributors and anyone who wants to test one layer at a time:

```bash
docker run --rm -it --memory 3g --cpus 4 --shm-size 1g -v "$PWD/runs:/src/runs" distfuzz:cpu bash

python -m distfuzz.collectives fuzz --mode random --minutes 5 --out runs/collectives   # c10d on Gloo
python -m distfuzz.collectives minimize runs/collectives/findings/<id>.json
python -m distfuzz.dtensor fuzz --world 4 --minutes 5 --out runs/dtensor                # DTensor ops
python -m distfuzz.dcp fuzz --minutes 5 --out runs/dcp                                  # checkpoint save/load
```

On the host, turn a finding into a plain PyTorch script and check how reliably it fails:

```bash
python -m distfuzz.repro images --only 2.14.0
python -m distfuzz.repro bundle runs/collectives/findings/<id>.json --name my-bug --n 5 --no-bisect
```

`torch.compile` fuzzing (`distfuzz.compile`) and fault injection (`--fault`) are in the [Quickstart](docs/quickstart.md).

## Reading the output

A finding is anything the oracle flags; findings are grouped by a signature such as `WRONG_RESULT|all_reduce|torch.int64`. Each one is saved with the program that produced it.

| You see | It means |
|---|---|
| `WRONG_RESULT`, `MISMATCH`, `STATE_MISMATCH`, `GNORM_MISMATCH` | a value differs from the single-process result, with no error |
| `GUARD` | the backend wrote outside the tensor it was given |
| `PARTIAL_ERR`, `RANK_DIVERGENT_ERROR` | some ranks raised while the others waited in a collective |
| `HANG\|cpu\|_view_ops.py:view_groups:...` | every rank is spinning in that function |
| `HANG\|collective\|wait_tensor\|ranks-missing:[2, 3]` | ranks 0 and 1 wait in a collective that ranks 2 and 3 never entered |
| `4 -> 2 calls; repro: .../<id>_repro.py` | the minimizer shrank the program from 4 calls to 2 |
| `reliability 3/3 deterministic` | the plain PyTorch repro failed in 3 of 3 fresh containers |

[Reading the output](docs/output.md) lists every kind and every file a run writes.

## Worked examples

| Example | Shows |
|---|---|
| [Gloo strided all_reduce](docs/examples/gloo-strided-allreduce.md) | a 5-minute campaign, minimizing a finding to 2 calls, a plain PyTorch repro |
| [Training profile and clipping](docs/examples/training-clip-inf.md) | training signatures, the known-bug predictor, replaying one configuration |
| [DTensor hang and the watchdog](docs/examples/dtensor-hang.md) | hang signatures that name the stuck loop, a 5-step repro of an infinite loop |
| [Checkpoint load hang](docs/examples/checkpoint-broadcast-hang.md) | a load that returns on one rank without loading, then hangs the next collective |
| [Two GPUs on Kaggle](docs/examples/kaggle-two-gpus.md) | NCCL collectives, DTensor on CUDA and the training profile on a free 2 x T4 notebook |

The outputs in the examples come from real runs.

## Bugs found

44 verified entries in PyTorch 2.14.0, each with a standalone script and a cause located at a source line. All but the two NCCL-only entries also reproduce on a November 2026 nightly. 34 matched no issue or pull request in the PyTorch tracker; 4 have a related pull request or an incomplete earlier fix; 6 extend an open issue. 22 return a wrong value with no error on any rank. Nothing has been filed upstream yet.

| Area | Entries | For example |
|---|---|---|
| Gloo, NCCL, c10d, CPU optimizer kernels | 9 | `all_reduce` on a strided view: wrong values on every rank (§1); NCCL `scatter`, `gather`, `reduce_scatter_tensor` skip the contiguity check (§75) |
| DTensor | 19 | `mse_loss` gradient times world size (§72); `flatten()` of a zero-size DTensor never returns (§3) |
| Checkpointing, FSDP2, TP, pipeline parallel, optimizers | 12 | zero-bubble pipeline schedules with resharding FSDP2 stages: weight gradients missing or wrong, no error (§33); inf-norm clipping with an empty FSDP2 shard hangs (§30); `set_state_dict(broadcast_from_rank0=True)` leaves ranks stale and then hangs (§6) |
| `torch.compile` over DTensor | 4 | CPU ranks share one AOTAutograd cache entry and load each other's graphs: wrong loss and gradients on every rank (§53) |

[Bugs found](docs/bugs.md) has the full list with causes, upstream status, impact ratings and repro scripts.

## GPUs

The collectives fuzzer runs on NCCL (`--backend nccl`), and the DTensor and training fuzzers run their ranks on CUDA (`--device cuda`), one GPU per rank, with a CPU reference. [`examples/kaggle/distfuzz_two_gpus.ipynb`](examples/kaggle/distfuzz_two_gpus.ipynb) runs all three on a free Kaggle 2 x T4 notebook; [Two GPUs on Kaggle](docs/examples/kaggle-two-gpus.md) explains the settings and the results of a recorded run.

## How it works

Each fuzzer generates programs from typed descriptions of an API, runs them on a set of persistent rank processes, and compares the result with an independent single-process computation: a reference model of every collective for c10d, the same program on plain tensors for DTensor, an unsharded training run for FSDP2 and checkpoints. Findings are deduplicated by signature, shrunk by delta debugging, and turned into standalone scripts. The design follows [syzkaller](https://github.com/google/syzkaller). [Architecture](docs/architecture.md) has the components, the mapping to syzkaller, the repository layout, and the current limitations. [Fuzzer details](docs/fuzzers.md) lists what the training profile and the compile fuzzer generate.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) and [docs/PR_GUIDE.md](docs/PR_GUIDE.md). `make check` runs lint, type checks and host tests; `make docker-test` runs the multi-rank tests.

## License

All rights reserved. No use, copying, modification or distribution without written permission; see [LICENSE](LICENSE).
