# Worked examples

Each example runs end to end with the exact commands shown. The outputs are from real runs; each page says when and how it was produced, and lines removed for length are marked with `...`.

| Example | What it shows | Fuzzer | Time |
|---|---|---|---|
| [Find and minimize a wrong all_reduce on Gloo](gloo-strided-allreduce.md) | A 5-minute campaign, picking a finding, shrinking it to two calls, a plain PyTorch repro with a reliability check | `distfuzz.collectives`, `distfuzz.repro` | 10 min |
| [Check training configurations with the training profile](training-clip-inf.md) | Training signatures, the known-bug predictor, minimizing a training run, replaying one configuration with a setting changed | `distfuzz.dcp --profile train` | 25 min |
| [Locate a DTensor hang with the watchdog](dtensor-hang.md) | Hang signatures that name the stuck loop or the missing ranks, and a five-step repro of an infinite loop | `distfuzz.dtensor` | 15 min |
| [A checkpoint load that hangs](checkpoint-broadcast-hang.md) | A `set_state_dict` broadcast load that returns on one rank without loading and hangs the next collective, from finding to standalone script | `distfuzz.dcp --profile train` | 20 min |
| [Two GPUs on Kaggle](kaggle-two-gpus.md) | NCCL collectives, DTensor on CUDA and the training profile on a free 2 x T4 notebook | all, with `--backend nccl` / `--device cuda` | 2.5 h |

All CPU examples use the `distfuzz:cpu` image from [Install](../install.md) and write to `./runs`. The commands mount that folder into the container with `-v "$PWD/runs:/src/runs"`.
