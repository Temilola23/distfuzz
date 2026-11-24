# Command reference

Every fuzzer is a Python module with subcommands. Anything that starts more than one rank runs inside the Docker image (see [Install](install.md)). `--help` on any subcommand prints the full list of flags; this page lists the ones you are likely to change.

## Common flags

The fuzzers share these flags.

| Flag | Meaning |
|---|---|
| `--minutes M` | Wall-clock budget. The fuzzer stops itself and writes its summary. |
| `--out DIR` | Output folder. Findings, statistics and per-rank logs go here. |
| `--seed S` | Random seed. The same seed and commit generate the same programs. |
| `--mode guided\|random` | Guided keeps programs that reach new code and mutates them; random generates every program fresh. |
| `--world N` | Number of ranks (`--world-size` is an alias in the collectives and DTensor fuzzers). |
| `--timeout S` | Seconds before a collective, program or segment counts as hung. |

## `distfuzz.collectives`

c10d collectives and point-to-point calls on Gloo or NCCL.

| Command | What it does |
|---|---|
| `fuzz --out DIR` | Run a campaign. |
| `minimize FINDING.json` | Shrink one finding to the fewest calls that still fail. Writes `FINDING.min.json` and `FINDING_repro.py` (a replay script that uses distfuzz). |
| `replay FINDING.json` | Re-run a crash or hang with the programs that ran before it. `--bisect` drops leading programs while it still reproduces. |
| `campaign --rounds R --out DIR` | Rounds that alternate random and guided mode; resumable. `make campaign` wraps it. |
| `bench --out DIR` | Measure executions per second with persistent ranks against a fresh world per program. |

`fuzz` flags beyond the common ones:

| Flag | Default | Meaning |
|---|---|---|
| `--backend gloo\|nccl` | `gloo` | Process group backend. `nccl` needs one GPU per rank. |
| `--device cpu\|cuda` | from the backend | Where tensors live. Only cpu with gloo and cuda with nccl are accepted. |
| `--cross-backend gloo\|nccl` | off | Also run each program on this backend and report where the two disagree (`XBACKEND_*` findings). |
| `--fault` | off | Add fault and lifecycle ops: drop a `Work`, abort a group, resize a tensor mid-collective, kill a rank. Gloo only. |
| `--gen-diverge P` | `0.25` | Chance that a generated program is rank-divergent (one rank skips or changes a call). |
| `--signal py\|cxx` | `py` | Coverage that admits programs to the corpus. `cxx` needs the `distfuzz:cov` image. |
| `--confirm-timeout S` | 4 on gloo, 15 on nccl | Re-run findings that involved a timeout at this longer timeout and drop those that do not recur. |
| `--detail` | off | Print per-rank values for each finding. |

## `distfuzz.dtensor`

DTensor ops on 1-D and 2-D meshes, compared with the same program on plain tensors.

| Command | What it does |
|---|---|
| `fuzz` | Run a campaign. |
| `minimize FINDINGS.jsonl` | Shrink findings and write repro scripts to `--out` (default `runs/dtensor-repros`). Pick findings with `--sig-substr TEXT` or `--index I`. |
| `report RUN_DIR...` | Summarize one or more runs by signature. |

`fuzz` flags beyond the common ones:

| Flag | Default | Meaning |
|---|---|---|
| `--device cpu\|cuda` | `cpu` | `cuda` runs DTensor over NCCL, one GPU per rank, and compares with a CPU reference. |
| `--no-compile` | off | Skip the `torch.compile` variant of each program. |
| `--keep-per-sig N` | `20` | Hang and crash programs kept per signature in `hangs.jsonl`. |
| `--hang-loop-secs S` | `5` | How long a rank may spin in a DTensor loop before the watchdog reports a hang. |

## `distfuzz.compile`

`torch.compile` over DTensor programs and small FSDP2 and TP models. Inductor on CPU needs a C++ compiler; build `distfuzz:cpu-cc` with `docker/Dockerfile.cc`.

| Command | What it does |
|---|---|
| `fuzz` | Run a campaign. |
| `minimize FINDINGS.jsonl --index I` | Shrink one finding; writes `min_I.json`. |
| `repro MIN.json --out repro.py` | Write a plain PyTorch script for a minimized finding; run it with `torchrun`. |
| `replay PROGRAMS.jsonl` | Re-run programs saved with `fuzz --save-all`. |

`fuzz` flags beyond the common ones:

| Flag | Default | Meaning |
|---|---|---|
| `--backends LIST` | `aot_eager,inductor` | Compile backends to draw from. |
| `--module-rate P` | `0.25` | Share of FSDP2 and TP model programs; the rest are DTensor op programs. |
| `--shared-cache` | off | Let ranks share one inductor cache directory, as `torchrun` does by default. |
| `--save-all` | off | Write every program to `programs.jsonl` for `replay`. |

## `distfuzz.dcp`

Distributed checkpointing, and with `--profile train`, whole training steps.

| Command | What it does |
|---|---|
| `fuzz` | Run a campaign. |
| `run SCENARIO.json` | Replay one scenario in fresh worlds. Exit code 1 means a finding reproduced. |
| `minimize FINDINGS.jsonl OUT_DIR [SIG_SUBSTRING...]` | Shrink findings and write standalone repro scripts (no distfuzz imports) to `OUT_DIR`. |
| `report RUN_DIR` | Sort a run's signatures into triage buckets. |

`fuzz` flags beyond the common ones:

| Flag | Default | Meaning |
|---|---|---|
| `--profile classic\|train` | `classic` | `classic` fuzzes checkpoint save and load chains; `train` generates training runs from production templates. |
| `--known-rate R` | `1.0` | Train profile: run a scenario that the known-bug predictor flags with probability R. `0.05` spends most of the budget on new ground. |
| `--max-w N` | `4` | Largest world size a scenario may use. |
| `--device cpu\|cuda` | `cpu` | `cuda` runs the workers on NCCL with one GPU per rank and a CPU reference. |

Set the environment variable `DCP_PG_TIMEOUT` above `--timeout` (for example `DCP_PG_TIMEOUT=600 ... --timeout 30`) so the harness reports a stuck collective with stack locations before the process group's own timeout fires.

## `distfuzz.repro`

Runs on the host and starts its own containers, one pinned image per PyTorch release.

| Command | What it does |
|---|---|
| `images [--only VERSION]` | Build the per-release images (2.4.1 to 2.14.0 and a pinned nightly). |
| `bundle SOURCE --name NAME` | Turn a finding JSON or a repro script into a bundle: plain PyTorch `repro.py`, reliability over `--n` fresh containers, bisection across releases (`--no-bisect` skips it). |
| `specs FILE` | Build every bundle listed in a specs file such as `examples/repro_specs.json`. |
| `reliability NAME` | Re-measure one bundle's reliability. |
| `dedup`, `dashboard` | Group bundles that are the same bug; write `bundles/index.html`. |
