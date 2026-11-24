# Reading the output

Every fuzzer writes the same three things: a progress stream on stdout, one record per distinct finding, and per-rank logs. This page explains each. The [worked examples](examples/index.md) show them on real runs.

## Findings and signatures

A finding is anything the oracle flags. Findings with the same signature are counted together, and the first program that produced each signature is kept. A signature is a short `|`-separated key, for example:

```
WRONG_RESULT|all_reduce|torch.int64
HANG|cpu|_view_ops.py:view_groups:dst_exhausted
TRAIN|PARTIAL_ERR|train:clip|fsdp2|clip:inf|clip-impl:accel|empty-shard|RuntimeError: _foreach_norm cannot ...
```

Numbers inside messages are replaced by `N` or `#`, so the same error with different sizes groups together. One root cause can still produce several signatures (one per dtype or per parallelism), and two different bugs can share one. A signature is a lead. It becomes a bug once a minimized repro shows it on plain PyTorch.

## Collectives fuzzer

### Progress line

Printed every 30 seconds:

```
[random] t=92.0s execs=511 exec/s=5.55 cover=1436 corpus=0 findings=24 hang=0 crash=9 oom=0 checked=290
```

| Field | Meaning |
|---|---|
| `execs`, `exec/s` | programs run so far, and per second end to end |
| `cover` | distinct lines of `torch/distributed` Python code reached |
| `corpus` | programs kept for mutation (always 0 in random mode) |
| `findings` | distinct signatures so far |
| `hang`, `crash` | programs where a rank timed out or died |
| `oom` | programs killed by the container's memory limit |
| `checked` | programs whose every value the reference model could predict and compare |

The final JSON line adds the reference model's verdict on each program (`valid`, `invalid`, `uncertain`) and `restarts`, the number of times the rank processes were restarted after a crash or hang.

### Finding kinds

| Kind | Meaning |
|---|---|
| `WRONG_RESULT` | A legal program left a wrong value in some tensor on some rank. The signature names the collective whose output was wrong and the dtype. |
| `GUARD` | Sentinel cells around a tensor changed: the backend wrote outside the tensor it was given. |
| `VALID_ERROR` | A legal program raised. Errors that involved a collective timeout are re-run at `--confirm-timeout` and dropped if they do not recur. |
| `SILENT_ACCEPT` | An illegal program (ranks disagree on shapes, ops or group membership) ran without an error. Often by design for Gloo, so triage these last. |
| `CRASH` | A rank died with a signal or an uncaught C++ exception. |
| `HANG` | A rank did not finish the program before the session deadline and the ranks were restarted. |
| `XBACKEND_OUTCOME`, `XBACKEND_VALUE` | With `--cross-backend`, the two backends disagreed on whether the program succeeds, or on a value the reference specifies. |

The signature's second field for `CRASH`, `HANG` and `GUARD` is the reference model's verdict on the program (`valid`, `invalid`, `uncertain`). A crash on a `valid` program is the most interesting.

### Output folder

| Path | Contents |
|---|---|
| `findings/<id>.json` | one file per signature: kind, signature, the program, per-rank evidence |
| `findings/<id>.min.json`, `findings/<id>_repro.py` | written by `minimize`: the shrunk program and a script that replays it through distfuzz |
| `summary.json` | final statistics and every signature with its count, first time and finding id |
| `stats.jsonl` | one progress snapshot per line, for plotting |
| `corpus/` | programs that reached new coverage; guided mode mutates them |
| `logs/rank<r>.log` | per-rank logs from crashes and hangs |

## DTensor fuzzer

### Progress

Each new signature is printed as it appears, with a JSON statistics line every 30 seconds:

```
[   13.8s] NEW HANG|cpu|_view_ops.py:view_groups:src_exhausted
[   23.5s] NEW MISMATCH|mse_loss|values
{"t": 111.2, "execs": 363, "eps": 3.26, "steps": 6218, "cov": 4457, "corpus": 77, "sigs": 86, "restarts": 11, "kinds": {...}}
```

`kinds` counts every finding by kind, including `DIST_ERR` (DTensor raises where plain tensors succeed, almost always an op or redistribution DTensor does not support). `DIST_ERR` is not printed as `NEW`.

### Finding kinds

| Kind | Meaning |
|---|---|
| `MISMATCH` | `full_tensor()` of a step's output differs from the same step on plain tensors. The last field says whether values, dtype or shape differ. |
| `FINAL_MISMATCH` | A tensor differs at the end of the program, after later in-place ops. |
| `META` | The DTensor's global shape or dtype differs from the plain result. |
| `LOCAL_SHAPE` | A rank's local shard does not have the shape its placements imply. |
| `GRAD_MISMATCH` | A gradient differs after backward. |
| `RANK_DIVERGENT_ERROR` | A step raised on some ranks and not others. The other ranks would wait in the next collective. |
| `RANK_HASH_DIVERGENT` | `full_tensor()` returned different values on different ranks. |
| `FULL_TENSOR_ERR` | Gathering the result raised. |
| `HANG` | The watchdog found ranks stuck; see below. |
| `CRASH`, `FATAL` | A rank died, or the harness lost a rank's connection. |

### Hang signatures

A watchdog thread in each rank samples the stack while a step runs.

| Signature | Meaning |
|---|---|
| `HANG\|cpu\|FILE:FUNCTION[:BRANCH]` | Ranks are spinning inside a `torch.distributed` function. For known loops the branch is named from the frame's variables. |
| `HANG\|collective\|CALL\|ranks-missing:[...]` | Some ranks wait in collective `CALL`; the listed ranks never entered it. |
| `HANG\|\|program timeout` | No rank reported where it was before the program timeout. |

### Output folder

| Path | Contents |
|---|---|
| `findings.jsonl` | first program for each signature, with the finding and the world size |
| `hangs.jsonl` | up to `--keep-per-sig` programs per `HANG` or `CRASH` signature |
| `stats.jsonl`, `summary.json` | statistics and counts per signature |
| `logs/` | per-rank logs |

`minimize` writes `<signature>_<index>.py` (a plain PyTorch script that runs the shrunk program on a mesh and compares it with plain tensors) and `<signature>_<index>.json` to `--out`, plus `minimized.jsonl` with one line per finding.

## Checkpoint and training fuzzer

### Progress

```
[   45.9s] NEW TRAIN|PARTIAL_ERR|train:clip|fsdp2|clip:inf|clip-impl:accel|empty-shard|RuntimeError: _foreach_norm cannot ...
{"t": 361.7, "execs": 113, "eps": 0.312, "segs": 209, "steps": 385, "cov": 20143, "corpus": 36, "sigs": 7, "restarts": 52,
 "kinds": {...}, "skipped": 0, "useful": 107, "known_execs": 6, "pred_run": 7, "pred_hit": 6, "unpred_known": 0}
```

| Field | Meaning |
|---|---|
| `execs`, `segs`, `steps` | scenarios, segments (one world size and parallelism each) and training steps run |
| `skipped` | scenarios the known-bug predictor flagged and `--known-rate` skipped |
| `pred_run`, `pred_hit` | flagged scenarios that ran, and how many of them did hit the predicted bug |
| `known_execs`, `useful` | scenarios whose findings were all known bugs, and the rest |
| `unpred_known` | known bugs the predictor did not flag in advance |

### Finding kinds

| Kind | Meaning |
|---|---|
| `LOSS_MISMATCH`, `GNORM_MISMATCH` | A step's loss or gradient norm differs from the single-process run. |
| `STATE_MISMATCH` | Parameters or optimizer state differ from the single-process run at the end of a segment. The last field names the category, such as `p:weight` or `o:exp_avg`. |
| `LOAD_MISMATCH`, `ROUNDTRIP_MISMATCH` | A checkpoint loaded into a fresh, poisoned model does not restore what was saved. |
| `ERR` | Every rank raised. |
| `PARTIAL_ERR` | Some ranks raised while the others waited in a collective. |
| `HANG` | The segment timed out. The signature carries each rank's stack location, read from a faulthandler dump: `state_dict.py:set_state_dict > distributed_c10d.py:broadcast`. |

Training signatures read `TRAIN|KIND|PHASE|AXES...|MESSAGE`. The phase is where the failure happened (`build`, `train:fwd`, `train:bwd`, `train:clip`, `train:opt`, `save`, `load`, `check`). The axes are the configuration choices that matter for that phase, such as the parallelism (`fsdp2`, `tp->fsdp_tp` for a load that changes layout), the clip norm, or `empty-shard` when some rank held an empty shard.

### Output folder

| Path | Contents |
|---|---|
| `findings.jsonl` | per signature: the finding, the full scenario (`scn`), the predictor's guess (`pred`) and the time |
| `stats.jsonl`, `summary.json` | statistics, counts per signature, Python lines covered |
| `logs/w<N>/` | per-rank logs for each world size |

`minimize FINDINGS OUT_DIR` writes a standalone script per finding (the harness inlined, no distfuzz imports; exit code 1 means it reproduced) and `minimized.jsonl` with the shrunk scenarios. `run SCENARIO.json` replays one scenario and prints one line per segment, `-> ok` or the findings.

## Compile fuzzer

Signatures name the track, backend, dynamic mode, kind and op, for example `COMPILE|inductor|mark|MISMATCH|redistribute|dtype` or `MODULE|fsdp_tp|aot_eager|static|per_module|PARAM_MISMATCH|inp.weight|values`. A difference counts only when eager DTensor matches the reference and compiled plain tensors do not show the same difference. `findings.jsonl` keeps the first program per signature and `failing.jsonl` every failing program.

## Bundles

`python -m distfuzz.repro bundle` writes `bundles/<name>/`:

| File | Contents |
|---|---|
| `repro.py` | plain PyTorch script, no distfuzz imports |
| `Dockerfile`, `requirements.txt`, `run.sh` | pinned environment; `sh run.sh` builds it and runs the repro with the same limits |
| `bundle.json` | machine-readable results: title, minimization, reliability, per-release verdicts |
| `REPORT.md`, `report.html` | human-readable report: title, reliability such as `3/3 deterministic`, the repro, and the release table when bisected |

`python -m distfuzz.repro dashboard` writes `bundles/index.html` with every bundle.
