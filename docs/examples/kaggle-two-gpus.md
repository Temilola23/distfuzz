# Run on two GPUs on Kaggle

The CPU image covers Gloo. NCCL, CUDA kernels and GPU-only code paths need GPUs, one per rank. Kaggle notebooks offer two Tesla T4 GPUs at no cost, which is enough for a world size of 2. This example runs the collectives fuzzer on NCCL, the DTensor fuzzer with CUDA meshes, and the training profile with NCCL workers, and shows how to read the results.

Two of the 44 bugs in [Bugs found](../bugs.md) were found this way and need NCCL to reproduce: §75 (NCCL `scatter`, `gather` and `reduce_scatter_tensor` skip the contiguity check that every other entry point makes) and §76 (equal-split `all_to_all_single` raises "NCCL operation in progress" under non-blocking communicators). Four more DTensor bugs, including §72, came from the GPU DTensor run and also reproduce on CPU.

## Output source

The outputs below are recorded from GPU fuzz run 2 on 15 November 2026 (Kaggle 2 x T4, torch 2.14.0+cu130, CUDA 13.0, NCCL 2.30.7, driver 580.178.04, Python 3.13), not from a fresh run for this page. That run used the same fuzzers from the `feat/gpu-backend` branch (commit `866518f`) packed into the notebook, ran two arms for 85 and 80 minutes, and set non-blocking NCCL communicators for every process. The notebook in this repository differs in three ways: it clones distfuzz from GitHub, it adds the training-profile arm, and it no longer sets non-blocking mode globally. The harness has also changed since (see [What changed since the recorded run](#what-changed-since-the-recorded-run)). It has not run on Kaggle in exactly this form yet. A dry run of cells 3 to 8 on CPU, with the arms switched to Gloo and CPU, ran end to end on 24 November and wrote the summary and the tarball.

## 1. Set up the notebook

1. Download [`examples/kaggle/distfuzz_two_gpus.ipynb`](https://github.com/Temilola23/distfuzz/blob/main/examples/kaggle/distfuzz_two_gpus.ipynb).
2. On [kaggle.com](https://www.kaggle.com), choose Create > New Notebook, then File > Import Notebook, and upload it.
3. In the notebook's settings panel, set Accelerator to GPU T4 x2 and turn Internet on. Kaggle asks for a verified phone number before it enables either.
4. Choose Save Version > Save & Run All (Commit). A committed run keeps going after you close the tab, and its outputs under `/kaggle/working` stay downloadable from the version's Output tab.

The budget is set in cell 3:

```python
MINUTES = {"collectives_nccl": 45, "dtensor_cuda": 45, "train_cuda": 45}
SEEDS = {"collectives_nccl": 111, "dtensor_cuda": 211, "train_cuda": 501}
```

Set an arm to 0 to skip it. With the defaults the run takes about 2.5 hours of Kaggle's weekly GPU quota. To repeat a run exactly, set `REF` in cell 2 to a commit hash instead of `main`.

## 2. What the cells do

| Cell | Does |
|---|---|
| 1 | Prints `nvidia-smi`, installs torch 2.14.0 for the driver's CUDA version (cu130 on driver 580 or newer, cu126 otherwise) and checks that two GPUs are visible. |
| 2 | Clones distfuzz and records the commit. The fuzzers import it through `PYTHONPATH`, so no `pip install` and no kernel restart. |
| 3 | Budget, seeds, and a runner that starts each fuzzer as a subprocess with a hard time cap and kills leftover rank processes so they do not hold the GPUs. |
| 4 | 30-second smoke tests of NCCL collectives and DTensor on CUDA. If NCCL fails, it retries once with `NCCL_P2P_DISABLE=1`, the usual fix on dual-T4 machines. |
| 5 | `python -m distfuzz.collectives fuzz --mode random --backend nccl --world-size 2 --gen-diverge 0.25` |
| 6 | `python -m distfuzz.dtensor fuzz --device cuda --world-size 2 --mode guided` |
| 7 | `python -m distfuzz.dcp fuzz --profile train --device cuda --max-w 2 --timeout 60 --known-rate 0.05`, with `DCP_PG_TIMEOUT=120` |
| 8 | Prints every arm's signatures and writes `/kaggle/working/gpufuzz.tgz`. |

The same commands work on any machine with two CUDA GPUs; drop the notebook and run them in a shell.

## 3. The smoke test

From the recorded run:

```
2.14.0+cu130 cuda 13.0 ['sm_75', 'sm_80', 'sm_86', 'sm_90', 'sm_100', 'sm_120'] gpus 2 Tesla T4 nccl (2, 30, 7)
...
[random] t=31.0s execs=22 exec/s=0.71 cover=1228 corpus=0 findings=0 hang=0 crash=0 oom=0 checked=15
{"execs": 22, "crash": 0, "hang": 0, "ok": 22, "valid": 15, ..., "nccl_unchecked": 0, "nccl_rejected": 2, ...}
```

`ok` above zero means NCCL works on this machine with peer-to-peer on. If both attempts fail, the cell stops with the path of the log to read.

## 4. Collectives on NCCL

The summary cell prints each signature with its count. From the recorded run (85 minutes, 1,559 programs):

```
## collectives_nccl: execs=1559 t=5105.7s ok=1507 crash=2 hang=50 checked=756 restarts=197
     23  nccl:VALID_ERROR|all_to_all_single|RuntimeError|NCCL Error #: NCCL operation in progress
     20  nccl:HANG|uncertain
      5  nccl:GUARD|uncertain|contig,expanded
      4  nccl:WRONG_RESULT|gather|list
      3  nccl:EXECUTOR_BUG|Blocking and nonblocking communicators are not allowed in the same group.
      3  nccl:WRONG_RESULT|scatter|torch.float32
      ...
      1  nccl:WRONG_RESULT|scatter|torch.int64
```

How these were triaged:

| Signature | Verdict |
|---|---|
| `VALID_ERROR\|all_to_all_single\|...NCCL operation in progress` (first after 139 s) | §76. A legal equal-split all-to-all raises on every rank under non-blocking communicators, although the output is correct. The same program on Gloo is fine. |
| `GUARD\|...\|contig,expanded`, `WRONG_RESULT\|gather\|list`, `WRONG_RESULT\|scatter\|...` | §75. NCCL `reduce_scatter_tensor` into an expanded output writes past it, and strided `scatter` and `gather` buffers give wrong values on non-root ranks. Gloo raises or is right on the same programs. |
| `HANG\|uncertain` | Harness artifact: after a rank-divergent program, a rank waited in `destroy_process_group` while its peer still had work pending. The executor now aborts these groups instead. |
| `EXECUTOR_BUG\|...` | Harness artifacts, kept under their own kind so they are not mistaken for PyTorch bugs. |

NCCL does not check that ranks agree on shapes or ops the way Gloo does, so programs the reference model calls invalid are counted in `nccl_unchecked` and not reported, and calls NCCL cannot run (`monitored_barrier`, `all_gather_coalesced`, bitwise reductions) are not generated.

## 5. DTensor on CUDA

The DTensor arm prints a JSON statistics line and the non-trivial signatures. Every `full_tensor()` result is moved to CPU and compared with a CPU reference, so this arm is also a CPU-versus-CUDA check. From the recorded run (80 minutes):

```
## dtensor_cuda: {'t': 4915.1, 'execs': 10636, 'eps': 2.16, 'steps': 178213, 'cov': 4626, 'corpus': 137, 'sigs': 261, 'restarts': 97, 'kinds': {'DIST_ERR': 7816, 'META': 513, 'LOCAL_SHAPE': 638, 'MISMATCH': 430, 'FINAL_MISMATCH': 83, 'HANG': 49, ..., 'GRAD_MISMATCH': 66, ...}}
   nccl:META|matmul|
   nccl:MISMATCH|reciprocal|values
   nccl:META|squeeze_|
   ...
```

Signatures from NCCL runs start with `nccl:`. In this run most value signatures were contamination: the §76 error in DTensor's `Shard(i) -> Shard(j)` redistribution left later programs on the same ranks with wrong values. Triage each value signature by replaying its program on CPU with the `distfuzz:cpu` image. §70 to §73, including the `mse_loss` gradient in the README, came out of the first GPU run's DTensor arm this way and reproduce on CPU.

## 6. Download and read the results

`gpufuzz.tgz` in the version's Output tab holds:

| Path | Contents |
|---|---|
| `env.txt` | torch, CUDA and NCCL versions, the driver, the distfuzz commit, the seeds and minutes |
| `SUMMARY.json` | final statistics and signatures for every arm |
| `collectives_nccl/` | `findings/<id>.json` per signature (with `backend` and `device` fields), `stats.jsonl`, `unconfirmed.jsonl` (timeouts that did not recur at the longer confirm timeout), per-rank `logs/` |
| `dtensor_cuda/`, `train_cuda/` | `findings.jsonl`, `stats.jsonl`, per-rank `logs/` |

`minimize` and `replay` in the collectives fuzzer run on Gloo only, so reproduce an NCCL finding by hand: copy the program from its finding file, check whether it also fails on Gloo with `python -m distfuzz.collectives minimize` in the CPU image, and if it does not, write a short script that runs the same calls on two GPUs.

## What changed since the recorded run

The recorded run exported `TORCH_NCCL_USE_COMM_NONBLOCKING=1` to every process. That bounded communicator setup and `wait()` and cut the time the collectives arm lost to hangs from about 86 % to about 30 % (collective timeouts that now raise cost roughly another quarter), but in the DTensor arm it made every equal-split all-to-all fail (§76), which contaminated later programs. Since then:

- The notebook leaves the variable unset, and the DTensor and training workers drop it even if it is set. The collectives executor still turns non-blocking mode on for its own groups, with a timeout derived from `--timeout`, so the collectives arm keeps the shorter hangs and can still find §76.
- On NCCL the collectives executor aborts each program's process groups instead of destroying them. The `HANG|uncertain` teardown hangs above came from `destroy_process_group`.
- The training profile no longer generates thread-based async checkpoint saves under `--device cuda` (they need a CPU process group, and the NCCL workers have none), and it uses float8 only where `torch._scaled_mm` runs, which on GPUs means compute capability 8.9 or newer. In the separate training-profile run of 15 November (two 80-minute arms on the same GPUs), these two harness artifacts made up 115 of 194 signatures. That run also rediscovered §6, §8 and §30 on NCCL.

These changes have passed the CPU test suite but have not yet run on GPUs; the next Kaggle run is their check.
