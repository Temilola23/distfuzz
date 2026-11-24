# Locate a DTensor hang with the watchdog

A hang in distributed code usually surfaces as a collective timeout many minutes later, on a rank that did nothing wrong. This example runs the DTensor fuzzer for five minutes and shows how its per-rank watchdog names each hang by where the ranks are stuck, so two different hangs get two different signatures. It then minimizes one of them to a five-line program. The bug is §3 in [Bugs found](../bugs.md): `flatten()` of a zero-size DTensor loops forever inside sharding propagation.

Time: about 15 minutes, most of it minimization. Output below is from a fresh run on 24 November 2026 (torch 2.14.0 CPU, Docker with 4 CPUs, 4 ranks on a 2 x 2 mesh), abridged where marked.

## 1. Run a campaign

```bash
docker run --rm --memory 3g --cpus 4 --shm-size 1g -v "$PWD/runs:/src/runs" distfuzz:cpu \
  python -m distfuzz.dtensor fuzz --mode guided --world 4 --minutes 5 --seed 3 --out runs/dtensor
```

The DTensor fuzzer prints each new signature as it appears, and a JSON statistics line every 30 seconds:

```
[    0.1s] NEW MISMATCH|/|values
[    1.2s] NEW META|@|
...
[   13.8s] NEW HANG|cpu|_view_ops.py:view_groups:src_exhausted
[   23.5s] NEW MISMATCH|mse_loss|values
...
[   37.6s] NEW RANK_DIVERGENT_ERROR|max|RuntimeError: max(): Expected reduction dim to be specified for input.numel() == N. ...
...
[  128.1s] NEW HANG|cpu|_view_ops.py:view_groups:dst_exhausted
[  231.6s] NEW HANG|collective|wait_tensor|ranks-missing:[2, 3]
[  254.8s] NEW HANG|collective|wait_tensor|ranks-missing:[0, 1, 2]
{"t": 300.9, "execs": 753, "eps": 2.5, "steps": 12794, "cov": 4575, "corpus": 92, "sigs": 140, "restarts": 25, "kinds": {"DIST_ERR": 569, "MISMATCH": 94, ..., "RANK_DIVERGENT_ERROR": 32, "HANG": 5}}
```

## 2. Read the hang signatures

Each rank runs a watchdog thread that samples the rank's stack while a program step runs. It reports one of two situations:

| Signature | Meaning |
|---|---|
| `HANG\|cpu\|FILE:FUNCTION:BRANCH` | A rank has spent more than `--hang-loop-secs` (default 5 s) inside one `torch.distributed` function without returning. Every rank is computing; nobody is waiting on anybody. For known loops the signature adds which branch is spinning, read from the frame's local variables. |
| `HANG\|collective\|CALL\|ranks-missing:[...]` | Some ranks are blocked in a collective (`CALL`) that the listed ranks never entered. The listed ranks finished the step, or raised, and moved on. |

A program step that raises on some ranks while the others block in a collective is reported as `RANK_DIVERGENT_ERROR` with the error message, which tells you why the ranks parted ways.

The first two hangs above are both in `view_groups`, the function in `torch/distributed/tensor/_ops/_view_ops.py` that works out how a reshape maps input dims to output dims. They spin in different branches (`src_exhausted` and `dst_exhausted`), so the fuzzer keeps them apart. Before the watchdog existed, every hang had the signature `HANG||program timeout`, and the second one stayed hidden behind the first.

The finding records the step that hung and what each rank was doing:

```json
{"sig": "HANG|cpu|_view_ops.py:view_groups:dst_exhausted",
 "finding": {"kind": "HANG", "step": 16, "expr": "v15.flatten(0, 1)", "secs": 5.38,
             "ranks": ["cpu:_view_ops.py:view_groups:dst_exhausted", "cpu:_view_ops.py:view_groups:dst_exhausted",
                       "cpu:_view_ops.py:view_groups:dst_exhausted", "cpu:_view_ops.py:view_groups:dst_exhausted"]},
 "prog": {"mesh": "2d", "inputs": [...], "steps": [... 19 steps ...]}, "t": 128.1, "world": 4}
```

All four ranks are in the same loop, at step 16, `v15.flatten(0, 1)`.

The last hang reads differently. `HANG|collective|wait_tensor|ranks-missing:[0, 1, 2]` at `torch.addmv(...)` means rank 3 issued a collective for that op and ranks 0 to 2 did not, so the ranks chose different ways to compute the same op. That is the pattern of §22 (ops without a sharding strategy of their own pick one in hash order). Minimize it before treating it as that bug.

## 3. Minimize the hang

```bash
docker run --rm --memory 3g --cpus 4 --shm-size 1g -v "$PWD/runs:/src/runs" distfuzz:cpu \
  python -m distfuzz.dtensor minimize runs/dtensor/findings.jsonl --sig-substr view_groups --out runs/dtensor-repros
```

The minimizer keeps the signature fixed, including the branch, and prints one JSON line per finding (abridged):

```
{"index": 27, "sig": "HANG|cpu|_view_ops.py:view_groups:src_exhausted", "status": "ok", "runs": 38,
 "steps": ["mk(0)", "v10.reshape((0, 5, 5))", "F.max_pool1d(v12, 2, return_indices=True)[1]", "mk(1)", "v13 @ v17"], "orig_steps": 19, ...}
{"index": 92, "sig": "HANG|cpu|_view_ops.py:view_groups:dst_exhausted", "status": "ok", "runs": 35,
 "steps": ["mk(0)", "v1.unflatten(0, (3, 2))", "mk(1)", "F.embedding(v14, v3)", "v15.flatten(0, 1)"], "orig_steps": 19,
 "inputs": [{"shape": [6], "dtype": "f32", "pl": ["R", "R"], ...}, {"shape": [0], "dtype": "i64", "kind": "index", "pl": ["R", "R"], ...}]}
```

The second one went from 19 steps to 5. An embedding lookup with an empty index tensor gives a DTensor of shape `(0, 2)`, and flattening it never returns. Each minimized program is also written as a standalone script (`runs/dtensor-repros/HANG_cpu_view_ops_py_view_groups_dst_exhausted_92.py`) that compares the DTensor run with plain tensors. It imports only torch. Run it with a time limit, since it hangs:

```bash
docker run --rm --memory 3g --cpus 4 --shm-size 1g -v "$PWD/runs:/src/runs" distfuzz:cpu \
  bash -c 'timeout 25 python runs/dtensor-repros/HANG_cpu_view_ops_py_view_groups_dst_exhausted_92.py; echo "exit $?"'
```

```
exit 124
```

Exit code 124 is `timeout` killing it after 25 seconds with no output.

## 4. Confirm with the hand-written repro

`examples/repros/dtensor/cases.py flatten_zero_size` builds a zero-size DTensor directly and dumps every rank's stack after 20 seconds:

```bash
docker run --rm --memory 3g --cpus 4 --shm-size 1g distfuzz:cpu \
  python examples/repros/dtensor/cases.py flatten_zero_size
```

```
Timeout (0:00:20)!
Thread 0x0000ffff854ed4c0 (most recent call first):
  File ".../torch/distributed/tensor/_ops/_view_ops.py", line 483 in view_groups
  File ".../torch/distributed/tensor/_ops/_view_ops.py", line 648 in <lambda>
  File ".../torch/distributed/tensor/_ops/_view_ops.py", line 1385 in reshape_strategy
  File ".../torch/distributed/tensor/_sharding_prop.py", line 789 in propagate_op_sharding_non_cached
  File ".../torch/distributed/tensor/_dispatch.py", line 299 in _propagate_op_sharding_dispatch_slow_path
  File "/src/examples/repros/dtensor/cases.py", line 171 in flatten_zero_size
  ...
```

The stack ends in the same function the watchdog named.

## Why it matters

A zero-size shard is ordinary in training: an empty batch on one rank, a MoE expert that received no tokens, an index tensor that a filter emptied. On a real job this loop pins every rank at 100 % CPU, and the process group timeout never fires because no rank is waiting in a collective. The watchdog signature turns "the job stopped making progress" into a file, a function and a branch.
