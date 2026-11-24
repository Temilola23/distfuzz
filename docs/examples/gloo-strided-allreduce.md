# Find and minimize a wrong all_reduce on Gloo

This example runs the collectives fuzzer for five minutes, picks one wrong-result finding, shrinks it to two calls, and turns it into a plain PyTorch script that you could attach to a bug report. The bug is §1 in [Bugs found](../bugs.md): Gloo's `all_reduce` on a strided view returns wrong values on every rank.

Time: about 10 minutes. Output below is from a fresh run on 24 November 2026 (torch 2.14.0 CPU, Docker with 4 CPUs), abridged where marked.

## 1. Run a campaign

```bash
make docker
docker run --rm --memory 3g --cpus 4 --shm-size 1g -v "$PWD/runs:/src/runs" distfuzz:cpu \
  python -m distfuzz.collectives fuzz --mode random --minutes 5 --out runs/collectives
```

The fuzzer prints a progress line every 30 seconds and a JSON summary at the end:

```
[random] t=31.5s execs=176 exec/s=5.59 cover=1412 corpus=0 findings=11 hang=0 crash=3 oom=0 checked=111
[random] t=62.0s execs=349 exec/s=5.63 cover=1436 corpus=0 findings=17 hang=0 crash=7 oom=0 checked=202
...
[random] t=274.3s execs=1450 exec/s=5.29 cover=1464 corpus=0 findings=44 hang=0 crash=27 oom=0 checked=785
{"execs": 1585, "crash": 30, "hang": 0, "ok": 1555, "valid": 789, "invalid": 219, "uncertain": 577, ..., "findings": 45}
```

In five minutes it ran 1,585 programs on 4 ranks and recorded 45 distinct signatures: 22 `WRONG_RESULT`, 9 `SILENT_ACCEPT`, 6 `GUARD`, 5 `VALID_ERROR` and 3 `CRASH`. [Reading the output](../output.md) explains each kind.

## 2. Pick a finding

`runs/collectives/summary.json` lists every signature with the time it first appeared and the id of its finding file:

```bash
python -c "
import json
s = json.load(open('runs/collectives/summary.json'))
for r in sorted(s['findings'], key=lambda r: r['first_t']):
    print(round(r['first_t'], 1), r['id'], r['sig'][:90])
"
```

```
1.4 3fe34963f5 GUARD|uncertain|contig,expanded,noncontig
1.4 d1cde0340b WRONG_RESULT|scatter|torch.float32
2.7 63351fa46a VALID_ERROR|all_gather_into_tensor|RuntimeError|ProcessGroupGloo::allgather: invalid tensor size
6.6 faf5bef293 CRASH|invalid|terminate called after throwing an instance of 'gloo::EnforceNotMet'
7.7 79af2e8047 WRONG_RESULT|all_reduce|torch.int64
...
```

`WRONG_RESULT|all_reduce|torch.int64` appeared after 7.7 seconds. Its finding file holds the program that triggered it and what the oracle saw:

```json
{"sig": "WRONG_RESULT|all_reduce|torch.int64", "kind": "WRONG_RESULT", "id": "79af2e8047",
 "detail": "mismatch on [(0, 't2'), (1, 't2'), (2, 't2'), (3, 't2')]",
 "info": {"ref": "valid", "racy": false, "checked": true, ...},
 "prog": {"world": 4, "calls": [
   {"op": "tensor", "args": {"spec": {"dtype": "uint8", "shape": [2, 3], "layout": "contig", ...}}, "rets": {"out": "t1"}},
   {"op": "all_reduce", "args": {"group": "world", "t": "t1", "op": "BOR", "async_op": false}},
   {"op": "tensor", "args": {"spec": {"dtype": "int64", "shape": [3], "layout": "noncontig", ...}}, "rets": {"out": "t2"}},
   {"op": "all_reduce", "args": {"group": "world", "t": "t2", "op": "SUM", "async_op": false}}]}}
```

`"ref": "valid"` means the reference model considers the program legal, so a mismatch is the backend's fault. The mismatch is on tensor `t2` on all four ranks.

## 3. Minimize it

```bash
docker run --rm --memory 3g --cpus 4 --shm-size 1g -v "$PWD/runs:/src/runs" distfuzz:cpu \
  python -m distfuzz.collectives minimize runs/collectives/findings/79af2e8047.json
```

```
4 -> 2 calls; repro: runs/collectives/findings/79af2e8047_repro.py
```

The minimized program in `79af2e8047.min.json` keeps two calls: build a 3-element int64 tensor as a strided (`noncontig`) view, then `all_reduce` it with `SUM`. The written `_repro.py` replays that program through distfuzz:

```
exec kind: ok
reference: valid
WRONG_RESULT|all_reduce|torch.int64
mismatch on [(0, 't2'), (1, 't2'), (2, 't2'), (3, 't2')]
```

## 4. Turn it into a plain PyTorch repro

`distfuzz.repro` runs on the host. It writes a script with no distfuzz imports, checks it still fails, and measures how reliably it fails in fresh containers:

```bash
pip install -e ".[dev]"                             # once, on the host
python -m distfuzz.repro images --only 2.14.0       # once: pinned image per torch release
python -m distfuzz.repro bundle runs/collectives/findings/79af2e8047.json \
  --name strided-allreduce --n 3 --no-bisect
```

```
[strided-allreduce] reproduce on torch 2.14.0
[strided-allreduce] title: c10d/gloo: wrong result in all_reduce
[strided-allreduce] minimize: {'done': True, 'units_before': 4, 'units_after': 2, 'tests': 3, 'verified': True}
[strided-allreduce] reliability x3 on torch 2.14.0
[strided-allreduce] reliability 3/3 deterministic other={}
```

`bundles/strided-allreduce/` now holds `repro.py`, a pinned `Dockerfile`, `run.sh`, `REPORT.md` and `report.html`. Drop `--no-bisect` to also test every release from 2.4.1 to the nightly; that builds one image per release the first time and takes much longer.

```bash
sh bundles/strided-allreduce/run.sh
```

```
all_reduce: MISMATCH rank 0 var t2
   expected (3,) torch.int64: [-7, -1, -6]
   actual   (3,) torch.int64: [-7, -1, -4]
all_reduce: MISMATCH rank 1 var t2
   expected (3,) torch.int64: [-7, -1, -6]
   actual   (3,) torch.int64: [-7, -1, -2]
...
[distfuzz-repro] DONE torch 2.14.0+cpu
```

(The ranks' lines interleave in the raw output; they are separated here.)

## What the output means

Every rank should hold the sum `[-7, -1, -6]`. The first two elements are right and the third is each rank's own input. The view has stride 2, so its elements sit at positions 0, 2 and 4 of the underlying buffer. Gloo reduces the first three contiguous elements starting at position 0, which are the first element, a gap, and the second element. The third element at position 4 never takes part, so every rank keeps its local value. The same root cause gives wrong results for `reduce_scatter` and writes into the gaps of the view.

## A shorter hand-written repro

`examples/repros/collectives/gloo_strided_allreduce.py` shows the same bug in 30 lines, including the case where only one rank passes a strided tensor and all four ranks get wrong values:

```bash
docker run --rm --memory 3g --cpus 4 --shm-size 1g distfuzz:cpu \
  torchrun --nproc-per-node 4 examples/repros/collectives/gloo_strided_allreduce.py
```

```
noncontig=[] async=False other_before_wait=False        OK
noncontig=[] async=True other_before_wait=False         OK
noncontig=[] async=True other_before_wait=True          OK
noncontig=[0] async=False other_before_wait=False       WRONG on ranks [0, 1, 2, 3]
noncontig=[0] async=True other_before_wait=False        WRONG on ranks [0, 1, 2, 3]
noncontig=[0] async=True other_before_wait=True         WRONG on ranks [0, 1, 2, 3]
noncontig=[0, 1, 2, 3] async=False other_before_wait=False WRONG on ranks [0, 1, 2, 3]
...
```

## Triage the rest

Signatures key on the operation and dtype, so one root cause shows up as several signatures: here `all_reduce` alone has seven, one per dtype. Before you call another signature a separate bug, minimize it and check whether it needs a strided or expanded tensor. The `GUARD` findings are the sentinel oracle reporting writes outside a view. The layouts in their signatures (`noncontig`, `expanded`) point at the same class of bug seen from the memory side.
