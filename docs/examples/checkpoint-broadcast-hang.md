# A checkpoint load that hangs

Loading a full state dict with `broadcast_from_rank0=True` is the usual way to start a sharded job from a checkpoint that only rank 0 reads (torchtune and accelerate do this). This example follows one finding of the training profile from the campaign to a standalone script and a 40-line hand-written repro. The bug is §6 in [Bugs found](../bugs.md): when non-zero ranks pass the empty dicts that `get_state_dict(..., full_state_dict=True, cpu_offload=True)` gives them, `set_state_dict` returns on those ranks without loading anything, while rank 0 waits in a broadcast. The job then hangs at the next collective. Upstream, open issue [#157781](https://github.com/pytorch/pytorch/issues/157781) reports the hang without a diagnosis; the cause is given below.

Time: about 30 minutes, most of it the campaign from the [training example](training-clip-inf.md), which this page reuses. Output below is from fresh runs on 24 November 2026 (torch 2.14.0 CPU, Docker with 4 CPUs), abridged where marked.

## 1. The finding

The 6-minute campaign in the [training example](training-clip-inf.md#1-run-a-campaign) reported this hang three times, under tensor parallel, FSDP2 and HSDP:

```
[  127.2s] NEW TRAIN|HANG|load|tp|load:sd:full_bcast|c10d_logger.py:wrapper > distributed_c10d.py:all_gather,state_dict.py:set_state_dict > distributed_c10d.py:broadcast
[  298.7s] NEW TRAIN|HANG|load|fsdp2|load:sd:full_bcast|...
[  357.4s] NEW TRAIN|HANG|load|hsdp|load:sd:full_bcast|...
```

`load:sd:full_bcast` is the load mode: a full state dict through `set_state_dict` with `broadcast_from_rank0=True`. The finding in `runs/train/findings.jsonl` says where each rank was when the segment timed out:

```json
{"kind": "HANG", "phase": "load", "msg": "segment timeout",
 "detail": "rank0:load@state_dict.py:set_state_dict > distributed_c10d.py:broadcast; rank1:load@c10d_logger.py:wrapper > distributed_c10d.py:all_gather",
 "seg": 1, "par": "tp", "w": 2, "known": "K6-joint-bcast-load"}
```

Rank 0 is still inside `set_state_dict`, in a broadcast. Rank 1 has already returned from `set_state_dict` and reached the next collective, an `all_gather` that the harness issues to compare the loaded state. Rank 1 never received the model weights.

## 2. Minimize it

```bash
docker run --rm --memory 3g --cpus 4 --shm-size 1g -e DCP_PG_TIMEOUT=600 -v "$PWD/runs:/src/runs" distfuzz:cpu \
  python -m distfuzz.dcp minimize runs/train/findings.jsonl runs/train-min "HANG|load|tp|"
```

```
[2] TRAIN|HANG|load|tp|...: minimized in 45 runs -> TRAIN_HANG_load_tp_load_sd_full_bcast_c10d_logger_py_wrapper_distribut_2.py
```

Each replay of a hang waits for the segment timeout, so this took about 15 minutes. The original scenario trained an attention and MLP model with AdamW and gradient clipping for three steps across two segments. The minimized one (abridged) does no training at all:

```json
{"tmpl": "hf-tp", "model": {"kind": "tok", "d0": 4, "vocab": 11, "blocks": [], "out": 11, ...},
 "optim": {"name": "sgd", "momentum": 0.9, ...}, "clip": null,
 "segs": [
   {"w": 1, "steps": 0, "par": "ddp", "save": {"api": "sd", "mode": "sharded", ...}},
   {"w": 2, "steps": 0, "par": "tp", "load": {"api": "sd", "mode": "full_bcast", ...}}]}
```

Save an embedding and a head from one rank, then load the checkpoint as a full state dict with broadcast into the same model under tensor parallel on two ranks. The parallelism does not matter; the load path does.

## 3. Run the standalone script

The script that `minimize` wrote has the harness inlined and no distfuzz imports:

```bash
docker run --rm --memory 3g --cpus 4 --shm-size 1g -v "$PWD/runs:/src/runs" distfuzz:cpu \
  bash -c 'cd /tmp && python /src/runs/train-min/TRAIN_HANG_load_tp_load_sd_full_bcast_c10d_logger_py_wrapper_distribut_2.py'
```

```
DISTFUZZ_PHASE rank=0 phase=save
DISTFUZZ_PHASE rank=0 phase=post_save_barrier
segment 0: w=1 par=ddp steps=0 -> ok
DISTFUZZ_PHASE rank=1 phase=build
DISTFUZZ_PHASE rank=0 phase=build
DISTFUZZ_PHASE rank=0 phase=load
DISTFUZZ_PHASE rank=1 phase=load
DISTFUZZ_ERR rank=0 phase=load msg=RuntimeError: [PATH:N] Timed out waiting Nms for send operation to complete
segment 1: w=2 par=tp steps=0 -> [('FATAL', None, 'Traceback (most recent call last): ...
```

Rank 0 times out sending the broadcast after the script's 30-second Gloo timeout; exit code 1 means it reproduced.

## 4. The hand-written repro

`examples/repros/dcp/bcast_no_optim_state.py` is the same bug in 40 lines with nothing but `nn.Linear`, `fully_shard` and SGD. Run it with a time limit, because it hangs:

```bash
docker run --rm --memory 3g --cpus 4 --shm-size 1g distfuzz:cpu \
  bash -c 'timeout 90 python examples/repros/dcp/bcast_no_optim_state.py 2 0.9 1; echo "exit $?"'
```

```
rank1 fsdp=1 momentum=0.9: ok
exit 124
```

Rank 1 reports that `set_state_dict` succeeded. Rank 0 never returns, and `timeout` kills the job after 90 seconds. With `TORCH_DISTRIBUTED_DEBUG=DETAIL`, PyTorch checks that the ranks issue matching collectives and names the mismatch:

```
RuntimeError: Detected mismatch between collectives on ranks. Rank 1 is running collective: CollectiveFingerPrint(SequenceNumber=8OpType=BARRIER), but Rank 0 is running collective: CollectiveFingerPrint(SequenceNumber=0OpType=GATHER).Collectives differ in the following aspects: 	 Sequence number: 8vs 0  Op type: BARRIERvs GATHER
```

Passing the same data through the two single-purpose calls, `set_model_state_dict` and then `set_optimizer_state_dict`, works:

```bash
docker run --rm --memory 3g --cpus 4 --shm-size 1g -e SPLIT=1 distfuzz:cpu \
  python examples/repros/dcp/bcast_no_optim_state.py 2 0.9 1
```

```
rank0 fsdp=1 momentum=0.9: ok
rank1 fsdp=1 momentum=0.9: ok
```

## The cause

`set_state_dict` decides whether it is loading only optimizer state with `optim_only=not model_state_dict` (`torch/distributed/checkpoint/state_dict.py:1537` in v2.14.0). Under `broadcast_from_rank0`, non-zero ranks pass an empty model dict, so they conclude there is no model state to load and return early, while rank 0 enters the broadcast of the model state. The ranks issue different collectives. `set_model_state_dict` does not make that decision per rank, which is why the split calls work.

## Why it matters

On NCCL the stale ranks do not fail at the load. They continue into the first training step with the weights they were initialized with, and the job hangs at the first collective of that step, far from the line that caused it. The safe pattern today is the split API, which torchtune and accelerate already use.
