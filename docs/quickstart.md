# Quickstart

Both quickstarts assume you ran `make docker` (see [Install](install.md)) and are in the repository root. Results go to `./runs` on the host.

## A. Check your training setup

The training profile asks one question: does this PyTorch build train the same model to the same numbers when it is sharded as when it runs on one process? It generates training runs from templates copied from torchtitan, torchtune, Hugging Face, accelerate and Lightning, runs them under FSDP2, tensor parallel, pipeline parallel, HSDP, DDP or expert parallel on 2 to 4 ranks, saves and reloads checkpoints under a different layout, and compares every step with a single-process run.

### 1. Run it against your PyTorch version

Build the image with the release you train on (2.14.0 is the default):

```bash
docker build --build-arg TORCH_SPEC=torch==2.14.0 -t distfuzz:mine .
docker run --rm --memory 3g --cpus 4 --shm-size 1g -e DCP_PG_TIMEOUT=600 -v "$PWD/runs:/src/runs" distfuzz:mine \
  python -m distfuzz.dcp fuzz --profile train --minutes 10 --timeout 30 --out runs/train
```

Each new problem prints one line as it is found:

```
[   45.9s] NEW TRAIN|PARTIAL_ERR|train:clip|fsdp2|clip:inf|clip-impl:accel|empty-shard|RuntimeError: _foreach_norm cannot ...
[  127.2s] NEW TRAIN|HANG|load|tp|load:sd:full_bcast|...,state_dict.py:set_state_dict > distributed_c10d.py:broadcast
```

The fields after `TRAIN` are the kind of failure, the phase (`train:clip`, `load`), and the parts of the configuration that matter: here FSDP2, an inf-norm clip, and a rank that held an empty shard. If one of these matches how you train, you have a lead.

### 2. Check one configuration

`runs/train/findings.jsonl` stores the full scenario behind each finding. Write one to a file, change the fields that differ from your setup, and replay it:

```bash
python -c "import json; json.dump(json.loads(open('runs/train/findings.jsonl').readline())['scn'], open('runs/mine.json', 'w'), indent=1)"
# edit runs/mine.json: world size, parallelism, clip norm, optimizer, checkpoint load mode, ...
docker run --rm --memory 3g --cpus 4 --shm-size 1g -v "$PWD/runs:/src/runs" distfuzz:mine \
  python -m distfuzz.dcp run runs/mine.json
```

`segment 0: w=2 par=fsdp steps=1 -> ok` and exit code 0 mean every step matched the single-process run. Exit code 1 means a difference, and the output says which phase and which tensor. The [training example](examples/training-clip-inf.md#5-check-one-configuration) lists the scenario fields.

### 3. Spend the budget on new ground

`--known-rate 0.05` skips 95 % of the scenarios that the known-bug predictor expects to hit a bug already in [Bugs found](bugs.md). Use it for longer campaigns:

```bash
docker run --rm --memory 3g --cpus 4 --shm-size 1g -e DCP_PG_TIMEOUT=600 -v "$PWD/runs:/src/runs" distfuzz:mine \
  python -m distfuzz.dcp fuzz --profile train --known-rate 0.05 --minutes 60 --timeout 30 --out runs/train-60
```

## B. Fuzz PyTorch's distributed layer

These fuzzers test the layers under the training stack one API at a time, and each one can shrink a finding to a small repro.

```bash
docker run --rm -it --memory 3g --cpus 4 --shm-size 1g -v "$PWD/runs:/src/runs" distfuzz:cpu bash
```

Inside the container:

```bash
# c10d collectives on Gloo: wrong values, writes outside a tensor, crashes, hangs, rank desyncs
python -m distfuzz.collectives fuzz --mode random --minutes 5 --out runs/collectives
python -m distfuzz.collectives minimize runs/collectives/findings/<id>.json

# DTensor ops on 1-D and 2-D meshes, compared with plain tensors
python -m distfuzz.dtensor fuzz --world 4 --minutes 5 --out runs/dtensor
python -m distfuzz.dtensor minimize runs/dtensor/findings.jsonl --sig-substr HANG

# checkpoint save and load chains across FSDP2, TP, HSDP and DDP
python -m distfuzz.dcp fuzz --minutes 5 --out runs/dcp

# fault injection: drop a Work, abort a group, resize a tensor mid-collective, kill a rank
python -m distfuzz.collectives fuzz --fault --minutes 5 --out runs/faults
```

`torch.compile` over DTensor and FSDP2 needs a C++ compiler for inductor, so it uses its own image:

```bash
docker build -f docker/Dockerfile.cc --build-arg BASE=distfuzz:cpu -t distfuzz:cpu-cc docker
docker run --rm --memory 3g --cpus 3 --shm-size 1g -v "$PWD/runs:/src/runs" distfuzz:cpu-cc \
  python -m distfuzz.compile fuzz --minutes 10 --out runs/compile
```

On the host, `distfuzz.repro` turns a finding into a plain PyTorch script and checks it in fresh containers:

```bash
python -m distfuzz.repro images --only 2.14.0
python -m distfuzz.repro bundle runs/collectives/findings/<id>.json --name my-bug --n 5 --no-bisect
```

The [first worked example](examples/gloo-strided-allreduce.md) walks through this loop end to end. On GPUs, add `--backend nccl` to the collectives fuzzer and `--device cuda` to the DTensor and training fuzzers; see [Two GPUs on Kaggle](examples/kaggle-two-gpus.md).
