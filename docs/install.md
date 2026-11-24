# Install

distfuzz has two parts. The fuzzers start several PyTorch ranks and run inside a Docker image. The host tools (`distfuzz.repro`, unit tests, lint) run in an ordinary Python environment and start containers when they need ranks.

## Requirements

- Docker with at least 4 CPUs and 4 GB of memory available to containers.
- Python 3.10 or newer on the host, for the host tools.
- For GPU runs: two CUDA GPUs and NCCL, or a Kaggle notebook (see [Two GPUs on Kaggle](examples/kaggle-two-gpus.md)).

## Quick install

```bash
git clone https://github.com/Temilola23/distfuzz.git && cd distfuzz
make docker                 # builds distfuzz:cpu: Python 3.12, torch 2.14.0 CPU, distfuzz
pip install -e ".[dev]"     # host tools; install torch first if your environment has none
```

`make docker` takes a few minutes the first time. Check it with a 30-second run:

```bash
docker run --rm --memory 3g --cpus 4 --shm-size 1g distfuzz:cpu \
  python -m distfuzz.collectives fuzz --mode random --minutes 0.5 --out /tmp/smoke
```

The last line is a JSON summary with `"execs"` above zero.

On the host, `pip install torch --index-url https://download.pytorch.org/whl/cpu` gives a CPU build of torch if you need one. The host tools do not start ranks themselves, so any torch 2.4 or newer works there.

## Why Docker

The fuzzers make ranks crash, hang and abort on purpose. On a laptop host a crashing rank can bring up OS crash dialogs and starve the machine, and a hung rank can hold a port until it is killed. In a container with `--memory 3g --cpus 4` the damage stays inside, and `docker run --rm` cleans up.

Keep `--shm-size 1g`. Ranks pass tensors through shared memory, and Docker's default of 64 MB is too small.

## Other images

| Image | Build | Use |
|---|---|---|
| `distfuzz:cpu` | `make docker` | Everything on CPU with torch 2.14.0. |
| `distfuzz:cpu-cc` | `docker build -f docker/Dockerfile.cc --build-arg BASE=distfuzz:cpu -t distfuzz:cpu-cc docker` | Adds `g++` so `torch.compile(backend="inductor")` works on CPU. Needed for `distfuzz.compile` with inductor and for the compile axis of the training profile. |
| nightly | `docker build --build-arg TORCH_SPEC=torch --build-arg TORCH_INDEX=https://download.pytorch.org/whl/nightly/cpu -t distfuzz:nightly .` | Test a nightly build of PyTorch. |
| another release | `docker build --build-arg TORCH_SPEC=torch==2.10.0 -t distfuzz:2.10 .` | Test the release you run in production. |
| `distfuzz:cov` | `docker build -f docker/Dockerfile.cov -t distfuzz:cov .` | PyTorch built from source with C++ edge coverage in Gloo and c10d. Needs about 20 GB of memory and 1 to 2 hours. |

## Working inside the container

`make docker-shell` opens a shell in `distfuzz:cpu` with `./runs` mounted at `/src/runs`, so results survive the container:

```bash
make docker-shell
# inside
python -m distfuzz.collectives fuzz --mode random --minutes 5 --out runs/collectives
```

To run one command without a shell, use the same mount:

```bash
docker run --rm --memory 3g --cpus 4 --shm-size 1g -v "$PWD/runs:/src/runs" distfuzz:cpu \
  python -m distfuzz.dtensor fuzz --minutes 5 --out runs/dtensor
```

## Checking the install

```bash
make check          # host: ruff, mypy, unit tests (multi-rank tests are skipped)
make docker-test    # multi-rank tests inside distfuzz:cpu, about 10 minutes on 4 CPUs
```
