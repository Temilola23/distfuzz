# Contributing to distfuzz

## Development setup

```bash
git clone https://github.com/Temilola23/distfuzz.git
cd distfuzz
python -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[dev]"
```

Docker is needed for anything that starts more than one rank:

```bash
make docker        # builds distfuzz:cpu (python 3.12, torch 2.14.0 CPU)
make docker-test   # multi-rank tests inside the container
```

## Where to run what

| What | Where |
|---|---|
| Unit tests, reference model, generators, lint | host (`make check`) |
| Anything that calls `init_process_group` with more than one rank: multi-rank tests, fuzz campaigns, repros | Docker only, with `--memory 3g --cpus 4` or tighter |

Multi-rank Gloo runs crash on purpose (that is what the fuzzer looks for). On a laptop host a crashing rank can bring up OS crash dialogs and starve the machine, so keep them in a container. Tests that start ranks carry `@pytest.mark.multirank` and are skipped unless `DISTFUZZ_MULTIRANK=1`, which the image sets.

## Branches

| Branch | Purpose |
|---|---|
| `main` | Always green. Changes land through squash-merged PRs only. |
| `feature/<name>` | New fuzzer, oracle, API description or tool |
| `fix/<name>` | Bug fixes, including fuzzer false positives |
| `docs/<name>` | Documentation only |
| `ci/<name>` | CI, build, Docker |

Branch from `main`, keep one concern per branch, and rebase on `main` before asking for review.

## Commits

[Conventional commits](https://www.conventionalcommits.org/) with a scope:

```
feat(dtensor): add Partial(max) inputs to the generator
fix(collectives): mark ring racy when an async op is in flight
docs: explain the WRONG_RESULT oracle
ci: cache the torch wheel
```

Scopes: `collectives`, `faults`, `dtensor`, `dcp`, `repro`, `examples`. The body says why, not what; the diff already shows what.

## Code style

- `ruff check .` and `ruff format --check .` must pass (line length 120). `mypy src` must pass.
- Standard library first. A new dependency needs a sentence in the PR explaining why a few lines of code would not do.
- No module docstrings, section-banner comments or comments that repeat the code. A comment should explain a constraint the code cannot show, for example why a timeout has the value it has.
- Fuzzer code must stay deterministic for a given seed. Anything random takes an `rng` argument.
- Every rank must issue the same collectives in the same order unless the program deliberately diverges. An accidental desync in the harness looks exactly like a bug in PyTorch.

## Tests

- Tests live in `tests/<component>/`. Test and helper module names must be unique across `tests/` (spawned ranks re-import them by name), so prefix them: `test_dtensor_unit.py`, not `test_unit.py`.
- A change to an oracle or a reference model needs a regression test that shows the false positive or false negative it removes.
- Known limitations are `pytest.mark.xfail(strict=True)` tests that assert the desired behaviour, so fixing the limitation fails the suite and forces the marker to go.

## Findings

A fuzzer signature is not a bug report. Before a finding is called a bug in this repo (README table, `BACKLOG.md`, an issue):

1. It reproduces from a plain PyTorch script with no distfuzz imports, in a fresh container.
2. It is not a fuzzer artifact: reference model, input construction or tolerance.
3. It is not documented behaviour.
4. It was checked on the latest release and on a nightly, and the upstream tracker was searched for duplicates.

Use the "Fuzzer finding" issue template. Nothing gets filed on pytorch/pytorch from this repo without the maintainer's sign-off.

## Pull requests

Read [docs/PR_GUIDE.md](docs/PR_GUIDE.md). In short: fill in the template, keep the diff reviewable, paste real test counts, wait for green CI, squash-merge.
