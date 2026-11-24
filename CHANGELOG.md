# Changelog

All notable changes to this project are documented here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- C++ edge coverage for the collectives fuzzer: `docker/Dockerfile.cov` builds PyTorch v2.14.0 with SanitizerCoverage in Gloo and c10d, and `--signal cxx` reads the counters in every rank. Guided mode calibrates the harness's own noise at start, re-runs programs with new C++ features twice, and shrinks them before they enter the corpus.

- DTensor generator: a tail group that reaches about 100 more strategy-table ops (torch.special and prims special functions, the rest of the `_foreach` family, CPU fused Adam/AdamW, functional batch-norm variants, prims casts and views, `*_copy` views, fractional pooling with fixed samples, 3-D `grid_sample` and `max_unpool3d`); linalg, broadcasting `linalg.cross`, batched solves, more losses, integer matmul and zero-size reshapes.
- DTensor fuzzer: every HANG and CRASH program is kept in `hangs.jsonl`, up to `--keep-per-sig` per signature.

### Fixed
- DTensor fuzzer: every timeout had the signature `HANG||program timeout` and only the first program per signature was kept, so the `view_groups` hang in VERIFIED.md §19 was always hidden behind the earlier §3 one. A watchdog thread in each rank now reports the stuck DTensor loop (with the `view_groups` branch) or the collective and the ranks missing from it; CPU loops are cut after 5 s instead of the 30 s program timeout. A rank-divergent error that deadlocks in a collective is reported as `RANK_DIVERGENT_ERROR` with its message. The minimiser matches HANG signatures exactly.
- Collectives oracle: gathered list entries whose value the reference cannot know (for example a non-root `reduce` buffer) were compared against `None` and always reported `WRONG_RESULT`. Their dtype and shape are still checked.
- Collectives oracle: `WRONG_RESULT` findings were named after the last call that mentioned the tensor, even one that only read it. They now name the earliest call whose output is wrong, from the reference model's record of which collective last wrote each value.

### Changed
- License: MIT replaced with all rights reserved.
- README: key terms, diagrams for the three fuzzers and the finding-to-report pipeline, a guide to reading fuzzer output, and an annotated repository layout.

## [0.1.0] - 2026-09-27

### Added
- `distfuzz.collectives`: c10d/Gloo collective fuzzer with a single-process reference model, GUARD sentinel oracle, per-rank divergence, persistent rank executor and ddmin minimizer.
- `--fault` mode for the collectives fuzzer: lifecycle and fault ops, kill-survivor oracles, crash history and `replay`.
- `distfuzz.dtensor`: DTensor differential fuzzer (DTensor vs plain tensors) on 1-D and 2-D meshes.
- `distfuzz.dcp`: distributed checkpoint fuzzer that reloads checkpoints under a different world size and parallelism.
- `distfuzz.repro`: syzbot-style repro bundles with reliability runs, release bisection and a static dashboard.
- Plain PyTorch repros for the verified bugs in `examples/repros/`.
- CI: ruff, mypy, host tests on Python 3.10 and 3.12, multi-rank tests in Docker.
