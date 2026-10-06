# Changelog

All notable changes to this project are documented here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Fixed
- Collectives oracle: gathered list entries whose value the reference cannot know (for example a non-root `reduce` buffer) were compared against `None` and always reported `WRONG_RESULT`. Their dtype and shape are still checked.

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
