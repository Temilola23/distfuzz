# Changelog

All notable changes to this project are documented here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow [Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.1.0] - 2026-09-27

### Added
- `distfuzz.collectives`: c10d/Gloo collective fuzzer with a single-process reference model, GUARD sentinel oracle, per-rank divergence, persistent rank executor and ddmin minimizer.
- `distfuzz.dtensor`: DTensor differential fuzzer (DTensor vs plain tensors) on 1-D and 2-D meshes.
- `distfuzz.dcp`: distributed checkpoint fuzzer that reloads checkpoints under a different world size and parallelism.
- `distfuzz.repro`: syzbot-style repro bundles with reliability runs, release bisection and a static dashboard.
- Plain PyTorch repros for the verified bugs in `examples/repros/`.
- CI: ruff, mypy, host tests on Python 3.10 and 3.12, multi-rank tests in Docker.
