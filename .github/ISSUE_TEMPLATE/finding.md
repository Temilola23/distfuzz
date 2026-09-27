---
name: Fuzzer finding (in PyTorch)
about: Track a behaviour the fuzzer found in torch.distributed, DTensor or DCP
title: "[Finding] "
labels: finding
---

## Signature

<!-- The finding signature, e.g. WRONG_RESULT|all_reduce|torch.int64 -->

## Minimal repro

<!-- Plain PyTorch, no distfuzz imports. Say how to launch it (torchrun, mp.spawn) and the world size. -->

```python
```

## Actual vs expected

## Reliability and versions

- Runs reproduced: k/N on torch <version>
- Other versions tried (nightly?):

## Upstream

- Duplicate search terms and closest issues:
- Status: not reported / reported as #...

## Triage

- [ ] Reproduces in a fresh container
- [ ] Not a fuzzer artifact (reference model, input construction, tolerance)
- [ ] Not documented behaviour
