# Backlog

Roadmap in priority order, then open questions about findings. Effort figures are estimates for one person working on this full time.

## Roadmap

| # | Item | Why | Effort |
|---|---|---|---|
| 1 | Baselines at equal budget: single-op property tests, an OpInfo sweep with uneven shapes, FreeFuzz, random mode; at least 5 seeds; plot time to each known bug | A single-op property test of about 250 lines found the Gloo strided bug in 3 to 28 s and 7 of 9 DTensor bug classes. Until the fuzzer beats that, the multi-call machinery is unproven | 1 week |
| 2 | Deflake in the fuzz loop: re-run each new signature 3 to 5 times in a fresh session and record k/N | syzkaller deflakes before admitting a program (`pkg/fuzzer/job.go`); the fuzzers here record one-shot findings | 3 days |
| 3 | Reload the corpus and signature database on start | The corpus is written but never read back, so every campaign starts from zero | 0.5 day |
| 4 | Crash signatures from stacks, not exit codes: `faulthandler` plus `gdb -batch` on core dumps, top 3 frames | 65 of 67 WRONG_RESULT signatures in one random run were a single Gloo bug; crash signatures say which rank died, not where | 2 days |
| 5 | C++ coverage of Gloo and `ProcessGroupGloo.cpp` (`-fsanitize-coverage=trace-pc-guard` or gcov, start with standalone Gloo) | Python-only feedback cannot see where the bugs live; guided mode never beat random | 2 to 3 weeks |
| 6 | ASan/UBSan build of Gloo and c10d, then TSan for Gloo's threads | Turns the in-flight `resize_` abort and the unconfirmed SIGSEGVs into precise reports. A standalone ASan Gloo (tcp transport) builds; the uv transport needs a patched CMake | 1 to 2 weeks |
| 7 | Continuous runs: nightly wheel, replay every repro, one-hour campaign, diff signatures | `distfuzz.repro` does one-off bisection; nothing runs on a schedule | 1 week |
| 8 | API breadth: the 6 remaining c10d calls (`*_coalesced`, `*_object` variants), DTensor `_foreach_*` and fused optimizer ops, NCCL on rented GPUs | 20/26 c10d calls and 86/364 DTensor strategy ops are described; NCCL is what people train on | 1 to 2 weeks each |
| 9 | Choice table: call priorities from shared argument types and corpus co-occurrence | Equivalent of syzkaller's `prog/prio.go`; mutator weights are guesses today | 3 days |
| 10 | Commit-level bisection between the two releases that bracket a regression | `vector_norm(inf)` on `Partial(max)` is correct on 2.10.0 and wrong from 2.11.0 | 2 days |

## Findings: open questions

- Independent verification of the DCP findings and the in-flight `resize_`/`set_` abort, the same way the Gloo and DTensor findings were checked (own repros, 20 runs, 2.14 and nightly, duplicate search).
- Which findings to report upstream, and in what form. Candidates in order: DTensor `flatten()` hang, DCP `flatten_state_dict=False` data loss, Gloo `all_reduce`/`reduce_scatter` strided corruption (as a comment on #192920 or a test for PR #191187), DTensor loss functions under uneven shards.
- Gloo send/recv into a strided buffer hangs until timeout. Not investigated.
- A rare SIGSEGV in Gloo's uv transport `UnboundBuffer::send` was seen once on macOS and never reproduced on Linux (the Linux wheels build only the tcp transport).
- `all_gather_into_tensor` rejects the stacked output form on Gloo although the docs describe it; zero-size `reduce_scatter_tensor`/`all_gather_into_tensor` abort the process through `gloo::EnforceNotMet` instead of raising. Both need a duplicate search.

## Fuzzer defects known and not fixed

- Collectives: a WRONG_RESULT is attributed to the last call that touches the tensor, so a trailing local op can hide the collective that corrupted it (strict xfail `test_O4_...`).
- Collectives: the reference model marks some combinations Gloo accepts as uncertain (bitwise ops on bool, complex AVG), which skips their value checks.
- DCP: the minimizer starts a fresh world per candidate and is slow; most DCP repros were reduced by hand.
