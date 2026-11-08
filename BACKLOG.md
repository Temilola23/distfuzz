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
| 8 | API breadth: DTensor fused optimizer ops (`_fused_adam*`), random ops (need a shared RNG check, not value equality), FFT and complex dtypes, `linalg.eig`; attention needs CUDA (CPU SDPA has no strategy); NCCL on rented GPUs | 26/26 c10d calls and 544 to 545 of 632 DTensor strategy ops are reached (556 over three seeds); NCCL is what people train on | 1 to 2 weeks each |
| 9 | Choice table: call priorities from shared argument types and corpus co-occurrence | Equivalent of syzkaller's `prog/prio.go`; mutator weights are guesses today | 3 days |
| 10 | Commit-level bisection between the two releases that bracket a regression | `vector_norm(inf)` on `Partial(max)` is correct on 2.10.0 and wrong from 2.11.0 | 2 days |

## Directions

Five larger directions, each a fuzzer of its own. Tracks 1, 2, 3 and 5 are being built on `feat/train-fuzz` and track 4 on `feat/compile-fuzz`; this section is the shared spec.

### 1. Whole-training-step fuzzing against a single-device reference

Goal: find bugs that only show when FSDP2, TP and the optimizer run together over several steps, where a single-op oracle sees nothing (gradient reduction, `clip_grad_norm_`, optimizer state sharding).

Design sketch: generate a small model from blocks (Linear with uneven in/out sizes, LayerNorm/RMSNorm, embedding, LoRA adapters on frozen bases, a top-k MoE router with a few experts) and a parallelism plan (FSDP2 `fully_shard` per block with random `reshard_after_forward`, TP `parallelize_module` with colwise/rowwise/sequence-parallel styles, 2-D meshes mixing both). Generate a step recipe: forward, loss, backward, `clip_grad_norm_` with `norm_type` drawn from 1, 2, 3, inf and -inf, `foreach` on or off, optimizer (SGD with momentum, Adam, AdamW, fused or foreach), N = 1 to 5 steps, optional gradient accumulation and `no_sync`. Inputs are integer-valued, as in the DTensor fuzzer, so sums are exact in float32.

Oracle: the same model, recipe and data on one device without any parallelism. After every step compare loss, every parameter (`full_tensor()`), every gradient before the optimizer step, the total norm returned by `clip_grad_norm_`, and the optimizer state. Tolerance per dtype as in `runtime.compare`; integer-valued data keeps it exact for SGD and close for Adam.

Success criteria: (a) the oracle passes on 1,000 random configurations of a known-good release with no false positives after triage; (b) it re-finds at least one known bug class, for example the `vector_norm(inf)` on `Partial(max)` regression (2.11.0 and later) through `clip_grad_norm_(norm_type=inf)`; (c) one 1-hour campaign per week on the nightly, with every new signature triaged.

### 2. Checkpoint resharding fuzzing

Goal: state saved under one world size and parallelism loads correctly under another. DCP already has a fuzzer for single-layout save and load; resharding is where the planners do real work.

Design sketch: a program is (W1, P1, save options) then (W2, P2, load options). Parallelism P is a DTensor placement per parameter (from the training-step generator or a production plan, see track 5), including uneven shards and 2-D meshes. Save options: `dcp.save` vs `dcp.async_save` (with and without waiting before the next mutation), full vs sharded state dict (`get_state_dict` options `full_state_dict`, `cpu_offload`), `flatten_state_dict` and `flatten_sharded_tensors`, default vs custom planners, optimizer state included or not. The load side draws its own options independently. World sizes 1 to 6 in Docker.

Oracle: after load, `full_tensor()` of every entry equals the saved values (both are generated from a seed, so the reference is the seed, not another run); optimizer state equals the saved state; keys match. A load that raises on a combination the docs say is supported is a finding.

Success criteria: re-finds the `flatten_state_dict=False` data loss already in the findings; every new signature reproduces from a plain DCP script; a 1-hour campaign covers every (save option, load option) pair at least once.

### 3. MoE expert parallelism with uneven and zero-token routing

Goal: exercise the token dispatch paths that real MoE training uses and that uniform test inputs never hit: experts that receive zero tokens, all tokens on one rank, splits that change every step.

Design sketch: generate a routing table per step (tokens per expert per rank, including zeros, one hot expert, and an empty rank), then run the dispatch and combine with `all_to_all_single` and explicit `input_split_sizes`/`output_split_sizes`, both directly and through DTensor/torchtitan-style expert-parallel modules where available. Vary dtype, hidden size (including odd sizes), number of experts per rank, and whether the backward pass runs. Include the permutation and un-permutation index ops (`index_select`, `scatter_add`, `bincount`/`histc` for counts) since those carry the uneven shapes.

Oracle: a single-process reference that applies the same routing and experts on the full batch; outputs and gradients must match. Zero-size splits must not hang, abort or raise where the reference succeeds. The hang watchdog (below) names the collective and the missing ranks.

Success criteria: zero-token experts, an empty rank and a single hot expert each appear in at least 10% of generated steps; known Gloo zero-size aborts (`reduce_scatter_tensor`/`all_gather_into_tensor`) are separated from new signatures; every new finding has a plain repro.

### 4. torch.compile with DTensor and FSDP2

Goal: compiled and eager DTensor programs give the same results, including the graph breaks, recompiles and collective reordering that compile adds.

Design sketch: reuse DTensor programs and the training-step generator, and run each one eagerly and under `torch.compile` (backends `eager`, `aot_eager`, `inductor` on CPU), with `fullgraph` on and off, dynamic shapes on and off, and FSDP2/TP modules compiled per block or whole. The existing `CMP(...)` step is the seed of this; it currently compiles one lambda.

Oracle: eager vs compiled (values, placements, local shapes, gradients); compiled vs the single-device reference as a tie-breaker when eager and compiled disagree; a compile-only crash is a finding if eager succeeds.

Success criteria: the oracle passes on known-good programs at the eager tolerance; compile failures are deduplicated by the first frame in `torch/_dynamo`, `torch/_inductor` or `torch/distributed`; one known compile/DTensor issue from the tracker is re-found.

### 5. Production configurations as the config source

Goal: spend fuzzing time on configurations people run. Random placements reach odd layouts no one uses and miss the combinations real plans produce.

Design sketch: harvest parallelism plans and model shapes from torchtitan (`parallelize_*` functions and model args for Llama, DeepSeek-style MoE), Hugging Face `base_model_tp_plan` entries in model configs, and accelerate FSDP/TP plugin defaults. Convert each into the distfuzz form (module name to placements, mesh shape, model dims), then shrink dims so programs fit in CPU memory while keeping their divisibility pattern (a dim that is uneven on 8 ranks stays uneven on 4). Feed them to tracks 1 to 4 as seeds and as a mutation source.

Oracle: whatever the consuming track uses; this track only changes inputs.

Success criteria: at least 30 distinct harvested plans; each one runs clean through the training-step oracle on a known-good release, or yields a triaged finding; a campaign seeded from harvested plans reaches more strategy ops per hour than random placements (measured with `reach.py`-style counting under DTensor dispatch).

## Follow-ups from the hang dedup fix

| # | Item | Why | Effort |
|---|---|---|---|
| H1 | Key collective hangs that carry no error on the step's op instead of the `ranks-missing` list, and keep the list in the finding | The rank-dependent strategy choice (VERIFIED.md §22) deadlocks with a different set of waiting ranks each time and produced five signatures in two 30-minute runs | 0.5 day |
| H2 | Name more CPU loops: only `view_groups` has a branch labeller. Add one for any other loop that shows up as `HANG\|cpu\|<file>:<func>` | Unknown loops are keyed by function only, so two branches of one loop would merge | 0.5 day per loop |
| H3 | Hangs inside C++ that hold the GIL never let the watchdog run and still fall back to `HANG\|\|program timeout`; dump them with `faulthandler.dump_traceback_later` per rank and parse the native frames | Keeps the legacy signature from merging those cases | 1 day |
| H4 | Per-signature deflake for HANG: re-run each new hang signature twice before admitting it, as the minimiser already does | The 12 s collective threshold can misfire on an overloaded host | 0.5 day |
| H5 | Taint the outputs of a view whose base is later mutated in place, or skip their FINAL checks | The known #147570 class (in-place through a view DTensor had to copy) accounted for 17 of the 65 new signatures in the final runs | 1 day |
| H6 | Validate placement count in `R(...)` targets, or record that DTensor accepts more placements than mesh dims | `redistribute` to `[Shard(0), Replicate()]` on a 1-D mesh is accepted silently; the mesh mutation keeps such targets on purpose, so every such step is a missing-validation case nobody has triaged | 0.5 day |

## Other items found on the way

| # | Item | Why | Effort |
|---|---|---|---|
| N1 | Count reach under DTensor dispatch, not plain-tensor dispatch | `reach.py` sees only ops that plain tensors dispatch to. About 30 of the 632 names are CompositeImplicit (`clip`, `relu6`, `square`, `reshape`, `contiguous`) or registered only as `.out` overloads (`zero`, `fill`, `copy`, `eye`), so they can never count, and the ceiling is lower than 632 | 1 day |
| N2 | Random ops with a shared-RNG oracle: `bernoulli`, `normal`, `uniform`, `exponential`, `geometric`, `log_normal`, `multinomial`, `rand_like`, `rrelu_with_noise` | 17 of the unreached strategy ops; DTensor's offset-based RNG tracker is its own source of bugs | 1 week |
| N3 | Complex and FFT: `_fft_c2c`, `_fft_r2c`, `_fft_c2r`, `_conj`, `view_as_complex`, `linalg_eig`; compare real and imaginary parts | 6 unreached ops; the value oracle works on `view_as_real` of the result | 3 days |
| N4 | c10d calls the collectives fuzzer does not describe: `split_group`, `_coalescing_manager` with mixed reduce-scatter/all-gather (`*_coalesced` tensor variants), `batch_isend_irecv` as a generated call (it exists only as a fault op) | These are what FSDP2 and pipeline parallelism call; the 26 described calls cover the public single-op API | 3 days |
| N5 | File the plain-PyTorch `ldl_solve` broadcast out-of-bounds read (`research/ops-2026-11/FINDINGS.md`) after sign-off | SIGSEGV and NaN on CPU tensors, outside distributed code; no duplicate found | 0.5 day |

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
