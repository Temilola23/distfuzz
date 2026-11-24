# Bugs found

distfuzz has produced 44 verified bug entries in PyTorch 2.14.0's distributed layer. Each entry has a standalone script that reproduces it with no distfuzz imports, and a cause located at a source line. For all but a few, a patch at that line makes the failure go away. All but two (§75 and §76, which need NCCL) also reproduce on a November 2026 nightly build.

Upstream status comes from searches of the pytorch/pytorch issue and pull request trackers. Thirty-four entries matched no issue or pull request. For three more a pull request describes or would fix the same mechanism, and for one (§53) the fix of a closed issue left this case out. Six extend an open issue. None of these bugs has been filed upstream yet; filing is in progress, and this page will link the reports once they exist.

Twenty-two of the 44 return a wrong value or shape with no error on any rank in at least one tested case. They are marked "Silent" below. The rest hang, raise on every rank, crash, or raise on some ranks while the others wait in a collective.

The impact column is a production-relevance rating from a read of twelve production training and serving stacks: how likely a real configuration is to reach the bug. "Not rated" means the entry was found after that study.

The section numbers (§) are the project's verification log numbers and appear in signatures, commit messages and repro file names.

## Gloo, NCCL, c10d and the CPU optimizer kernels

| § | Bug | Effect | Cause (v2.14.0) | Found by | Upstream status | Impact |
|---|---|---|---|---|---|---|
| 1 | `all_reduce`, `reduce_scatter` on a strided view | Silent: wrong values on every rank, writes outside the view | `Utils.hpp:584-591` | collectives fuzzer | extends open [#192920](https://github.com/pytorch/pytorch/issues/192920) (adds these two ops) | Low |
| 5 | `all_to_all_single`, strided output | Silent: writes outside the output view | `ProcessGroupGloo.cpp:2444` | Gloo audit | no prior report found (open PR [#191187](https://github.com/pytorch/pytorch/pull/191187) would fix) | Low |
| 5 | `reduce_scatter_single`, 0-dim or extra-dim output | Crash: segfault or out-of-bounds read | `ProcessGroupGloo.cpp:1583-1585` | Gloo audit | no prior report found | Low |
| 5 | async `reduce_scatter` | Other: `is_completed()` stays False after `wait()` | `ProcessGroupGloo.cpp:1543-1554` | Gloo audit | no prior report found | Low |
| 25 | `all_reduce` MIN/MAX with a NaN off rank 0 | Silent: NaN dropped; DTensor `amax` and relatives return a number | `gloo/math.h:51, 66` | DTensor fuzzer | no prior report found | not rated |
| 26 | CPU fused Adam, AdamW, SGD, Adagrad, non-contiguous grad | Silent: optimizer state wrong (Adam `exp_avg` off by 8.36) | `FusedAdamKernel.cpp:354-359` | DTensor fuzzer | no prior report found | not rated |
| 74 | one odd-stride tensor in `_broadcast_coalesced` | Error: overlap error; DDP fails to construct | `c10d/Utils.hpp:517-525` | GPU cross-backend run | cause of open [#68407](https://github.com/pytorch/pytorch/issues/68407) | not rated |
| 75 | NCCL `scatter`, `gather`, `reduce_scatter_tensor` on strided or expanded tensors | Silent: wrong values on non-root ranks, writes outside the view | `ProcessGroupNCCL.cpp`, no contiguity check in three entry points | GPU collectives run | extends open [#107352](https://github.com/pytorch/pytorch/issues/107352), [#128066](https://github.com/pytorch/pytorch/issues/128066) | not rated |
| 76 | NCCL equal-split `all_to_all_single` and DTensor `Shard`->`Shard`, non-blocking communicators | Error: spurious "NCCL operation in progress" on every rank; output correct | `nccl.cpp:776`, `NCCL_CHECK` instead of `NCCL_CHECK_TIMEOUT` | GPU fuzz run 2 | no prior report found | not rated |

## DTensor

| § | Bug | Effect | Cause (v2.14.0) | Found by | Upstream status | Impact |
|---|---|---|---|---|---|---|
| 2 | `vector_norm(inf)` on a max-partial tensor | Silent: 6 instead of 3; wrong again since 2.11 | `_math_ops.py:909` | DTensor fuzzer | no prior report found | Low |
| 3 | `flatten()` of a zero-size tensor | Hang: infinite loop on all ranks | `_view_ops.py:465, 487-490` | DTensor fuzzer | no prior report found | Low |
| 19 | view to trailing zero-size dims | Hang: infinite loop on all ranks | `_view_ops.py:483-486` | relaxed baseline, then DTensor fuzzer | no prior report found | Low |
| 11 | mse, smooth L1, Huber loss, uneven shards | Silent: NaN or a wrong mean | `_decompositions.py:240-246` | DTensor fuzzer | no prior report found | Low |
| 72 | `mse_loss` backward, sharded operand | Silent: gradient world-size times too large | decomposition path (no patch) | GPU DTensor run | no prior report found | not rated |
| 12 | `max()` / `min()` with an empty shard | Hang: empty ranks raise, others wait | `_math_ops.py:386-388` | DTensor fuzzer | no prior report found | Low |
| 13 | embedding output reduced twice | Error: second reduction fails | `placement_types.py:1981` | DTensor fuzzer | no prior report found | Low |
| 15 | embedding, a rank with no rows | Hang: that rank raises, others wait | `placement_types.py:1944` | root-causing variants | no prior report found | Low |
| 14 | `squeeze()` with a zero-size dim | Silent: shard moves to the wrong dim | `_view_ops.py:589` | DTensor fuzzer | no prior report found | Low |
| 16 | integer replicated-to-partial split | Silent: float result; odd values lose 1 | `placement_types.py:1857` | DTensor fuzzer | no prior report found (closed PR [#175317](https://github.com/pytorch/pytorch/pull/175317) would have fixed) | Low |
| 17 | `conv1d` when inputs need redistributing | Silent: wrong values | `_tp_conv.py:249-274` | DTensor fuzzer | no prior report found (mechanism in closed PR [#186129](https://github.com/pytorch/pytorch/pull/186129)) | Low |
| 20 | `linalg.cross`, sharded size-1 broadcast dim | Silent or hang: wrong sum, or one rank raises | `_math_ops.py:1755-1758` | relaxed baseline, then DTensor fuzzer | no prior report found | Low |
| 21 | batched linear solves | Error or hang: operand split along a matrix dim | `_math_ops.py:1629-1630, 1672` | DTensor fuzzer | no prior report found | not rated |
| 22 | ops with no strategy of their own (`addmv`, `addr`) | Silent or hang: ranks choose strategies in hash order | `_decompositions.py:310, 332` | DTensor fuzzer | no prior report found | not rated |
| 23 | uneven `Shard(i)` to `Shard(j)` | Hang: views raise on some ranks | `placement_types.py:757-762` | DTensor fuzzer | no prior report found | not rated |
| 24 | `argmax` / `argmin` on a 2-D mesh | Error: `full_tensor()` raises on every rank | `_nonlinear_redux.py:199-251` | DTensor fuzzer | no prior report found | not rated |
| 70 | in-place views (`squeeze_`, `t_`) | Silent: stale global shape | `_dispatch.py:567-569` | GPU DTensor run | no prior report found | not rated |
| 71 | `gather` with a smaller index on a sharded dim | Silent: wrong values | `_tensor_ops.py:661-666` | GPU DTensor run | no prior report found | not rated |
| 73 | `argmax` / `argmin` tie across shards | Silent: a later index returned | `_nonlinear_redux.py:246-247` | GPU DTensor run | no prior report found | not rated |

## Checkpointing, FSDP2, tensor and pipeline parallelism, optimizers

| § | Bug | Effect | Cause (v2.14.0) | Found by | Upstream status | Impact |
|---|---|---|---|---|---|---|
| 4 | DCP load with `flatten_state_dict=False` | Silent: non-tensor state not restored | `default_planner.py:391` | DCP fuzzer | no prior report found | Low |
| 6 | `set_state_dict` with `broadcast_from_rank0` | Hang: on NCCL the load returns stale and the next step hangs | `state_dict.py:1537` | DCP fuzzer | duplicate of open [#157781](https://github.com/pytorch/pytorch/issues/157781) (adds cause) | Medium |
| 7 | full `set_state_dict`, stateless optimizer | Error: assertion | `state_dict.py:1157` | DCP fuzzer | no prior report found | Low |
| 10 | full and flattened optimizer state dict | Error: cannot load its own output | `state_dict.py:1141-1143` | DCP fuzzer | no prior report found | Low |
| 8 | `SGD(fused=True)` on DTensor parameters | Error: no sharding strategy | `_pointwise_ops.py:544-552` | DCP fuzzer | no prior report found | Low |
| 9 | `RowwiseParallel`, fewer inputs than ranks | Error: weight gradient loses its sharding | `_ops/utils.py:173` | DCP fuzzer | no prior report found | Low |
| 30 | inf-norm gradient clipping, FSDP2 empty shard | Hang: empty rank raises, others wait | `_math_ops.py:917, 919` | impact study, then training profile | no prior report found | Medium |
| 31 | FSDP2 over a TP shard empty in another dim | Hang: wrong on one rank, others fail | `_fsdp_param.py:341, 350, 1266` | training profile | no prior report found | Low |
| 32 | full state dict into FSDP2 with TP, dim not divisible by dp x tp | Error: bare assertion on every rank | `placement_types.py:1031-1078` | training profile | no prior report found | Medium-Low |
| 33 | zero-bubble pipeline schedules with resharding FSDP2 stages | Silent: weight gradients missing or off | `pipelining/stage.py:901-913` | training profile | no prior report found | Medium |
| 34 | SGD momentum under sequence parallelism | Error: DCP refuses to save the momentum buffer | `optim/sgd.py:361, 452-454` | training profile | no prior report found | Low |
| 35 | TP embedding output into compiled blocks | Silent or error: gradient lost | `frontend_utils.py:188` | training profile, compile mode | duplicate of open [#201700](https://github.com/pytorch/pytorch/issues/201700) (adds cause) | not rated |

## torch.compile over DTensor

| § | Bug | Effect | Cause (v2.14.0) | Found by | Upstream status | Impact |
|---|---|---|---|---|---|---|
| 50 | bool `all_gather` or `broadcast` under inductor | Silent: int64 result; a mask then gathers rows | `comm_lowering.py:218-220` | compile mode | no prior report found | not rated |
| 51 | `dynamic=True` on a 2 x 2 mesh | Hang: wrong global shape on mesh row 1 | `subclass_codegen.py` (no patch) | compile mode | new symptom of open [#185296](https://github.com/pytorch/pytorch/issues/185296) | not rated |
| 52 | guard misses a change of global shape | Silent: stale global shape, a row dropped | `guards.py:1199-1205` | compile mode (minimized repro) | no prior report found | not rated |
| 53 | one AOTAutograd cache key for all CPU ranks | Silent: wrong loss and gradients on every rank | `_api.py:465-475` | compile mode | no prior report found (fix of closed [#188390](https://github.com/pytorch/pytorch/issues/188390) incomplete) | not rated |

## Repro scripts in this repository

These entries have a plain PyTorch script under [`examples/repros/`](https://github.com/Temilola23/distfuzz/tree/main/examples/repros). Run them inside the Docker image; several hang or crash on purpose.

| § | Script | Command |
|---|---|---|
| 1 | `collectives/gloo_strided_allreduce.py`, `collectives/gloo_strided_sweep.py` | `torchrun --nproc-per-node 4 examples/repros/collectives/gloo_strided_sweep.py` |
| 2 | `dtensor/cases.py` | `python examples/repros/dtensor/cases.py inf_norm_partial_max` |
| 3 | `dtensor/cases.py` | `python examples/repros/dtensor/cases.py flatten_zero_size` |
| 4 | `dcp/noflat_drops_values.py` | `python examples/repros/dcp/noflat_drops_values.py` |
| 6 | `dcp/bcast_no_optim_state.py` | `python examples/repros/dcp/bcast_no_optim_state.py 2 0.9 1` |
| 7 | `dcp/full_sd_stateless_optim.py` | `python examples/repros/dcp/full_sd_stateless_optim.py` |
| 8 | `dcp/fused_sgd_dtensor.py` | `python examples/repros/dcp/fused_sgd_dtensor.py` |
| 9 | `dcp/tp_rowwise_grad_placement.py` | `python examples/repros/dcp/tp_rowwise_grad_placement.py 4 3` |
| 10 | `dcp/flat_osd_full_load.py` | `python examples/repros/dcp/flat_osd_full_load.py 2 1` |
| 11 | `dtensor/cases.py` | `python examples/repros/dtensor/cases.py mse_loss_uneven 2` |
| 12 | `dtensor/cases.py` | `python examples/repros/dtensor/cases.py max_empty_shard` |
| 13 | `dtensor/cases.py` | `python examples/repros/dtensor/cases.py embedding_full_tensor_twice` |
| 14 | `dtensor/cases.py` | `python examples/repros/dtensor/cases.py squeeze_zero_size` |
| 15 | `dtensor/cases.py` | `python examples/repros/dtensor/cases.py embedding_empty_shard` |
| 16 | `dtensor/cases.py` | `python examples/repros/dtensor/cases.py int_replicate_to_partial 2` |
| 17 | `dtensor/cases.py` | `python examples/repros/dtensor/cases.py conv1d_sharded_weight` |
| 30 | `train/clip_inf_empty_shard.py` | `python examples/repros/train/clip_inf_empty_shard.py` |
| 72 | `dtensor/mse_loss_grad.py` | `torchrun --nproc-per-node 2 examples/repros/dtensor/mse_loss_grad.py` |

For the other entries, run the fuzzer that found them and minimize the finding; the [worked examples](examples/index.md) show how. `python -m distfuzz.dcp minimize` and `python -m distfuzz.compile repro` write standalone scripts directly.

## Known issues the fuzzers also reach

The fuzzers rediscover these already reported problems. They are not counted in the 44.

| Area | Problem | Upstream |
|---|---|---|
| Gloo | `reduce`, `broadcast`, `scatter` on strided views return wrong values | [#192920](https://github.com/pytorch/pytorch/issues/192920), [#24836](https://github.com/pytorch/pytorch/issues/24836) |
| Gloo | `resize_` or `set_` on a tensor while an async collective on it is in flight: uncatchable SIGABRT, sometimes SIGSEGV | related: [#81684](https://github.com/pytorch/pytorch/issues/81684) |
| DTensor | `x[i] = v` on a sharded dim is silently dropped | [#147570](https://github.com/pytorch/pytorch/issues/147570) |
| DTensor | backward through `split(...)[k]` fails with mixed Tensor and DTensor | [#118461](https://github.com/pytorch/pytorch/issues/118461) |
| DTensor | `fill_` on `Partial(sum)` keeps the Partial (by design; an upstream test asserts it) | [#172485](https://github.com/pytorch/pytorch/issues/172485) |
| TP | uneven tensor-parallel shards in an MLP | [#150336](https://github.com/pytorch/pytorch/issues/150336) |
