# Repros

Plain PyTorch scripts, no distfuzz imports. Run them in the Docker image (`make docker-shell`), not on a laptop host: several of them crash or hang on purpose.

| Bug | Script | Run |
|---|---|---|
| Gloo `all_reduce` on a strided view: wrong on every rank | `collectives/gloo_strided_allreduce.py` | `torchrun --nproc-per-node 4 gloo_strided_allreduce.py` |
| Gloo strided views across all collectives, with out-of-view writes | `collectives/gloo_strided_sweep.py` | `torchrun --nproc-per-node 4 gloo_strided_sweep.py` |
| DTensor A1 `__setitem__` dropped | `dtensor/cases.py` | `python cases.py setitem_shard` |
| DTensor A2 `fill_` on `Partial(sum)` (by design upstream) | `dtensor/cases.py` | `python cases.py fill_partial` |
| DTensor A3 loss functions under uneven shards | `dtensor/cases.py` | `python cases.py mse_loss_uneven 2`, `python cases.py loss_uneven_mixed` |
| DTensor A4 `flatten()` of a zero-size DTensor hangs | `dtensor/cases.py` | `python cases.py flatten_zero_size` (20 s watchdog) |
| DTensor A5 inf-norm of `Partial(max)` | `dtensor/cases.py` | `python cases.py inf_norm_partial_max` |
| DTensor A6 `max()` with an empty shard desyncs ranks | `dtensor/cases.py` | `python cases.py max_empty_shard` |
| DTensor A7 `_MaskPartial` reduced twice | `dtensor/cases.py` | `python cases.py embedding_full_tensor_twice` |
| DTensor A8 backward through `split()[k]` | `dtensor/cases.py` | `python cases.py split_backward` |
| DTensor A9 `squeeze()` on zero-size moves the shard dim | `dtensor/cases.py` | `python cases.py squeeze_zero_size` |
| DCP A1 `flatten_state_dict=False` drops loaded values | `dcp/noflat_drops_values.py` | `python noflat_drops_values.py` |
| DCP A2 flattened optimizer state + full state dict `KeyError` | `dcp/flat_osd_full_load.py` | `python flat_osd_full_load.py 2 1` |
| DCP A3 full state dict into a stateless optimizer | `dcp/full_sd_stateless_optim.py` | `python full_sd_stateless_optim.py` |
| DCP A4 `broadcast_from_rank0` rank-divergent collectives | `dcp/bcast_no_optim_state.py` | `TORCH_DISTRIBUTED_DEBUG=DETAIL python bcast_no_optim_state.py 2 0.9 1` |
| DCP A5 fused SGD on DTensor params | `dcp/fused_sgd_dtensor.py` | `python fused_sgd_dtensor.py` |
| DCP A6 TP row-wise grad placement | `dcp/tp_rowwise_grad_placement.py` | `python tp_rowwise_grad_placement.py 4 3` |
| DCP known: phantom optimizer step (#164929), uneven TP MLP (#150336), `ignore_frozen_params` round trip | `dcp/get_state_dict_phantom_step.py`, `dcp/tp_uneven_mlp.py`, `dcp/ignore_frozen_roundtrip.py` | see the usage line in each file |

`dcp/run_all.sh` runs every DCP script and prints the symptom each one documents. The DTensor file also keeps five cases that were bugs on torch 2.11 and are fixed in 2.14, as controls.
