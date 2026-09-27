#!/bin/sh
# usage (inside the Docker image, from this directory): sh run_all.sh
for f in noflat_drops_values.py full_sd_stateless_optim.py get_state_dict_phantom_step.py ignore_frozen_roundtrip.py fused_sgd_dtensor.py; do
  echo "### $f"; timeout 120 python "$f" 2>&1 | grep -iv warn | tail -4
done
echo "### flat_osd_full_load.py 2 1"
timeout 120 python flat_osd_full_load.py 2 1 2>&1 | grep -iv warn | tail -4
echo "### bcast_no_optim_state.py 2 0.9 1 (TORCH_DISTRIBUTED_DEBUG=DETAIL)"
TORCH_DISTRIBUTED_DEBUG=DETAIL timeout 120 python bcast_no_optim_state.py 2 0.9 1 2>&1 | grep "^rank" | cut -c1-260
echo "### bcast_no_optim_state.py 2 0.9 1 with SPLIT=1 (control, should work)"
SPLIT=1 timeout 120 python bcast_no_optim_state.py 2 0.9 1 2>&1 | grep "^rank"
echo "### tp_uneven_mlp.py 3 5"
timeout 120 python tp_uneven_mlp.py 3 5 2>&1 | grep "^rank" | cut -c1-160
echo "### tp_rowwise_grad_placement.py 4 3"
timeout 120 python tp_rowwise_grad_placement.py 4 3 2>&1 | grep "^W=" | cut -c1-200
