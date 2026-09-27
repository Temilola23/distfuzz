# A1: dcp.load with flatten_state_dict=False silently drops non-tensor values (epoch stays 0)
# usage: python noflat_drops_values.py
import tempfile

import torch
import torch.distributed.checkpoint as dcp

for fst in (True, False):
    d = tempfile.mkdtemp()
    dcp.save(
        {"w": torch.ones(2), "epoch": 5},
        checkpoint_id=d,
        no_dist=True,
        planner=dcp.DefaultSavePlanner(flatten_state_dict=False),
    )
    sd = {"w": torch.zeros(2), "epoch": 0}
    dcp.load(
        sd,
        checkpoint_id=d,
        no_dist=True,
        planner=dcp.DefaultLoadPlanner(flatten_state_dict=False, flatten_sharded_tensors=fst),
    )
    print(f"flatten_sharded_tensors={fst}: w={sd['w'].tolist()} epoch={sd['epoch']} (expected 5)")
