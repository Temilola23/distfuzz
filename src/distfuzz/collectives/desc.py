from __future__ import annotations

from typing import Any

CALLS: dict[str, dict[str, Any]] = {
    "tensor": dict(args={"spec": "tspec"}, rets={"out": "tensor"}, weight=0),
    "local": dict(args={"t": "tensor", "fn": ("enum", ["add1", "mul2", "zero", "neg"])}, rets={}, weight=3),
    "new_group": dict(args={"ranks": "ranks", "local_sync": "bool_rare"}, rets={"g": "group"}, weight=3),
    "new_subgroups": dict(args={"size": "subgroup_size"}, rets={"g": "group"}, weight=2),
    "all_reduce": dict(
        args={"t": "tensor", "op": "redop", "group": "group", "async_op": "bool"}, rets={"w": "work"}, weight=10
    ),
    "broadcast": dict(
        args={"t": "tensor", "root": "root", "root_mode": "root_mode", "group": "group", "async_op": "bool"},
        rets={"w": "work"},
        weight=8,
    ),
    "reduce": dict(
        args={
            "t": "tensor",
            "root": "root",
            "root_mode": "root_mode",
            "op": "redop",
            "group": "group",
            "async_op": "bool",
        },
        rets={"w": "work"},
        weight=5,
    ),
    "all_gather": dict(
        args={"out": "tspec", "n_delta": "n_delta", "t": "tensor", "group": "group", "async_op": "bool"},
        rets={"w": "work", "outs": "list"},
        weight=6,
    ),
    "all_gather_into_tensor": dict(
        args={"out": "tspec", "t": "tensor", "group": "group", "async_op": "bool"},
        rets={"w": "work", "out": "tensor"},
        weight=6,
    ),
    "reduce_scatter": dict(
        args={
            "out": "tspec",
            "ins": "tspec",
            "n_delta": "n_delta",
            "op": "redop",
            "group": "group",
            "async_op": "bool",
        },
        rets={"w": "work", "out": "tensor"},
        weight=5,
    ),
    "reduce_scatter_tensor": dict(
        args={"out": "tspec", "t": "tensor", "op": "redop", "group": "group", "async_op": "bool"},
        rets={"w": "work", "out": "tensor"},
        weight=5,
    ),
    "all_to_all_single": dict(
        args={"inspec": "tspec", "matrix": "split_matrix", "even": "bool", "group": "group", "async_op": "bool"},
        rets={"w": "work", "out": "tensor", "in": "tensor"},
        weight=5,
    ),
    "all_to_all": dict(
        args={"out": "tspec", "ins": "tspec", "group": "group", "async_op": "bool"},
        rets={"w": "work", "outs": "list"},
        weight=1,
    ),
    "scatter": dict(
        args={
            "t": "tensor",
            "ins": "tspec",
            "n_delta": "n_delta",
            "root": "root",
            "root_mode": "root_mode",
            "list_everywhere": "bool_rare",
            "group": "group",
            "async_op": "bool",
        },
        rets={"w": "work"},
        weight=4,
    ),
    "gather": dict(
        args={
            "t": "tensor",
            "out": "tspec",
            "n_delta": "n_delta",
            "root": "root",
            "root_mode": "root_mode",
            "list_everywhere": "bool_rare",
            "group": "group",
            "async_op": "bool",
        },
        rets={"w": "work", "outs": "list"},
        weight=4,
    ),
    "barrier": dict(args={"group": "group", "async_op": "bool"}, rets={"w": "work"}, weight=2),
    "monitored_barrier": dict(args={"group": "group", "wait_all": "bool"}, rets={}, weight=1),
    "p2p": dict(
        args={"t": "tensor", "src": "rank", "dst": "rank", "tag": ("int", 0, 3), "async_op": "bool"}, rets={}, weight=5
    ),
    "ring": dict(args={"t": "tensor", "shift": "shift"}, rets={"out": "tensor"}, weight=3),
    "wait": dict(args={"w": "work"}, rets={}, weight=4),
    "all_gather_object": dict(args={"seed": ("int", 0, 99), "group": "group"}, rets={"outs": "list"}, weight=2),
    "broadcast_object_list": dict(
        args={"seed": ("int", 0, 99), "k": ("int", 1, 3), "root": "root", "root_mode": "root_mode", "group": "group"},
        rets={"outs": "list"},
        weight=2,
    ),
    # Fault and lifecycle ops: weight 0, so only `--fault` mode (FAULT_WEIGHTS) generates them.
    # The reference model treats them as "uncertain", so only the crash, hang and guard oracles apply.
    "drop_work": dict(args={"w": "work", "gc": "bool"}, rets={}, weight=0, fault=True),
    "wait_twice": dict(args={"w": "work"}, rets={}, weight=0, fault=True),
    "wait_late": dict(args={"w": "work"}, rets={}, weight=0, fault=True),
    "destroy_group": dict(args={"group": "group"}, rets={}, weight=0, fault=True),
    "abort_group": dict(args={"group": "group"}, rets={}, weight=0, fault=True),
    "async_resize": dict(
        args={"t": "tensor", "op": "redop", "group": "group", "how": "resize_how"}, rets={}, weight=0, fault=True
    ),
    "async_free": dict(args={"t": "tensor", "op": "redop", "group": "group"}, rets={}, weight=0, fault=True),
    "batch_mismatch": dict(args={"t": "tensor", "kind": "batch_kind"}, rets={}, weight=0, fault=True),
    "crash_rank": dict(args={"victim": "rank", "group": "group", "when": "crash_when"}, rets={}, weight=0, fault=True),
    "short_timeout_probe": dict(args={"group": "group"}, rets={}, weight=0, fault=True),
}

GENERATABLE = [n for n, d in CALLS.items() if d["weight"] > 0]
FAULT_OPS = [n for n, d in CALLS.items() if d.get("fault")]
FAULT_WEIGHTS = {
    "drop_work": 6,
    "wait_twice": 5,
    "wait_late": 4,
    "destroy_group": 6,
    "abort_group": 5,
    "async_resize": 8,
    "async_free": 7,
    "batch_mismatch": 6,
    "crash_rank": 3,
    "short_timeout_probe": 4,
}
