# distfuzz

distfuzz checks whether PyTorch's distributed stack computes the same thing as a single process. It generates distributed programs and whole training runs (FSDP2, tensor and pipeline parallel, HSDP, DDP, expert parallel, distributed checkpoints, `torch.compile`), runs each one on several ranks, and compares every value, gradient and checkpoint with the same computation on one process. When they differ, it shrinks the case to a few lines and writes a standalone PyTorch script.

It has found 44 verified bugs in PyTorch 2.14.0. For 34 of them a search of the PyTorch tracker found no earlier report, and 22 return a wrong value with no error on any rank.

## Who it is for

If you run distributed training, the training profile tells you whether the PyTorch release you use trains a given parallelism and checkpoint setup to the same numbers as a single process, and where it does not. Start with [Quickstart A](quickstart.md#a-check-your-training-setup).

If you work on PyTorch's distributed layer, the collectives, DTensor, checkpoint and compile fuzzers test one API at a time and hand you a minimized repro for each difference. Start with [Quickstart B](quickstart.md#b-fuzz-pytorchs-distributed-layer).

## A silent bug, in four lines

`F.mse_loss` on a sharded DTensor gives the right loss and twice the right gradient on 2 ranks:

```python
x, y = torch.tensor([2.0, 1.0]), torch.tensor([5.0, -1.0])
x2 = distribute_tensor(x, mesh, [Shard(0)]).requires_grad_()
F.mse_loss(x2, distribute_tensor(y, mesh, [Shard(0)])).backward()
# one process: grad [-3.0, 2.0]    sharded on 2 ranks: grad [-6.0, 4.0]
```

The full script is `examples/repros/dtensor/mse_loss_grad.py`. Nothing raises and nothing warns; the gradient is multiplied by the world size, on 2.14.0 and on the November 2026 nightly.

## Where to go next

| Page | For |
|---|---|
| [Install](install.md) | Docker image, host tools, images for other PyTorch releases |
| [Quickstart](quickstart.md) | the two ways to use distfuzz, in a few commands each |
| [Reading the output](output.md) | progress lines, finding kinds, signatures, output folders, bundles |
| [Worked examples](examples/index.md) | five end-to-end runs with real output |
| [Bugs found](bugs.md) | all 44 entries with causes, upstream status and repro scripts |
| [Command reference](cli.md) | every subcommand and the flags you are likely to change |
| [Fuzzer details](fuzzers.md) | what the training profile and the compile fuzzer generate, axis by axis |
| [Architecture](architecture.md) | how the fuzzers work, the syzkaller mapping, limitations |
