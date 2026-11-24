# §72: F.mse_loss backward is world_size times too large when an operand is sharded
# torchrun --nproc-per-node 2 mse_loss_grad.py   (run inside the Docker image)
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import Shard, distribute_tensor

dist.init_process_group("gloo")
mesh = init_device_mesh("cpu", (2,))
x, y = torch.tensor([2.0, 1.0]), torch.tensor([5.0, -1.0])

x1 = x.clone().requires_grad_()
F.mse_loss(x1, y).backward()  # one process

x2 = distribute_tensor(x, mesh, [Shard(0)]).requires_grad_()
loss = F.mse_loss(x2, distribute_tensor(y, mesh, [Shard(0)]))
loss.backward()  # sharded over 2 ranks
loss, grad = loss.full_tensor(), x2.grad.full_tensor()

if dist.get_rank() == 0:
    print("loss  ", F.mse_loss(x, y).item(), loss.item())
    print("grad  ", x1.grad.tolist(), grad.tolist())
dist.destroy_process_group()
