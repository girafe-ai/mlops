"""Reference solution for tp_layers.py.

Forward paths only — backward works because every op used here
(``matmul``, ``all_gather``, ``all_reduce``) has an autograd-aware
counterpart in ``torch.distributed.nn.functional``. We import those
variants so the backward pass does the right collective automatically.
"""

import torch
import torch.distributed as dist
import torch.distributed.nn.functional as dist_F
import torch.nn as nn


class ColumnParallelLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, gather_output: bool = True):
        super().__init__()
        self.world_size = dist.get_world_size()
        assert out_features % self.world_size == 0, (
            f"out_features={out_features} not divisible by world_size={self.world_size}. "
            f"TP degree must divide all sharded dimensions — use a power-of-2 world_size."
        )
        self.in_features = in_features
        self.out_features = out_features
        self.out_per_rank = out_features // self.world_size
        self.gather_output = gather_output

        self.weight = nn.Parameter(torch.empty(self.out_per_rank, in_features))
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y_local = torch.matmul(x, self.weight.t())
        if not self.gather_output:
            return y_local
        gathered = dist_F.all_gather(y_local)  # list of tensors
        return torch.cat(gathered, dim=-1)


class RowParallelLinear(nn.Module):
    def __init__(
        self, in_features: int, out_features: int, input_is_parallel: bool = True
    ):
        super().__init__()
        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()
        assert in_features % self.world_size == 0
        self.in_features = in_features
        self.out_features = out_features
        self.in_per_rank = in_features // self.world_size
        self.input_is_parallel = input_is_parallel

        self.weight = nn.Parameter(torch.empty(out_features, self.in_per_rank))
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.input_is_parallel:
            start = self.rank * self.in_per_rank
            end = start + self.in_per_rank
            x = x[..., start:end]
        y_partial = torch.matmul(x, self.weight.t())
        return dist_F.all_reduce(y_partial)
