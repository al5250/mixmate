from abc import abstractmethod

from torch.nn import Module
from torch import Tensor
import torch
import torch.nn.functional as F


class ProximalOperator(Module):

    @abstractmethod
    def forward(self, z: Tensor, step_size: float) -> Tensor:
        pass

    @abstractmethod
    def compute_reg_pen(self, z: Tensor) -> Tensor:
        pass


class DoubleSidedBiasedReLU(ProximalOperator):

    def __init__(self, sparse_penalty: float) -> None:
        super().__init__()
        self.sparse_penalty = sparse_penalty

    def forward(self, z: Tensor, step_size: float) -> Tensor:
        bias = self.sparse_penalty * step_size
        x = (z.abs() - bias).relu() * z.sign()
        return x
    
    def compute_reg_pen(self, z: Tensor) -> Tensor:
        return self.sparse_penalty * torch.abs(z).sum(dim=-1)


class GroupSparseBiasedReLU(ProximalOperator):

    def __init__(
        self,
        num_groups: int,
        group_size: int,
        group_sparse_penalty: float
    ) -> None:
        super().__init__()
        self.num_groups = num_groups
        self.group_size = group_size
        self.group_sparse_penalty = group_sparse_penalty

    def forward(self, z: Tensor, step_size: float) -> Tensor:
        z = z.unflatten(dim=-1, sizes=(self.num_groups, self.group_size))
        bias = self.group_sparse_penalty * step_size
        x = (z.norm(dim=-1, keepdim=True) - bias).relu() * F.normalize(z, dim=-1)
        x = x.flatten(start_dim=-2, end_dim=-1)
        return x
    
    def compute_reg_pen(self, z: Tensor) -> Tensor:
        z = z.unflatten(dim=-1, sizes=(self.num_groups, self.group_size))
        return self.group_sparse_penalty * torch.norm(z, dim=-1).sum(dim=-1)
