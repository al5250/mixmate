from typing import Tuple

import torch
from torch import Tensor
from torch.nn import Module, Parameter


class BiasedAttention(Module):

    def __init__(self, num_components: int) -> None:
        super().__init__()
        self.num_components = num_components
        self.biases = Parameter(torch.zeros(num_components))

    def forward(
        self,
        keys: Tensor,
        values: Tensor,
        dim: int = -1
    ) -> Tuple[Tensor, Tensor]:
        keys = keys.transpose(-1, dim)
        values = values.transpose(-1, dim)

        probs = torch.softmax(keys + self.biases, dim=-1)
        # probs = torch.softmax(keys, dim=-1)
        out = torch.sum(probs * values, dim=-1)

        out = out.transpose(-1, dim)
        probs = probs.transpose(-1, dim)
        return out, probs

    @property
    def log_prior(self) -> Tensor:
        return torch.log_softmax(self.biases, dim=-1)
