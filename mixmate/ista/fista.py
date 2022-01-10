from typing import Callable, Optional

import torch
from torch import Tensor
import numpy as np

from mixmate.ista.proximal import ProximalOperator


def fista(
    y: Tensor,
    A: Tensor,
    prox: ProximalOperator,
    num_iter: int,
    step_size: float,
    mask: Optional[Tensor] = None
) -> Tensor:
    """Run batched variant of the Fast Iterative Shrinkage-Thresholding Algorithm.

    Iteratively solves the problem {arg min_x ||Ax - y||^2 + f(x)} using FISTA.
    Supports simultaneous optimization of multiple FISTA problems in parallel.

    Parameters
    ----------
    y : Tensor
        The input data of size (... x B x M) with batch dimension B
        and data dimension M.
    A : Tensor
        The dictionaries of size (... x M x N) with data dimension M
        and code dimension N.
    prox : Callable[[Tensor], Tensor]
        The proximal operator prox_f(z) = {arg min_x ||x - z||^2 + f(x)}.
    num_iter : int
        Number of iterations to run FISTA.
    step_size: float
        Step size of gradient descent within FISTA.
    mask : Tensor, optional
        The masks over the input of size (... x B x M) with batch 
        dimension B and data dimension M.

    Returns
    -------
    Tensor
        The outputs codes of size (... x B x N) with batch dimension B
        and code dimension N.

    """
    if mask is not None:
        assert torch.all(y[..., ~mask] == 0.0)

    y_size = y.size()
    batch_size, dim_data = y_size[-2:]
    dim_code = A.size(dim=-1)
    x_size = list(y_size)
    x_size[-1] = dim_code

    A_transpose = A.transpose(-2, -1)
    t_old = 1.0

    x_old = torch.zeros(x_size, device=y.device, dtype=y.dtype)
    x_tmp = torch.zeros_like(x_old)

    for t in range(num_iter):
        if mask is None:
            grad = ((x_tmp @ A_transpose) - y) @ A
        else:
            grad = (mask * (x_tmp @ A_transpose) - y) @ A
        x_new = prox(x_tmp - grad * step_size, step_size)
        t_new = (1 + np.sqrt(1 + 4 * t_old * t_old)) / 2
        x_tmp = x_new + ((t_old - 1) / t_new) * (x_new - x_old)
        x_old, t_old = x_new, t_new

    return x_new
