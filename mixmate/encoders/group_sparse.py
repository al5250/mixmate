import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import MSELoss
from torch import Tensor

from mixmate.encoders.ae import AutoEncoder

import pdb


class GroupSparseAE(AutoEncoder):
    """
    Copied from https://github.com/ds2p/group-sparse-ae/blob/master/mnist/src/model.py.
    """

    def __init__(
        self,
        num_layers,
        num_groups,
        group_size,
        input_size,
        sparse_penalty,
        group_sparse_penalty,
        two_sided=True,
        accelerate=True,
        train_step=False,
        W=None,
        step=None,
    ):
        super(GroupSparseAE, self).__init__()

        # hyperparameters
        self.num_layers = int(num_layers)
        self.num_groups = int(num_groups)
        self.group_size = int(group_size)
        self.input_size = int(input_size)
        self.sparse_penalty = float(sparse_penalty)
        self.group_sparse_penalty = float(group_sparse_penalty)
        self.two_sided = bool(two_sided)
        self.accelerate = bool(accelerate)
        self.train_step = bool(train_step)
        self.hidden_size = self.num_groups * self.group_size
        self._recon_loss = MSELoss(reduction='none')

        # parameters
        if W is None:
            W = F.normalize(torch.randn(self.hidden_size, self.input_size), dim=1)
        if step is None:
            step = W.svd().S[0] ** -2
        self.register_parameter("W", torch.nn.Parameter(W))
        if self.train_step:
            self.register_parameter("step", torch.nn.Parameter(torch.tensor(step)))
        else:
            self.register_buffer("step", torch.tensor(step))

    def normalize(self):
        # rescale each dictionary atom to have norm 1.
        self.W.div_(self.W.norm(dim=1, keepdim=True))

    def forward(self, y):
        x = self.encode(y)
        y = self.decode(x).view(y.shape)
        return y, x

    def encode(self, y):
        if self.accelerate:
            return self.encode_fista(y)
        else:
            return self.encode_ista(y)

    def encode_ista(self, y):
        batch_size, device = y.shape[0], y.device
        y = y.view(batch_size, self.input_size)

        x = torch.zeros(batch_size, self.hidden_size, device=device)
        for k in range(self.num_layers):
            grad = (x @ self.W - y) @ self.W.T
            x = self.activate_soft(x - grad * self.step)
        return x

    def encode_fista(self, y):
        batch_size, device = y.shape[0], y.device
        y = y.view(batch_size, self.input_size)

        x_old = torch.zeros(batch_size, self.hidden_size, device=device)
        x_tmp = torch.zeros(batch_size, self.hidden_size, device=device)
        t_old = torch.tensor(1.0, device=device)
        for k in range(self.num_layers):
            grad = (x_tmp @ self.W - y) @ self.W.T
            x_new = self.activate(x_tmp - grad * self.step)
            t_new = (1 + torch.sqrt(1 + 4 * t_old * t_old)) / 2
            x_tmp = x_new + ((t_old - 1) / t_new) * (x_new - x_old)
            x_old, t_old = x_new, t_new
        return x_new

    def decode(self, x):
        return x @ self.W

    def activate(self, x):
        # if codes must be nonnegative
        if not self.two_sided:
            x = x.relu()

        # individual shrinkage
        scale = (x.abs() - self.sparse_penalty * self.step).relu()
        x = scale * x.sign()

        # reshape
        x = x.view(-1, self.num_groups, self.group_size)

        # group shrinkage
        scale = (
            x.norm(dim=2, keepdim=True) - self.group_sparse_penalty * self.step
        ).relu()
        x = scale * F.normalize(x, dim=2)

        # reshape
        x = x.view(-1, self.hidden_size)

        return x

    def calc_recon_loss(self, data: Tensor, recon: Tensor) -> Tensor:
        recon_loss = self._recon_loss(data, recon).mean(dim=(-3, -2, -1)) * self.input_size
        return recon_loss

    def calc_reg_loss(self, code: Tensor) -> Tensor:
        reg_loss = self.sparse_penalty * torch.abs(code).sum(dim=-1)
        return reg_loss
