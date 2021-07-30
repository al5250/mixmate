from typing import Union

import torch
from torch import Tensor
from torch.nn import MSELoss
from crsae.pytorch.src.model import CRsAE2D

from mixmate.encoders import AutoEncoder


class CRsAE(AutoEncoder):

    def __init__(self, **hyperparams):
        super().__init__()
        self._crsae = CRsAE2D(hyperparams)
        self._recon_loss = MSELoss(reduction='none')

    def forward(self, data: Tensor) -> Union[Tensor, Tensor]:
        recon, code, _ = self._crsae(data)
        return recon, code

    def calc_recon_loss(self, data: Tensor, recon: Tensor) -> Tensor:
        recon_loss = self._recon_loss(data, recon).mean(dim=(-3, -2, -1))
        return recon_loss

    def calc_reg_loss(self, code: Tensor) -> Tensor:
        reg_loss = torch.abs(code).mean(dim=(-3, -2, -1))
        return reg_loss
