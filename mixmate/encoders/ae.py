from abc import abstractmethod

from typing import Tuple

from torch import Tensor
from torch.nn import Module


class AutoEncoder(Module):

    @abstractmethod
    def forward(self, data: Tensor) -> Tuple[Tensor, Tensor]:
        """Reconstructs a data point by passing it through encoder and decoder.

        Parameters
        ----------
        data : Tensor
            The data point to be auto-encoded.

        Returns
        -------
        Tensor
            The reconstructed data point.
        Tensor
            The latent representation of the data point.

        """
        pass

    @abstractmethod
    def calc_recon_loss(self, data: Tensor, recon: Tensor) -> Tensor:
        """Compute the reconstruction loss.

        Parameters
        ----------
        data : Tensor
            The original data point.
        recon : Tensor
            The reconstructed data point.

        Returns
        -------
        Tensor
            The batchwise reconstruction loss.

        """

    @abstractmethod
    def calc_reg_loss(self, code: Tensor) -> Tensor:
        """Compute the regularization loss.

        Parameters
        ----------
        code : Tensor
            A representation of a data point in latent space.

        Returns
        -------
        Tensor
            The batchwise regularization loss.

        """
