from typing import Tuple

import math
import torch
from torch import Tensor
from torch.nn import Module, Parameter
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from hydra.utils import instantiate
import numpy as np

from mixmate.encoders.ae import AutoEncoder
from mixmate.mixture.attention import BiasedAttention
from mixmate.ista import fista, DoubleSidedBiasedReLU

import pdb


class SemiSupervisedMixMATEv2(pl.LightningModule):
    """Second version of MixMATE with sparse auto-encoders.

    Supports parallel computation of multiple auto-encoders, unlike the first
    version of MixMATE that performs them in sequence.

    """

    def __init__(
        self,
        labeled_loader: DataLoader,
        num_components: int,
        num_layers: int,
        input_size: int,
        hidden_size: int,
        step_size: float,
        sparse_penalty: float,
        beta: float = 1.,
        alpha: float = 1.,
        lr: float = 1e-3
    ) -> None:
        super().__init__()
        self.labeled_loader = labeled_loader
        self.labeled_iterator = iter(self.labeled_loader)
        self.num_components = num_components
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.step_size = step_size

        self.beta = beta
        self.alpha = alpha
        self.lr = lr

        W = torch.randn((self.num_components, self.input_size, self.hidden_size))
        self.W = Parameter(F.normalize(W, dim=1))
        self.prox = DoubleSidedBiasedReLU(sparse_penalty)
        self.attn = BiasedAttention(self.num_components)

        self.train_accuracy = pl.metrics.Accuracy()
        self.valid_accuracy = pl.metrics.Accuracy()

        # Freeze params
        self.attn.biases.requires_grad = False

    def encode(self, data: Tensor) -> Tensor:
        code = fista(data, self.W, self.prox, self.num_layers, self.step_size)
        return code

    def decode(self, code: Tensor) -> Tensor:
        recon = code @ self.W.transpose(-2, -1)
        return recon

    @torch.no_grad()
    def _normalize(self):
        # rescale each dictionary atom to have norm 1.
        self.W.div_(self.W.norm(dim=1, keepdim=True))

    def forward(self, data: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        img_shape = data.size()[-3:]
        data = data.flatten(start_dim=-3, end_dim=-1)

        codes = self.encode(data)
        recons = self.decode(codes)

        recon_losses = ((recons - data) ** 2).sum(dim=-1)
        reg_losses = self.prox.sparse_penalty * torch.abs(codes).sum(dim=-1)
        energies = recon_losses + reg_losses
        # energies = recon_losses
        energy, probs = self.attn(keys=-energies, values=energies, dim=0)

        probs = probs.transpose(0, 1)
        recons = recons.transpose(0, 1).unflatten(dim=-1, sizes=img_shape)
        codes = codes.transpose(0, 1)
        energies = energies.transpose(0, 1)

        log_prior = self.attn.log_prior.expand_as(probs)
        kl_div = F.kl_div(log_prior, probs, reduction='none').sum(dim=-1)

        return probs, recons, codes, energy, kl_div, energies

    def _compute_accuracy(self, pred_probs: Tensor, target: Tensor) -> Tensor:
        pred_clusters = torch.argmax(pred_probs, dim=-1)
        acc = self.train_accuracy(pred_clusters, target)
        return acc

    def _compute_global_losses(self, energy: Tensor, kl_div: Tensor) -> Tensor:
        global_energy = energy.mean(dim=-1) / self.input_size
        global_kl_div = self.beta * kl_div.mean(dim=-1) / self.input_size
        global_loss = global_energy + global_kl_div
        return global_loss, global_energy, global_kl_div

    def _compute_entropy(self, probs: Tensor) -> Tensor:
        return Categorical(probs).entropy().mean(dim=-1)

    def _compute_expected_sparsity(self, probs: Tensor, codes: Tensor) -> Tensor:
        sparsity = (torch.abs(codes) < 1e-6).sum(dim=2) / codes.size(dim=2)
        expected_sparsity = torch.sum(probs * sparsity, dim=1)
        return expected_sparsity.mean(dim=0)

    def training_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        data, labels = batch
        probs, recons, codes, energy, kl_div, _ = self.forward(data)
        acc = self._compute_accuracy(probs, labels)
        global_loss, global_energy, global_kl_div = self._compute_global_losses(
            energy, kl_div
        )
        entropy = self._compute_entropy(probs)
        sparsity = self._compute_expected_sparsity(probs, codes)

        # Log train metrics
        self.log('Training/loss', global_loss, on_step=True, on_epoch=True)
        self.log('Training/energy', global_energy, on_step=True, on_epoch=True)
        self.log('Training/kl', global_kl_div, on_step=True, on_epoch=True)
        self.log('Training/acc', acc, on_step=True, on_epoch=True)
        self.log('Training/entropy', entropy, on_step=True, on_epoch=True)
        self.log('Training/sparsity', sparsity, on_step=True, on_epoch=True)

        tensorboard = self.logger.experiment

        # Log attention biases
        tensorboard.add_scalars(
            'Training/attn_bias',
            dict([(str(k), v) for k, v in enumerate(self.attn.biases)]),
            self.global_step
        )

        try:
            data_supervised, labels_supervised = next(self.labeled_iterator)
        except StopIteration:
            self.labeled_iterator = iter(self.labeled_loader)
            data_supervised, labels_supervised = next(self.labeled_iterator)
        data_supervised = data_supervised.to(data)
        labels_supervised = labels_supervised.to(labels)

        probs_, recons_, codes_, energy_, kl_div_, energies_ = self.forward(data_supervised)
        supervised_loss = self.alpha * F.cross_entropy(-energies_, labels_supervised)

        self.log('Training/supervised_loss', supervised_loss, on_step=True, on_epoch=True)

        if self.alpha > 0:
            global_loss += supervised_loss

        return global_loss

    def validation_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        data, labels = batch
        probs, recons, codes, energy, kl_div, _ = self.forward(data)
        acc = self._compute_accuracy(probs, labels)
        global_loss, global_energy, global_kl_div = self._compute_global_losses(
            energy, kl_div
        )
        entropy = self._compute_entropy(probs)
        sparsity = self._compute_expected_sparsity(probs, codes)

        # Log valid metrics
        self.log('Validation/loss', global_loss, on_step=False, on_epoch=True)
        self.log('Validation/energy', global_energy, on_step=False, on_epoch=True)
        self.log('Validation/kl', global_kl_div, on_step=False, on_epoch=True)
        self.log('Validation/acc', acc, on_step=False, on_epoch=True)
        self.log('Validation/entropy', entropy, on_step=False, on_epoch=True)
        self.log('Validation/sparsity', sparsity, on_step=False, on_epoch=True)

        # Log sample images
        if batch_idx == 0 and self.current_epoch % 2 == 0:
            imgs = torch.flatten(
                torch.cat([data.unsqueeze(dim=1), recons], dim=1),
                start_dim=0,
                end_dim=1
            )
            imgs = imgs.clamp(0, 1)
            tensorboard = self.logger.experiment
            tensorboard.add_images(
                'Validation/sample_imgs', imgs[:100], self.global_step
            )

            for idx in range(self.num_components):
                dim = int(np.sqrt(self.input_size))
                filters = self.W[idx].transpose(0, 1).view((-1, 1, dim, dim))
                filters = (filters.max() - filters) / (filters.max() - filters.min())
                tensorboard.add_images(
                    f'Autoencoder_{idx}/filters', filters[:16], self.global_step
                )

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), self.lr)
        return optimizer

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx,
                   optimizer_closure, on_tpu, using_native_amp, using_lbfgs):
        optimizer.step(closure=optimizer_closure)
        self._normalize()
