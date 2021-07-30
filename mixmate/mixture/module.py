from typing import Dict, Any

import torch
from torch import Tensor
from torch.nn import Module, ModuleList
import torch.nn.functional as F
from torch.distributions import Categorical
import pytorch_lightning as pl
from hydra.utils import instantiate

from mixmate.encoders.ae import AutoEncoder
from mixmate.mixture.attention import BiasedAttention

import pdb


class MixMATE(pl.LightningModule):

    def __init__(
        self,
        autoencoder_config: Dict[str, Any],
        num_components: int,
        beta: float = 1.,
        lr: float = 1e-3
    ) -> None:
        super().__init__()
        self.num_components = num_components
        self.beta = beta
        self.lr = lr

        self.autoencoders: ModuleList[AutoEncoder] = ModuleList(
            [instantiate(autoencoder_config) for _ in range(self.num_components)]
        )
        self.attn = BiasedAttention(self.num_components)
        self.train_accuracy = pl.metrics.Accuracy()
        self.valid_accuracy = pl.metrics.Accuracy()

        # Freeze params
        self.attn.biases.requires_grad = False
        # for ae in self.autoencoders:
        #     for param in ae.parameters():
        #         param.requires_grad = False

    def forward(self, data: Tensor) -> Tensor:
        autoencoder_losses = []
        autoencoder_recons = []
        for autoencoder in self.autoencoders:
            recon, code = autoencoder(data)
            recon_loss = autoencoder.calc_recon_loss(data, recon)
            reg_loss = autoencoder.calc_reg_loss(code)
            autoencoder_losses.append(recon_loss + reg_loss)
            # autoencoder_losses.append(recon_loss)
            autoencoder_recons.append(recon)
        energies = torch.stack(autoencoder_losses, dim=-1)

        energy, probs = self.attn(-energies, energies)
        log_prior = F.log_softmax(self.attn.biases).expand_as(probs)
        kl_div = F.kl_div(log_prior, probs, reduction='none').sum(dim=-1)

        return probs, energy, kl_div, autoencoder_recons

    def training_step(self, batch: Tensor, batch_idx: int) -> Tensor:
        data, labels = batch
        probs, energy, kl_div, recons = self.forward(data)
        pred_clusters = torch.argmax(probs, dim=-1)
        acc = self.train_accuracy(pred_clusters, labels)

        dim = self.autoencoders[0].input_size
        global_energy = energy.mean(dim=-1) / dim
        # global_kl_div = self.beta * kl_div.mean(dim=-1)
        global_kl_div = self.beta * Categorical(probs).entropy().mean(dim=-1) / dim
        global_loss = global_energy + global_kl_div

        # Log train metrics
        self.log('Training/train_loss', global_loss, on_step=True, on_epoch=True)
        self.log('Training/train_energy', global_energy, on_step=True, on_epoch=True)
        self.log('Training/train_kl', global_kl_div, on_step=True, on_epoch=True)
        self.log('Training/train_acc', acc, on_step=True, on_epoch=True)

        tensorboard = self.logger.experiment

        # Log attention biases
        tensorboard.add_scalars(
            'Training/attn_bias',
            dict([(str(k), v) for k, v in enumerate(self.attn.biases)]),
            self.global_step
        )

        # Log sample images
        # imgs = torch.flatten(
        #     torch.stack([data] + recons, dim=1), start_dim=0, end_dim=1
        # )
        # imgs = imgs.clamp(0, 1)
        # tensorboard.add_images('Training/sample_imgs', imgs[:15], batch_idx)

        return global_loss

    def validation_step(self, batch: Tensor, batch_idx: int) -> Tensor:
        data, labels = batch
        probs, energy, kl_div, recons = self.forward(data)
        pred_clusters = torch.argmax(probs, dim=-1)
        acc = self.valid_accuracy(pred_clusters, labels)
        global_energy = energy.mean(dim=-1)
        # global_kl_div = self.beta * kl_div.mean(dim=-1)
        global_kl_div = self.beta * Categorical(probs).entropy().mean(dim=-1)
        global_loss = global_energy + global_kl_div

        # Log valid metrics
        self.log('Validation/valid_loss', global_loss, on_step=False, on_epoch=True)
        self.log('Validation/valid_energy', global_energy, on_step=False, on_epoch=True)
        self.log('Validation/valid_kl', global_kl_div, on_step=False, on_epoch=True)
        self.log('Validation/valid_acc', acc, on_step=False, on_epoch=True)

        # Log sample images
        if batch_idx == 0 and self.current_epoch % 2 == 0:
            imgs = torch.flatten(
                torch.stack([data] + recons, dim=1), start_dim=0, end_dim=1
            )
            imgs = imgs.clamp(0, 1)
            tensorboard = self.logger.experiment
            tensorboard.add_images(
                'Validation/sample_imgs', imgs[:30], self.global_step
            )

            for idx, ae in enumerate(self.autoencoders):
                filters = ae.W.view((-1, 1, 28, 28))
                filters = (filters.max() - filters) / (filters.max() - filters.min())
                tensorboard.add_images(
                    f'Autoencoder_{idx}/filters', filters[:16], self.global_step
                )


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), self.lr)
        return optimizer
