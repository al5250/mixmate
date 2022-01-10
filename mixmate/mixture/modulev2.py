from typing import Tuple, Optional

import math
import torch
from torch import Tensor
from torch.nn import Module, Parameter
import torch.nn.functional as F
from torch.distributions import Categorical
import pytorch_lightning as pl
from hydra.utils import instantiate
from munkres import Munkres
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import normalized_mutual_info_score, adjusted_rand_score

from mixmate.mixture.attention import BiasedAttention
from mixmate.ista import ProximalOperator, fista



class MixMATEv2(pl.LightningModule):
    """Second version of MixMATE with sparse, dictionary learning auto-encoders.

    Supports parallel computation of multiple auto-encoders, unlike the first
    version of MixMATE that performs them in sequence.

    """

    def __init__(
        self,
        num_components: int,
        num_layers: int,
        input_size: int,
        hidden_size: int,
        step_size: float,
        prox: ProximalOperator,
        beta: float = 1.,
        lr: float = 1e-3,
        freeze_bias: bool = True
    ) -> None:
        super().__init__()

        self.save_hyperparameters()

        self.num_components = num_components
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.step_size = step_size

        self.beta = beta
        self.lr = lr
        self.freeze_bias = freeze_bias

        W = torch.randn((self.num_components, self.input_size, self.hidden_size))
        self.W = Parameter(F.normalize(W, dim=1))
        self.prox = instantiate(prox)
        self.attn = BiasedAttention(self.num_components)

        self.train_accuracy = pl.metrics.Accuracy()
        self.valid_accuracy = pl.metrics.Accuracy()

        # Freeze params
        if  self.freeze_bias:
            self.attn.biases.requires_grad = False

    def encode(self, data: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        code = fista(data, self.W, self.prox, self.num_layers, self.step_size, mask)
        return code

    def decode(self, code: Tensor) -> Tensor:
        recon = code @ self.W.transpose(-2, -1)
        return recon

    @torch.no_grad()
    def _normalize(self):
        # rescale each dictionary atom to have norm 1.
        self.W.div_(self.W.norm(dim=1, keepdim=True))

    def forward(
        self, 
        data: Tensor, 
        mask: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        img_shape = data.size()[-3:]
        data = data.flatten(start_dim=-3, end_dim=-1)
        if mask is not None:
            mask = mask.flatten(start_dim=-3, end_dim=-1)

        # Run parallel encoders and decoders (mask is for missing data)
        codes = self.encode(data, mask)
        recons = self.decode(codes)

        # Compute energies and losses
        if mask is None:
            recon_losses = ((recons - data) ** 2).sum(dim=-1)
        else:
            recon_losses = (mask * (recons - data) ** 2).sum(dim=-1)
        reg_losses = self.prox.compute_reg_pen(codes)
        energies = recon_losses + reg_losses
        energy, probs = self.attn(keys=-energies, values=energies, dim=0)

        probs = probs.transpose(0, 1)
        recons = recons.transpose(0, 1).unflatten(dim=-1, sizes=img_shape)
        codes = codes.transpose(0, 1)

        log_prior = self.attn.log_prior.expand_as(probs)
        kl_div = F.kl_div(log_prior, probs, reduction='none').sum(dim=-1)

        return probs, recons, codes, energy, kl_div

    def _compute_accuracy(self, pred_probs: Tensor, target: Tensor) -> Tensor:
        pred_clusters = torch.argmax(pred_probs, dim=-1)
        acc = self.train_accuracy(pred_clusters, target)
        return acc
    
    def _compute_cluster_accuracy(self, cluster, target_cluster, numOfclusters) -> float:
        M = np.zeros((numOfclusters, numOfclusters))
        for i in range(numOfclusters):
            for j in range(numOfclusters):
                M[i][j] = np.sum(np.logical_and(cluster == i, target_cluster == j))
        m = Munkres()
        indexes = m.compute(-M)
        corresp = []
        for i in range(numOfclusters):
            corresp.append(indexes[i][1])
        pred_corresp = [corresp[int(predicted)] for predicted in cluster]
        acc = np.sum(pred_corresp == target_cluster) / float(len(target_cluster))
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
        mask = (data >= 0)
        if torch.all(mask):
            probs, recons, codes, energy, kl_div = self.forward(data)
        else:
            data[~mask] = 0.0
            probs, recons, codes, energy, kl_div = self.forward(data, mask)
       
        global_loss, global_energy, global_kl_div = self._compute_global_losses(
            energy, kl_div
        )
        entropy = self._compute_entropy(probs)
        sparsity = self._compute_expected_sparsity(probs, codes)

        # Log train metrics
        self.log('Training/loss', global_loss, on_step=True, on_epoch=True)
        self.log('Training/energy', global_energy, on_step=True, on_epoch=True)
        self.log('Training/kl', global_kl_div, on_step=True, on_epoch=True)
        # self.log('Training/acc', acc, on_step=True, on_epoch=True)
        self.log('Training/entropy', entropy, on_step=True, on_epoch=True)
        self.log('Training/sparsity', sparsity, on_step=True, on_epoch=True)

        return global_loss
        
    def validation_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        data, labels = batch

        mask = (data >= 0)
        if torch.all(mask):
            probs, recons, codes, energy, kl_div = self.forward(data)
        else:
            data[~mask] = 0.0
            probs, recons, codes, energy, kl_div = self.forward(data, mask)

        global_loss, global_energy, global_kl_div = self._compute_global_losses(
            energy, kl_div
        )
        entropy = self._compute_entropy(probs)
        sparsity = self._compute_expected_sparsity(probs, codes)

        # Log valid metrics
        self.log('Validation/loss', global_loss, on_step=False, on_epoch=True)
        self.log('Validation/energy', global_energy, on_step=False, on_epoch=True)
        self.log('Validation/kl', global_kl_div, on_step=False, on_epoch=True)
        # self.log('Validation/acc', acc, on_step=False, on_epoch=True)
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
        
        preds = torch.argmax(probs, dim=-1)
        return {'loss': global_loss, 'pred': preds.cpu(), 'targ': labels.cpu()}

    def validation_epoch_end(self, validation_step_outputs):
        preds = torch.cat(
            [out['pred'] for out in validation_step_outputs], dim=0
        )
        targs = torch.cat(
            [out['targ'] for out in validation_step_outputs], dim=0
        )
        preds_np = preds.numpy()
        targs_np = targs.numpy()
        acc = self._compute_cluster_accuracy(preds_np, targs_np, self.num_components)
        nmi = normalized_mutual_info_score(targs_np, preds_np)
        ari = adjusted_rand_score(targs_np, preds_np)
        self.log('Validation/acc', acc, on_epoch=True)
        self.log('Validation/nmi', nmi, on_epoch=True)
        self.log('Validation/ari', ari, on_epoch=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), self.lr)
        return optimizer

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx,
                   optimizer_closure, on_tpu, using_native_amp, using_lbfgs):
        optimizer.step(closure=optimizer_closure)
        self._normalize()


class SingleAutoencoder(MixMATEv2):

    def __init__(
        self,
        num_layers: int,
        input_size: int,
        hidden_size: int,
        step_size: float,
        prox: ProximalOperator,
        beta: float = 1.,
        lr: float = 1e-3
    ) -> None:
        super().__init__(1, num_layers, input_size, hidden_size, step_size, prox, beta, lr)
    
    def validation_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        data, labels = batch
        probs, recons, codes, energy, kl_div = self.forward(data)
        global_loss, global_energy, global_kl_div = self._compute_global_losses(
            energy, kl_div
        )
        entropy = self._compute_entropy(probs)
        sparsity = self._compute_expected_sparsity(probs, codes)

        # Log valid metrics
        self.log('Validation/loss', global_loss, on_step=False, on_epoch=True)
        self.log('Validation/energy', global_energy, on_step=False, on_epoch=True)
        self.log('Validation/kl', global_kl_div, on_step=False, on_epoch=True)
        # self.log('Validation/acc', acc, on_step=False, on_epoch=True)
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
        
        preds = torch.argmax(probs, dim=-1)
        return {'loss': global_loss, 'codes': codes.cpu(), 'targ': labels.cpu()}
    
    def validation_epoch_end(self, validation_step_outputs):
        codes = torch.cat(
            [out['codes'][:, 0, :] for out in validation_step_outputs], dim=0
        )
        targs = torch.cat(
            [out['targ'] for out in validation_step_outputs], dim=0
        )

        codes = codes.numpy()
        targs = targs.numpy()

        kmeans = KMeans(n_clusters=10)
        labels = kmeans.fit_predict(codes)
        self.log(
            'Validation/Kmeans_Acc', 
            self._compute_cluster_accuracy(labels, targs, 10), 
            on_step=False, 
            on_epoch=True
        )