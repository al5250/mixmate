from omegaconf import DictConfig
from hydra.utils import instantiate
import pytorch_lightning as pl
import hydra
import torch
import pdb
import numpy as np
import torch.nn.functional as F

from mixmate.mixture import SemiSupervisedMixMATEv2


@hydra.main(config_path='configs/', config_name='trainv2_semi')
def trainv2(config: DictConfig) -> None:
    dataset = instantiate(config.dataset)
    mixmate = SemiSupervisedMixMATEv2(dataset._train_labeled_loader, **config.mixmate)
    trainer = pl.Trainer(**config.trainer)

    targets = dataset._train_labeled_loader.dataset.dataset.targets
    imgs = dataset._train_labeled_loader.dataset.dataset.data
    idx = dataset._train_labeled_loader.dataset.indices
    labeled_targets = targets[idx]
    labeled_imgs = imgs[idx]

    for i in range(mixmate.num_components):
        class_imgs = torch.tensor(
            labeled_imgs[labeled_targets == i],
            dtype=torch.float
        )

        class_imgs = class_imgs.view((-1, 28 * 28))[:mixmate.hidden_size]
        class_imgs = F.normalize(
            class_imgs - class_imgs.mean(dim=1, keepdim=True),
            dim=1
        )
        mixmate.W.data[i] = class_imgs.T

    mixmate._normalize()

    # data = dataset._dataset.data
    # num_data = len(data)
    # for i in range(mixmate.num_components):
    #     idx = np.random.choice(num_data, mixmate.hidden_size, replace=False)
    #     imgs = torch.tensor(data[idx], dtype=torch.float).detach().clone()
    #     mixmate.W.data[i] = F.normalize(imgs.view((-1, 28 * 28)).transpose(0, 1)

    trainer.fit(mixmate, dataset.train_loader, dataset.valid_loader)

if __name__ == "__main__":
    trainv2()
