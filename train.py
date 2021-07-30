from omegaconf import DictConfig
from hydra.utils import instantiate
import pytorch_lightning as pl
import hydra
import torch
import pdb

from mixmate.mixture import MixMATE


@hydra.main(config_path='configs/', config_name='train')
def train(config: DictConfig) -> None:
    dataset = instantiate(config.dataset)
    mixmate = MixMATE(**config.mixmate)
    trainer = pl.Trainer(**config.trainer)

    dataset._train_unlabeled_loader.dataset.indices
    
    num_prelabeled = config.num_prelabeled
    if num_prelabeled is not None:
        for i in range(mixmate.num_components):
            imgs = torch.tensor(
                dataset._dataset.data[dataset._dataset.targets == i],
                dtype=torch.float
            )
            # Initialize with PCA
            ae = mixmate.autoencoders[i]
            if num_prelabeled < ae.hidden_size:
                factor = int(ae.hidden_size / num_prelabeled)
                imgs = imgs.view((-1, 28 * 28))[:num_prelabeled]
                ae.W.data = torch.flatten(
                    imgs.expand((factor, num_prelabeled, 28 * 28)),
                    start_dim=0,
                    end_dim=1
                )
            else:
                V = imgs.view((-1, 28 * 28))[:num_prelabeled].svd().V
                ae.W.data = V[:, :ae.hidden_size].T


    trainer.fit(mixmate, dataset.train_loader, dataset.valid_loader)

if __name__ == "__main__":
    train()
