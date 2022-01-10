import types

import numpy as np
import torchvision as tv
from torchvision.transforms import ToTensor
import torch
from torch import Tensor
from torch.utils.data import DataLoader
from torch.utils.data import ConcatDataset

from mixmate.datasets.dataset import Dataset


class TorchVisionDataset(Dataset):

    def __init__(
        self,
        name: str,
        data_path: str,
        download: bool,
        batch_size: int,
        erase_prob: float = 0.0,
        erase_frac: float = 0.0
    ) -> None:
    
        cls = getattr(tv.datasets, name)

        if erase_prob > 0.0 and erase_frac > 0.0:
            cls = dataset_with_masked_outputs(cls)

        train_dataset = cls(
            root=data_path, train=True, download=download, transform=ToTensor()
        )
        val_dataset = cls(
            root=data_path, train=False, download=download, transform=ToTensor()
        )
        
        if erase_prob > 0.0 and erase_frac > 0.0:
            train_dataset.masks = self.generate_erase_masks(train_dataset.data, erase_prob, erase_frac)
            val_dataset.masks = self.generate_erase_masks(val_dataset.data, erase_prob, erase_frac)
        
        joint_dataset = ConcatDataset([train_dataset, val_dataset])
        self._train_loader = DataLoader(
            joint_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=4
        )
        self._val_loader = DataLoader(
            joint_dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=4
        )

    @staticmethod
    def generate_erase_masks(data: Tensor, erase_prob: float, erase_frac: float) -> Tensor:
        rand_vals = torch.rand_like(data.flatten(start_dim=1, end_dim=-1), dtype=torch.float)
        quantiles = torch.quantile(rand_vals, q=erase_frac, dim=-1, keepdim=True)
        erase_mask = torch.rand_like(quantiles) < erase_prob
        quantiles[~erase_mask] = -np.inf
        masks = (rand_vals >= quantiles)
        masks = masks.view_as(data)
        return masks

    @property
    def train_loader(self) -> DataLoader:
        return self._train_loader

    @property
    def valid_loader(self) -> DataLoader:
        return self._val_loader
    
    @staticmethod
    def mask_dataset_outputs(dataset: torch.utils.data.Dataset, masks: Tensor) -> None:
        """
        Modifies the given Dataset object to return a masked image when getting item.

        From https://discuss.pytorch.org/t/how-to-retrieve-the-sample-indices-of-a-mini-batch/7948/19.

        """
        def __getitem__(self, index):
            data, target = dataset.__class__.__getitem__(self, index)
            data[~masks[index]] = -1
            return data, target
        dataset.__getitem__ = types.MethodType(__getitem__, dataset)


def dataset_with_masked_outputs(cls):
    """
    Modifies the given Dataset class to return masked data.
    """

    def __getitem__(self, index):
        data, target = cls.__getitem__(self, index)
        data[..., ~self.masks[index]] = -1
        return data, target

    return type(cls.__name__, (cls,), {
        '__getitem__': __getitem__,
    })