from typing import Optional

import torchvision as tv
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data.dataset import Subset
import torch

from mixmate.datasets.dataset import Dataset


class MNIST(Dataset):

    def __init__(
        self,
        data_path: str,
        download: bool,
        classes_only: Optional[str] = None,
        valid_size: float = 10000,
        train_batch_size: int = 32,
        valid_batch_size: int = 32,
        center_data: bool = False
    ) -> None:
        self._dataset = tv.datasets.MNIST(
            root=data_path, train=True, download=download, transform=ToTensor()
        )
        if classes_only is not None:
            self._classes = np.array([int(c) for c in classes_only])
            idx = np.isin(self._dataset.targets, self._classes)
            self._dataset.targets = self._dataset.targets[idx]
            self._dataset.data = self._dataset.data[idx]
        else:
            self._classes = np.array(range(10))
        
        self.center_data = center_data
        if self.center_data:
            self._dataset.data = self._dataset.data.to(torch.float)
            self._dataset.data = self._dataset.data - self._dataset.data.mean(dim=0)

        num_data = len(self._dataset.data)
        train_size = num_data - valid_size
        mnist_train, mnist_valid = random_split(self._dataset, [train_size, valid_size])

        self._train_loader = DataLoader(
            mnist_train, batch_size=train_batch_size, shuffle=True, drop_last=True, num_workers=4
        )
        self._valid_loader = DataLoader(
            mnist_valid, batch_size=valid_batch_size, shuffle=False, drop_last=False, num_workers=4
        )

    @property
    def train_loader(self) -> DataLoader:
        return self._train_loader

    @property
    def valid_loader(self) -> DataLoader:
        return self._valid_loader


class FashionMNIST(Dataset):

    def __init__(
        self,
        data_path: str,
        download: bool,
        valid_size: float = 10000,
        train_batch_size: int = 32,
        valid_batch_size: int = 32
    ) -> None:
        self._dataset = tv.datasets.FashionMNIST(
            root=data_path, train=True, download=download, transform=ToTensor()
        )

        num_data = len(self._dataset.data)
        train_size = num_data - valid_size
        mnist_train, mnist_valid = random_split(self._dataset, [train_size, valid_size])

        self._train_loader = DataLoader(
            mnist_train, batch_size=train_batch_size, shuffle=True, drop_last=True, num_workers=4
        )
        self._valid_loader = DataLoader(
            mnist_valid, batch_size=valid_batch_size, shuffle=False, drop_last=False, num_workers=4
        )

    @property
    def train_loader(self) -> DataLoader:
        return self._train_loader

    @property
    def valid_loader(self) -> DataLoader:
        return self._valid_loader


class USPS(Dataset):

    def __init__(
        self,
        data_path: str,
        download: bool,
        valid_size: float = 10000,
        train_batch_size: int = 32,
        valid_batch_size: int = 32
    ) -> None:
        self._dataset = tv.datasets.USPS(
            root=data_path, train=True, download=download, transform=ToTensor()
        )

        num_data = len(self._dataset.data)
        train_size = num_data - valid_size
        mnist_train, mnist_valid = random_split(self._dataset, [train_size, valid_size])

        self._train_loader = DataLoader(
            mnist_train, batch_size=train_batch_size, shuffle=True, drop_last=True, num_workers=4
        )
        self._valid_loader = DataLoader(
            mnist_valid, batch_size=valid_batch_size, shuffle=False, drop_last=False, num_workers=4
        )

    @property
    def train_loader(self) -> DataLoader:
        return self._train_loader

    @property
    def valid_loader(self) -> DataLoader:
        return self._valid_loader


class SemiSupervisedMNIST(Dataset):

    def __init__(
        self,
        data_path: str,
        download: bool,
        classes_only: str = '0123456789',
        valid_size: float = 10000,
        labeled_per_class: int = 0,
        unlabeled_batch_size: int = 32,
        labeled_batch_size: int = 32,
        valid_batch_size: int = 32
    ) -> None:
        self._dataset = tv.datasets.MNIST(
            root=data_path, train=True, download=download, transform=ToTensor()
        )

        self._classes = np.array([int(c) for c in classes_only])
        idx = np.isin(self._dataset.targets, self._classes)
        self._dataset.targets = self._dataset.targets[idx]
        self._dataset.data = self._dataset.data[idx]

        num_data = len(self._dataset.data)
        train_size = num_data - valid_size
        mnist_train, mnist_valid = random_split(self._dataset, [train_size, valid_size])

        labeled_idxs = []
        unlabeled_idxs = []
        for c in self._classes:
            mask_c = (self._dataset.targets[mnist_train.indices] == c)
            idx_c = np.array(mnist_train.indices)[mask_c]
            labeled_idxs.append(idx_c[:labeled_per_class])
            unlabeled_idxs.append(idx_c[labeled_per_class:])
        labeled_idx = np.concatenate(labeled_idxs)
        unlabeled_idx = np.concatenate(unlabeled_idxs)

        mnist_unlabeled = Subset(mnist_train.dataset, indices=unlabeled_idx)
        mnist_labeled = Subset(mnist_train.dataset, indices=labeled_idx)

        self._train_unlabeled_loader = DataLoader(
            mnist_unlabeled,
            batch_size=unlabeled_batch_size,
            shuffle=True,
            drop_last=True
        )
        self._train_labeled_loader = DataLoader(
            mnist_labeled,
            batch_size=labeled_batch_size,
            shuffle=True,
            drop_last=True
        )
        self._valid_loader = DataLoader(
            mnist_valid, batch_size=valid_batch_size, shuffle=False, drop_last=False
        )

    @property
    def train_loader(self) -> DataLoader:
        return self._train_unlabeled_loader

    @property
    def valid_loader(self) -> DataLoader:
        return self._valid_loader
