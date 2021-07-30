import torchvision as tv
from torchvision.transforms import ToTensor
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
        split: bool = False
    ) -> None:
        dataset = getattr(tv.datasets, name)
        train_dataset = dataset(
            root=data_path, train=True, download=download, transform=ToTensor()
        )
        val_dataset = dataset(
            root=data_path, train=False, download=download, transform=ToTensor()
        )
        if split:
            self._train_loader = DataLoader(
                train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=4
            )
            self._val_loader = DataLoader(
                val_dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=4
            )
        else:
            joint_dataset = ConcatDataset([train_dataset, val_dataset])
            self._train_loader = DataLoader(
                joint_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=4
            )
            self._val_loader = DataLoader(
                joint_dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=4
            )

    @property
    def train_loader(self) -> DataLoader:
        return self._train_loader

    @property
    def valid_loader(self) -> DataLoader:
        return self._val_loader