from typing import Iterable, Tuple

from torch import Tensor


class Dataset:

    @property
    def train_loader(self) -> Iterable[Tuple[Tensor, ...]]:
        pass

    @property
    def valid_loader(self) -> Iterable[Tuple[Tensor, ...]]:
        pass
