import pathlib
from typing import Union, Optional, Callable

import torch
from torchvision.datasets import CIFAR10, CIFAR100

from gate.base.utils.loggers import get_logger

log = get_logger(__name__)


class CIFAR10ClassificationDataset(CIFAR10):
    def __init__(
        self,
        root: Union[str, pathlib.Path],
        train: bool,
        download: bool,
        input_transform: Optional[Callable],
        target_transform: Optional[Callable] = None,
    ):
        super(CIFAR10ClassificationDataset, self).__init__(
            root=root,
            train=train,
            download=download,
            transform=input_transform,
            target_transform=target_transform,
        )
        self.name = self.__class__.__name__

    def __getitem__(self, index):
        x, y = super().__getitem__(index)
        return {"image": x}, {"image": torch.ones(size=(1,)).type(torch.LongTensor) * y}


class CIFAR100ClassificationDataset(CIFAR100):
    def __init__(
        self,
        root: Union[str, pathlib.Path],
        train: bool,
        download: bool,
        input_transform: Optional[Callable],
        target_transform: Optional[Callable] = None,
    ):
        super(CIFAR100ClassificationDataset, self).__init__(
            root=root,
            train=train,
            download=download,
            transform=input_transform,
            target_transform=target_transform,
        )
        self.name = self.__class__.__name__

    def __getitem__(self, index):
        x, y = super().__getitem__(index)
        return {"image": x}, {"image": torch.ones(size=(1,)).type(torch.LongTensor) * y}


class CIFAR10ReconstructionDataset(CIFAR10):
    def __init__(
        self,
        root: Union[str, pathlib.Path],
        train: bool,
        download: bool,
        input_transform: Optional[Callable],
        target_transform: Optional[Callable] = None,
    ):
        super(CIFAR10ReconstructionDataset, self).__init__(
            root=root, train=train, download=download, transform=input_transform
        )
        self.target_transforms = target_transform
        self.name = self.__class__.__name__

    def __getitem__(self, index):
        x, y = super().__getitem__(index)
        input_images = x.clone()
        target_images = x.clone()

        if self.target_transforms:
            if isinstance(self.target_transforms, list):
                for transform in self.target_transforms:
                    target_images = transform(target_images)
            else:
                target_images = self.target_transforms(target_images)

        return {"image": input_images}, {"image": target_images}


class CIFAR100ReconstructionDataset(CIFAR100):
    def __init__(
        self,
        root: Union[str, pathlib.Path],
        train: bool,
        download: bool,
        input_transform: Optional[Callable],
        target_transform: Optional[Callable] = None,
    ):
        super(CIFAR100ReconstructionDataset, self).__init__(
            root=root, train=train, download=download, transform=input_transform
        )
        self.target_transforms = target_transform
        self.name = self.__class__.__name__

    def __getitem__(self, index):
        x, y = super().__getitem__(index)
        input_images = x.clone()
        target_images = x.clone()
        if self.target_transforms:
            if isinstance(self.target_transforms, list):
                for transform in self.target_transforms:
                    target_images = transform(target_images)
            else:
                target_images = self.target_transforms(target_images)

        return {"image": input_images}, {"image": target_images}
