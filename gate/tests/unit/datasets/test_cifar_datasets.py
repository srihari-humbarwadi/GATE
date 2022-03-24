import inspect
import logging
import pathlib

import pytest
import torch
import torchvision.transforms as transforms
from rich.logging import RichHandler

from gate.datamodules.custom_transforms import UnNormalize
from gate.datasets.cifar import (
    CIFAR10ClassificationDataset,
    CIFAR100ClassificationDataset,
    CIFAR10ReconstructionDataset,
    CIFAR100ReconstructionDataset,
)

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
ch = RichHandler()
ch.setLevel(logging.INFO)

# create formatter
formatter = logging.Formatter("%(levelname)s - %(message)s")

# add formatter to ch
ch.setFormatter(formatter)

# add ch to logger
log.addHandler(ch)


@pytest.mark.parametrize(
    "dataset",
    [
        CIFAR10ClassificationDataset,
        CIFAR100ClassificationDataset,
        CIFAR10ReconstructionDataset,
        CIFAR100ReconstructionDataset,
    ],
)
@pytest.mark.parametrize("train", [True, False])
@pytest.mark.parametrize("download", [True, False])
def test_cifar_datasets(dataset, train, download):
    log.info("Testing dataset: %s", dataset.__name__)
    input_transforms = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]
            ),
        ]
    )

    argument_names = inspect.signature(dataset.__init__).parameters.keys()
    log.info(f"Items: {argument_names} {'input_transform' in argument_names}")
    target_transforms = None

    if "ReconstructionDataset" in dataset.__name__:
        target_transforms = transforms.Compose(
            [
                UnNormalize(
                    mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]
                ),
                transforms.ToPILImage(),
                transforms.RandomRotation(45),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]
                ),
            ]
        )

    dataset_instance = dataset(
        root=pathlib.Path("tests/data/cifar/"),
        input_transform=input_transforms,
        target_transform=target_transforms,
        train=train,
        download=True,
    )

    for i in range(len(dataset_instance)):
        item = dataset_instance[i]
        x, y = item
        assert x["image"].shape == (3, 32, 32)
        assert torch.is_tensor(x["image"])
        assert torch.is_tensor(y["image"])
        break
