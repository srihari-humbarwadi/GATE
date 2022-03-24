import logging

import pytest
import torch
from rich.logging import RichHandler

from gate.class_configs.base import (
    CIFAR10DatasetConfig,
    DataLoaderConfig,
    CIFAR100DatasetConfig,
)
from gate.datamodules.cifar import CIFAR10DataModule, CIFAR100DataModule

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
    "datamodule",
    [
        CIFAR10DataModule,
        CIFAR100DataModule,
    ],
)
@pytest.mark.parametrize("batch_size", [1, 2])  # , 4, 8, 16, 32, 64, 128, 256, 512])
@pytest.mark.parametrize("num_workers", [1, 4])
@pytest.mark.parametrize("pin_memory", [True, False])
@pytest.mark.parametrize("drop_last", [True, False])
@pytest.mark.parametrize("shuffle", [True, False])
@pytest.mark.parametrize("prefetch_factor", [1, 2])
@pytest.mark.parametrize("persistent_workers", [True, False])
def test_single_layer_fine_tuning(
    datamodule,
    batch_size,
    num_workers,
    pin_memory,
    drop_last,
    shuffle,
    prefetch_factor,
    persistent_workers,
):
    # log.info(f"Testing datamodule: {datamodule.__name__}")

    dataset_config = (
        CIFAR10DatasetConfig()
        if "CIFAR10" in datamodule.__name__
        else CIFAR100DatasetConfig()
    )

    data_loader_config = DataLoaderConfig(
        train_batch_size=batch_size,
        val_batch_size=batch_size,
        test_batch_size=batch_size,
        pin_memory=pin_memory,
        train_drop_last=drop_last,
        eval_drop_last=drop_last,
        train_shuffle=shuffle,
        eval_shuffle=shuffle,
        prefetch_factor=prefetch_factor,
        persistent_workers=persistent_workers,
        num_workers=num_workers,
    )

    datamodule = datamodule(
        dataset_config=dataset_config, data_loader_config=data_loader_config
    )

    datamodule.setup(stage="fit")

    for idx, item in enumerate(datamodule.train_dataloader()):
        x, y = item
        assert x["image"].shape == (batch_size, 3, 32, 32)
        assert torch.is_tensor(x["image"])
        assert torch.is_tensor(y["image"])
        break
