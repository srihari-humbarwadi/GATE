import multiprocessing

import hydra.utils
import pytest

from gate.base.utils.loggers import get_logger
from gate.configs.datamodule import CIFAR10DataModuleConfig
from gate.configs.datamodule.base import DataLoaderConfig
from gate.configs.datamodule.standard_classification import (
    CIFAR10EvalTransformConfig,
    CIFAR10TrainTransformConfig,
)
from gate.configs.datasets.standard_classification import CIFAR10DatasetConfig

log = get_logger(__name__, set_default_handler=True)


@pytest.mark.parametrize("batch_size", [multiprocessing.cpu_count() * 2])
@pytest.mark.parametrize("num_workers", [multiprocessing.cpu_count()])
@pytest.mark.parametrize("pin_memory", [True])
@pytest.mark.parametrize("drop_last", [True])
@pytest.mark.parametrize("shuffle", [True])
@pytest.mark.parametrize("prefetch_factor", [2])
@pytest.mark.parametrize("persistent_workers", [True])
def test_omniglot_fewshot_datamodules(
    batch_size,
    num_workers,
    pin_memory,
    drop_last,
    shuffle,
    prefetch_factor,
    persistent_workers,
):
    dataset_config = CIFAR10DatasetConfig(
        dataset_root=".test/datasets/cifar10", download=True
    )

    data_loader_config = DataLoaderConfig(
        train_batch_size=100,
        val_batch_size=100,
        test_batch_size=100,
        pin_memory=pin_memory,
        train_drop_last=drop_last,
        eval_drop_last=drop_last,
        train_shuffle=shuffle,
        eval_shuffle=shuffle,
        prefetch_factor=prefetch_factor,
        persistent_workers=persistent_workers,
        num_workers=num_workers,
    )

    datamodule_config = CIFAR10DataModuleConfig(
        dataset_config=dataset_config,
        data_loader_config=data_loader_config,
        transform_train=CIFAR10TrainTransformConfig(),
        transform_eval=CIFAR10EvalTransformConfig(),
    )

    datamodule = hydra.utils.instantiate(datamodule_config, _recursive_=False)

    datamodule.setup(stage="fit")

    for idx, item in enumerate(datamodule.train_dataloader()):
        x, y = item
        assert x["image"].shape[1:] == (3, 32, 32)
        break
