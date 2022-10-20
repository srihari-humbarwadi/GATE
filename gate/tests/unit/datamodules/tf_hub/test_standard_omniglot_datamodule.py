import multiprocessing

import hydra.utils
import pytest

from gate.base.utils.loggers import get_logger
from gate.configs.datamodule import OmniglotDataModuleConfig
from gate.configs.datamodule.base import DataLoaderConfig
from gate.configs.datamodule.standard_classification import (
    OmniglotTrainTransformConfig,
)
from gate.configs.datasets.standard_classification import OmniglotDatasetConfig

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
    dataset_config = OmniglotDatasetConfig(
        dataset_root=".test/datasets/omniglot", download=True
    )

    transform_train = OmniglotTrainTransformConfig()

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

    datamodule_config = OmniglotDataModuleConfig(
        dataset_config=dataset_config,
        data_loader_config=data_loader_config,
        transform_train=transform_train,
        transform_eval=transform_train,
    )

    datamodule = hydra.utils.instantiate(datamodule_config, _recursive_=False)

    datamodule.setup(stage="fit")

    for idx, item in enumerate(datamodule.train_dataloader()):
        x, y = item
        x["image"].shape[1:] == (1, 28, 28)
        break
