from typing import Any

from gate.configs.datamodule.base import DataLoaderConfig
from gate.configs.datasets.standard_classification import (
    CIFAR10DatasetConfig,
    CIFAR100DatasetConfig,
    OmniglotDatasetConfig,
)
from gate.datamodules.image_classification import PreSplitDataModule


class OmniglotDataModule(PreSplitDataModule):
    def __init__(
        self,
        dataset_config: OmniglotDatasetConfig,
        data_loader_config: DataLoaderConfig,
        transform_train: Any,
        transform_eval: Any,
    ):
        super(OmniglotDataModule, self).__init__(
            dataset_config=dataset_config,
            data_loader_config=data_loader_config,
            transform_train=transform_train,
            transform_eval=transform_eval,
            split_name_to_phase_dict=dict(
                train="train", val="small1", test="test"
            ),
        )


class CIFAR10DataModule(PreSplitDataModule):
    def __init__(
        self,
        dataset_config: CIFAR10DatasetConfig,
        data_loader_config: DataLoaderConfig,
        transform_train: Any,
        transform_eval: Any,
    ):
        super(CIFAR10DataModule, self).__init__(
            dataset_config=dataset_config,
            data_loader_config=data_loader_config,
            transform_train=transform_train,
            transform_eval=transform_eval,
            split_name_to_phase_dict=dict(
                train="train", val="test", test="test"
            ),
        )


class CIFAR100DataModule(PreSplitDataModule):
    def __init__(
        self,
        dataset_config: CIFAR100DatasetConfig,
        data_loader_config: DataLoaderConfig,
        transform_train: Any,
        transform_eval: Any,
    ):
        super(CIFAR100DataModule, self).__init__(
            dataset_config=dataset_config,
            data_loader_config=data_loader_config,
            transform_train=transform_train,
            transform_eval=transform_eval,
            split_name_to_phase_dict=dict(
                train="train", val="test", test="test"
            ),
        )
