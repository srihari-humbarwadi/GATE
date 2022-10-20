from dataclasses import dataclass
from typing import Any, Optional

from gate.configs import get_module_import_path
from gate.datasets.tf_hub.standard.cifar import (
    CIFAR10ClassificationDataset,
    CIFAR100ClassificationDataset,
)
from gate.datasets.tf_hub.standard.omniglot import OmniglotClassificationDataset


@dataclass
class DatasetConfig:
    """
    Class for configuring the CIFAR dataset.
    """

    dataset_root: str


@dataclass
class CIFAR10DatasetConfig(DatasetConfig):
    """
    Class for configuring the CIFAR dataset.
    """

    dataset_root: str
    download: bool = True
    val_set_percentage: float = 0.1
    train: Optional[bool] = None
    input_transform: Optional[Any] = None
    target_transform: Optional[Any] = None
    _target_: str = get_module_import_path(CIFAR10ClassificationDataset)


@dataclass
class CIFAR100DatasetConfig(DatasetConfig):
    """
    Class for configuring the CIFAR dataset.
    """

    dataset_root: str
    download: bool = True
    val_set_percentage: float = 0.1
    train: Optional[bool] = None
    input_transform: Optional[Any] = None
    target_transform: Optional[Any] = None
    _target_: str = get_module_import_path(CIFAR100ClassificationDataset)


@dataclass
class PreSplitDatasetConfig(DatasetConfig):
    """
    Class for configuring the CIFAR dataset.
    """

    dataset_root: str
    download: bool
    _target_: str
    input_transform: Optional[Any] = None
    target_transform: Optional[Any] = None
    split_name: Optional[str] = None


@dataclass
class OmniglotDatasetConfig(DatasetConfig):
    """
    Class for configuring the CIFAR dataset.
    """

    dataset_root: str
    download: bool = True
    split_name: str = "train"
    input_transform: Optional[Any] = None
    target_transform: Optional[Any] = None
    _target_: str = get_module_import_path(OmniglotClassificationDataset)
