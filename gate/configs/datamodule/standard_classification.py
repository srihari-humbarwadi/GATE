from dataclasses import MISSING, dataclass, field
from typing import Any, Dict, Optional

from gate.configs import get_module_import_path
from gate.configs.datamodule.base import DataLoaderConfig
from gate.configs.datasets.standard_classification import (
    DatasetConfig,
    OmniglotDatasetConfig,
)
from gate.configs.datasets.transforms import (
    cifar10_eval_transforms,
    cifar10_train_transforms,
    cifar100_eval_transforms,
    cifar100_train_transforms,
    omniglot_transform_config,
    stl10_eval_transforms,
    stl10_train_transforms,
)
from gate.configs.string_variables import DATASET_DIR
from gate.datamodules.tf_hub.standard_classification import (
    CIFAR10DataModule,
    CIFAR100DataModule,
    OmniglotDataModule,
)
from gate.datasets.tf_hub.standard.cifar import (
    CIFAR10ClassificationDataset,
    CIFAR100ClassificationDataset,
)


@dataclass
class StandardDatasetTransformConfig:
    input_transform: Optional[Any] = MISSING
    target_transform: Optional[Any] = MISSING


@dataclass
class OmniglotTrainTransformConfig(StandardDatasetTransformConfig):
    input_transform: Optional[Dict] = field(
        default_factory=lambda: dict(
            _target_=get_module_import_path(omniglot_transform_config)
        )
    )
    target_transform: Optional[Dict] = None


@dataclass
class OmniglotEvalTransformConfig:
    input_transform: Optional[Dict] = field(
        default_factory=lambda: dict(
            _target_=get_module_import_path(omniglot_transform_config)
        )
    )
    target_transform: Optional[Dict] = None


@dataclass
class CIFAR10TrainTransformConfig:
    input_transform: Optional[Dict] = field(
        default_factory=lambda: dict(
            _target_=get_module_import_path(cifar10_train_transforms)
        )
    )
    target_transform: Optional[Dict] = None


@dataclass
class CIFAR10EvalTransformConfig:
    input_transform: Optional[Dict] = field(
        default_factory=lambda: dict(
            _target_=get_module_import_path(cifar10_eval_transforms)
        )
    )
    target_transform: Optional[Dict] = None


@dataclass
class CIFAR100TrainTransformConfig:
    input_transform: Optional[Dict] = field(
        default_factory=lambda: dict(
            _target_=get_module_import_path(cifar100_train_transforms)
        )
    )
    target_transform: Optional[Dict] = None


@dataclass
class CIFAR100EvalTransformConfig:
    input_transform: Optional[Dict] = field(
        default_factory=lambda: dict(
            _target_=get_module_import_path(cifar100_eval_transforms)
        )
    )
    target_transform: Optional[Dict] = None


@dataclass
class STL10TrainTransformConfig:
    input_transform: Optional[Dict] = field(
        default_factory=lambda: dict(
            _target_=get_module_import_path(stl10_train_transforms)
        )
    )
    target_transform: Optional[Dict] = None


@dataclass
class STL10EvalTransformConfig:
    input_transform: Optional[Dict] = field(
        default_factory=lambda: dict(
            _target_=get_module_import_path(stl10_eval_transforms)
        )
    )
    target_transform: Optional[Dict] = None


# ------------------------------------------------------------------------------
# datamodule configs


@dataclass
class CIFAR10DatasetConfig(DatasetConfig):
    """
    Class for configuring the CIFAR10 dataset.
    """

    dataset_root: str
    download: bool = True
    split_name: str = "train"
    input_transform: Optional[Any] = None
    target_transform: Optional[Any] = None
    _target_: str = get_module_import_path(CIFAR10ClassificationDataset)


@dataclass
class CIFAR100DatasetConfig(DatasetConfig):
    """
    Class for configuring the CIFAR100 dataset.
    """

    dataset_root: str
    download: bool = True
    split_name: str = "train"
    input_transform: Optional[Any] = None
    target_transform: Optional[Any] = None
    _target_: str = get_module_import_path(CIFAR100ClassificationDataset)


@dataclass
class DataModuleConfig:
    _target_: str
    dataset_config: DatasetConfig
    data_loader_config: DataLoaderConfig


@dataclass
class OmniglotDataModuleConfig:
    dataset_config: OmniglotDatasetConfig = OmniglotDatasetConfig(
        dataset_root=DATASET_DIR
    )
    data_loader_config: DataLoaderConfig = DataLoaderConfig()
    transform_train: Any = OmniglotTrainTransformConfig()
    transform_eval: Any = OmniglotEvalTransformConfig()
    _target_: str = get_module_import_path(OmniglotDataModule)


@dataclass
class CIFAR10DataModuleConfig:
    dataset_config: CIFAR10DatasetConfig = CIFAR10DatasetConfig(
        dataset_root=DATASET_DIR
    )
    data_loader_config: DataLoaderConfig = DataLoaderConfig()
    transform_train: Any = CIFAR10TrainTransformConfig()
    transform_eval: Any = CIFAR10EvalTransformConfig()
    _target_: str = get_module_import_path(CIFAR10DataModule)


@dataclass
class CIFAR100DataModuleConfig:
    dataset_config: CIFAR100DatasetConfig = CIFAR100DatasetConfig(
        dataset_root=DATASET_DIR
    )
    data_loader_config: DataLoaderConfig = DataLoaderConfig()
    transform_train: Any = CIFAR100TrainTransformConfig()
    transform_eval: Any = CIFAR100EvalTransformConfig()
    _target_: str = get_module_import_path(CIFAR100DataModule)
