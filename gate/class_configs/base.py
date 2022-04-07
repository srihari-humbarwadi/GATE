import os
from dataclasses import dataclass, field
from typing import Dict, Union, List, Optional, Tuple, Any

import torch


# ------------------------------------------------------------------------------
# General configs
# from gate.datasets.data_utils import collate_fn_replace_corrupted
from dotted_dict import DottedDict

from gate.datasets.data_utils import collate_fn_replace_corrupted

# ------------------------------------------------------------------------------
# Config rules:

# Structured Configs use Python dataclasses to describe your configuration
# structure and types. They enable:
#
# - Runtime type checking as you compose or mutate your config
# - Static type checking when using static type checkers (mypy, PyCharm, etc.)
#
# Structured Configs supports:
# - Primitive types (int, bool, float, str, Enums)
# - Nesting of Structured Configs
# - Containers (List and Dict) containing primitives or Structured Configs
# - Optional fields
#
# Structured Configs Limitations:
# - Union types are not supported (except Optional)
# - User methods are not supported
#
# There are two primary patterns for using Structured configs
# - As a config, in place of configuration files (often a starting place)
# - As a config schema validating configuration files (better for complex use cases)
#
# With both patterns, you still get everything Hydra has to offer
# (config composition, Command line overrides etc).


@dataclass
class ShapeConfig:
    """
    Modality configuration for the types of processing a model can do.
    """

    image: Optional[Any] = None
    audio: Optional[Any] = None
    text: Optional[Any] = None
    video: Optional[Any] = None


# ------------------------------------------------------------------------------
# task configs

#
@dataclass
class TaskConfig:
    output_shape_dict: ShapeConfig
    # metric_class_dict: Dict[str, Union[str, torch.nn.Module]]
    _target_: str = "gate.tasks.base.TaskModule"


@dataclass
class ImageClassificationTaskModuleConfig(TaskConfig):
    _target_: str = "gate.tasks.classification.ImageClassificationTaskModule"


# ------------------------------------------------------------------------------
# model configs


@dataclass
class ModelConfig:
    input_shape_dict: ShapeConfig
    _target_: str = "gate.models.base.ModelModule"


@dataclass
class ImageResNetConfig(ModelConfig):
    model_name_to_download: str = "resnet18"
    pretrained: bool = False
    _target_: str = "gate.models.resnet.ImageResNet"


@dataclass
class AudioImageResNetConfig(ImageResNetConfig):
    model_name_to_download: str = "resnet18"
    pretrained: bool = False
    audio_kernel_size: int = 3
    _target_: str = "gate.models.resnet.AudioImageResNet"


# ------------------------------------------------------------------------------
# Learner configs


@dataclass
class ModalitiesSupportedConfig:
    """
    Modality configuration for the types of processing a model can do.
    """

    image: bool = False
    audio: bool = False
    text: bool = False
    video: bool = False
    image_text: bool = False
    audio_text: bool = False
    video_text: bool = False
    image_audio: bool = False
    image_video: bool = False
    audio_video: bool = False
    image_audio_text: bool = False
    image_video_text: bool = False
    audio_video_text: bool = False
    image_audio_video: bool = False
    image_audio_video_text: bool = False


@dataclass
class LearnerConfig:
    _target_: str = "gate.learners.base.LearnerModule"


@dataclass
class LinearLayerFineTuningSchemeConfig(LearnerConfig):
    optimizer_config: Dict = field(default_factory=dict)
    lr_scheduler_config: Dict = field(default_factory=dict)
    fine_tune_all_layers: bool = False
    max_epochs: int = 100
    min_learning_rate: float = 1e-6
    lr: float = 1e-3
    betas: List[float] = field(default_factory=lambda: [0.9, 0.999])
    eps: float = 1e-8
    weight_decay: float = 0.0
    amsgrad: bool = False
    _target_: str = "gate.learners.single_layer_fine_tuning.LinearLayerFineTuningScheme"


# ------------------------------------------------------------------------------
# dataset configs


@dataclass
class DatasetConfig:
    """
    Class for configuring the CIFAR dataset.
    """

    dataset_name: str = "dataset"
    dataset_root: str = f"{os.getenv('DATASET_DIR') or 'datasets'}/dataset"


@dataclass
class CIFAR10DatasetConfig(DatasetConfig):
    """
    Class for configuring the CIFAR dataset.
    """

    dataset_name: str = "cifar10"
    dataset_root: str = f"{os.getenv('DATASET_DIR') or 'datasets'}/cifar10"
    modality_config: ModalitiesSupportedConfig = ModalitiesSupportedConfig(image=True)
    input_shape_dict: DottedDict = field(
        default_factory=lambda: dict(image=dict(channels=3, height=32, width=32))
    )
    val_set_percentage: float = 0.1
    download: bool = True
    _target_: str = "gate.datasets.cifar.CIFAR10ClassificationDataset"


@dataclass
class CIFAR100DatasetConfig(DatasetConfig):
    """
    Class for configuring the CIFAR dataset.
    """

    dataset_name: str = "cifar100"
    dataset_root: str = f"{os.getenv('DATASET_DIR') or 'datasets'}/cifar100"
    input_shape_dict: DottedDict = field(
        default_factory=lambda: dict(image=dict(channels=3, height=32, width=32))
    )
    modality_config: ModalitiesSupportedConfig = ModalitiesSupportedConfig(image=True)
    val_set_percentage: float = 0.1
    download: bool = True
    _target_: str = "gate.datasets.cifar.CIFAR100ClassificationDataset"


# ------------------------------------------------------------------------------
# data loader configs


@dataclass
class DataLoaderConfig:
    seed: int = 0
    train_batch_size: int = 64
    val_batch_size: int = 64
    test_batch_size: int = 64
    num_workers: int = 0
    pin_memory: bool = False
    train_drop_last: bool = False
    eval_drop_last: bool = False
    train_shuffle: bool = True
    eval_shuffle: bool = True
    prefetch_factor: int = 2
    persistent_workers: bool = False
    collate_fn: Any = collate_fn_replace_corrupted
