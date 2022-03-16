import functools
import os
from dataclasses import dataclass
from typing import Dict, Union, List, Optional, Tuple, Any
import torch
import torchmetrics
from numpy import random
import gate
from gate.datasets.data_utils import collate_fn_replace_corrupted

# ------------------------------------------------------------------------------
# General configs
from gate.datasets.datasets import CIFAR10ClassificationDataset
from gate.learners.base import LearnerModule, LinearLayerFineTuningScheme
from gate.models.base import ModelModule, AudioImageResNet
from gate.tasks.base import ImageClassificationTaskModule, TaskModule


@dataclass
class ShapeConfig:
    """
    Modality configuration for the types of processing a model can do.
    """

    image: Optional[Union[Tuple, Dict]] = None
    audio: Optional[Union[Tuple, Dict]] = None
    text: Optional[Union[Tuple, Dict]] = None
    video: Optional[Union[Tuple, Dict]] = None


# ------------------------------------------------------------------------------
# task configs


@dataclass
class TaskConfig:
    output_shape_dict: Union[ShapeConfig, Dict[str, Union[tuple, List]]]
    metric_class_dict: Dict[str, Union[str, torchmetrics.Metric, torch.nn.Module]]
    _target_: Union[str, TaskModule] = gate.tasks.base.TaskModule


@dataclass
class ImageClassificationTaskModuleConfig(TaskConfig):
    _target_: Union[
        str, ImageClassificationTaskModule
    ] = gate.tasks.base.ImageClassificationTaskModule


# ------------------------------------------------------------------------------
# model configs


@dataclass
class ModelConfig:
    input_modality_shape_config: ShapeConfig
    _target_: str = "gate.models.base.ModelModule"


@dataclass
class AudioImageResNetConfig(ModelConfig):
    model_name_to_download: str = "resnet18"
    pretrained: bool = False
    audio_kernel_size: int = 3
    _target_: Union[str, AudioImageResNet] = gate.models.base.AudioImageResNet


# ------------------------------------------------------------------------------
# Learner configs


@dataclass
class LearnerModalityConfig:
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
    model: torch.nn.Module
    task_config: TaskConfig
    modality_config: LearnerModalityConfig
    _target_: Union[str, LearnerModule] = gate.learners.base.LearnerModule


@dataclass
class LinearLayerFineTuningSchemeConfig(LearnerConfig):
    fine_tune_all_layers = False
    max_epochs: int = 100
    min_learning_rate: float = 1e-6
    lr: float = 1e-3
    betas: Tuple[float] = (0.9, 0.999)
    eps: float = 1e-8
    weight_decay: float = 0.0
    amsgrad: bool = False
    _target_: Union[
        str, LinearLayerFineTuningScheme
    ] = gate.learners.base.LinearLayerFineTuningScheme


# ------------------------------------------------------------------------------
# dataset configs


@dataclass
class DatasetConfig:
    """
    Class for configuring the CIFAR dataset.
    """

    dataset_name: str = "dataset"
    dataset_root: str = f"{os.environ['DATASETS_DIR']}/dataset"


@dataclass
class CIFAR10DatasetConfig(DatasetConfig):
    """
    Class for configuring the CIFAR dataset.
    """

    dataset_name: str = "cifar10"
    dataset_root: str = f"{os.environ['DATASETS_DIR']}/cifar10"
    val_set_percentage: float = 0.1
    download: bool = True
    _target_: Union[
        str, CIFAR10ClassificationDataset
    ] = gate.datasets.datasets.CIFAR10ClassificationDataset


@dataclass
class CIFAR100DatasetConfig(DatasetConfig):
    """
    Class for configuring the CIFAR dataset.
    """

    dataset_name: str = "cifar100"
    dataset_root: str = f"{os.environ['DATASETS_DIR']}/cifar100"
    val_set_percentage: float = 0.1
    download: bool = True
    _target_: Union[
        str, CIFAR10ClassificationDataset
    ] = gate.datasets.datasets.CIFAR100ClassificationDataset


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
