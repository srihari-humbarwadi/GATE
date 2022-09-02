import multiprocessing

import hydra
import pytest
import torch
from dotted_dict import DottedDict

from gate.base.utils.loggers import get_logger
from gate.configs.datamodule import CIFAR10DataModuleConfig
from gate.configs.datamodule.base import DataLoaderConfig
from gate.configs.datasets.standard_classification import CIFAR10DatasetConfig
from gate.configs.datasets.transforms import cifar10_train_transforms
from gate.configs.learner import (
    CosineAnnealingLRConfig,
    FullModelFineTuningSchemeConfig,
    SingleLinearLayerFineTuningSchemeConfig,
)
from gate.configs.model.timm_model_configs import TimmImageResNet18Config
from gate.configs.task.image_classification import ImageClassificationTaskConfig
from gate.models.timm_hub import TimmImageModel
from gate.tasks.standard_classification import ImageClassificationTaskModule
from gate.train_eval_agents.base import TrainingEvaluationAgent

log = get_logger(__name__, set_default_handler=True)


@pytest.mark.parametrize(
    "model_config",
    [TimmImageResNet18Config],
)
@pytest.mark.parametrize(
    "task_config",
    [
        ImageClassificationTaskConfig(
            output_shape_dict=dict(
                image=dict(
                    num_classes=10,
                )
            )
        ),
    ],
)
@pytest.mark.parametrize(
    "learner_config",
    [
        FullModelFineTuningSchemeConfig(
            lr_scheduler_config=CosineAnnealingLRConfig(
                T_max=10, batch_size=multiprocessing.cpu_count() * 2
            ),
        ),
        SingleLinearLayerFineTuningSchemeConfig(
            lr_scheduler_config=CosineAnnealingLRConfig(
                T_max=10, batch_size=multiprocessing.cpu_count() * 2
            )
        ),
    ],
)
@pytest.mark.parametrize("batch_size", [multiprocessing.cpu_count() * 2])
@pytest.mark.parametrize("num_workers", [multiprocessing.cpu_count()])
@pytest.mark.parametrize("pin_memory", [True])
@pytest.mark.parametrize("drop_last", [True])
@pytest.mark.parametrize("shuffle", [True])
@pytest.mark.parametrize("prefetch_factor", [2])
@pytest.mark.parametrize("persistent_workers", [True])
def test_single_layer_fine_tuning(
    learner_config,
    model_config,
    task_config,
    batch_size,
    num_workers,
    pin_memory,
    drop_last,
    shuffle,
    prefetch_factor,
    persistent_workers,
):
    dataset_config = CIFAR10DatasetConfig(
        dataset_root="datasets/cifar10", download=True
    )

    transform_train = cifar10_train_transforms()

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
        transform_train=transform_train,
        transform_eval=transform_train,
    )

    datamodule = hydra.utils.instantiate(datamodule_config, _recursive_=False)

    datamodule.setup(stage="fit")

    train_eval_agent = TrainingEvaluationAgent(
        learner_config=learner_config,
        model_config=model_config,
        task_config=task_config,
        modality_config=DottedDict(image=True),
        datamodule=datamodule,
    )
