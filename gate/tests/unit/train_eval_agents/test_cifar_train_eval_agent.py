import pytest

import torch
import torch.nn.functional as F
from dotted_dict import DottedDict

from gate.base.utils.loggers import get_logger
from gate.class_configs.base import (
    ModalitiesSupportedConfig,
    ImageClassificationTaskModuleConfig,
    ShapeConfig,
    AudioImageResNetConfig,
    ImageResNetConfig,
    LinearLayerFineTuningSchemeConfig,
    CIFAR100DatasetConfig,
    DataLoaderConfig,
)
from gate.datamodules.cifar import CIFAR100DataModule
from gate.learners.single_layer_fine_tuning import LinearLayerFineTuningScheme
from gate.models.resnet import ImageResNet
from gate.train_eval_agents.base import TrainingEvaluationAgent

log = get_logger(__name__, set_default_handler=True)


@pytest.mark.parametrize(
    "model_config",
    [
        ImageResNetConfig(
            model_name_to_download="resnet18",
            pretrained=True,
            input_shape_dict=ShapeConfig(
                image=DottedDict(
                    shape=DottedDict(channels=3, width=224, height=224),
                    dtype=torch.float32,
                ),
            ),
        )
    ],
)
@pytest.mark.parametrize(
    "task_config",
    [
        ImageClassificationTaskModuleConfig(
            output_shape_dict=ShapeConfig(
                image=DottedDict(
                    num_classes=10,
                )
            )
        ),
    ],
)
@pytest.mark.parametrize(
    "learner_config",
    [
        LinearLayerFineTuningSchemeConfig(
            optimizer_config=dict(_target_="torch.optim.Adam", lr=0.001),
            lr_scheduler_config=dict(
                _target_="torch.optim.lr_scheduler.CosineAnnealingLR",
                T_max=10,
                eta_min=0,
                verbose=True,
            ),
        ),
    ],
)
def test_single_layer_fine_tuning(
    learner_config,
    model_config,
    task_config,
):
    dataset_config = CIFAR100DatasetConfig()

    data_loader_config = DataLoaderConfig(
        train_batch_size=2,
        val_batch_size=2,
        test_batch_size=2,
        pin_memory=False,
        train_drop_last=False,
        eval_drop_last=False,
        train_shuffle=False,
        eval_shuffle=False,
        prefetch_factor=2,
        persistent_workers=False,
        num_workers=2,
    )

    datamodule = CIFAR100DataModule(
        dataset_config=dataset_config, data_loader_config=data_loader_config
    )

    datamodule.setup(stage="fit")

    train_eval_agent = TrainingEvaluationAgent(
        learner_config=learner_config,
        model_config=model_config,
        task_config=task_config,
        modality_config=ModalitiesSupportedConfig(image=True),
        datamodule=datamodule,
    )

    # dummy_x = {
    #     "image": torch.randn(size=(2,) + model.input_shape_dict.image),
    # }
    #
    # log.info(f"dummy_x.shape: {dummy_x['image'].shape}")
