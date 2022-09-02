import hydra.utils
import pytest
import torch
from dotted_dict import DottedDict
from omegaconf import DictConfig

from gate.base.utils.loggers import get_logger
from gate.configs.learner import (
    CosineAnnealingLRConfig,
    EpisodicFullModelFineTuningSchemeConfig,
    EpisodicPrototypicalNetworkConfig,
)
from gate.configs.learner.learning_rate_scheduler_config import (
    BiLevelLRSchedulerConfig,
)
from gate.configs.task.image_classification import ImageClassificationTaskConfig
from gate.models.timm_hub import TimmImageModel

log = get_logger(__name__, set_default_handler=True)


@pytest.mark.parametrize(
    "device",
    [torch.cuda.current_device() if torch.cuda.is_available() else "cpu"],
)
@pytest.mark.parametrize(
    "learner",
    [
        EpisodicFullModelFineTuningSchemeConfig,
    ],
)
@pytest.mark.parametrize(
    "fine_tune_all_layers",
    [
        True,
        False,
    ],
)
@pytest.mark.parametrize(
    "max_epochs",
    [100],
)
@pytest.mark.parametrize(
    "min_learning_rate",
    [0.00001],
)
@pytest.mark.parametrize(
    "lr",
    [0.01],
)
@pytest.mark.parametrize(
    "betas",
    [[0.9, 0.999]],
)
@pytest.mark.parametrize(
    "eps",
    [0.000001],
)
@pytest.mark.parametrize(
    "weight_decay",
    [0.00001],
)
@pytest.mark.parametrize(
    "amsgrad",
    [
        False,
    ],
)
def test_single_layer_fine_tuning(
    device,
    learner,
    fine_tune_all_layers,
    max_epochs,
    min_learning_rate,
    lr,
    betas,
    eps,
    weight_decay,
    amsgrad,
):
    task_config = ImageClassificationTaskConfig(
        output_shape_dict=DictConfig({"image": dict(num_classes=10)}),
    )

    module = hydra.utils.instantiate(
        learner,
        lr_scheduler_config=BiLevelLRSchedulerConfig(
            inner_loop_lr_scheduler_config=(
                CosineAnnealingLRConfig(T_max=5, batch_size=32)
            ),
            outer_loop_lr_scheduler_config=(
                CosineAnnealingLRConfig(T_max=5, batch_size=32)
            ),
        ),
        _recursive_=False,
    )

    model = TimmImageModel(
        input_shape_dict=DottedDict(
            image=DottedDict(
                shape=DottedDict(
                    channels=3,
                    width=224,
                    height=224,
                ),
                dtype=torch.float32,
            ),
        ),
        model_name_to_download="resnet18",
        pretrained=True,
    )

    dummy_x = {
        "image": torch.randn(
            size=[
                2,
                model.input_shape_dict.image.shape.channels,
                model.input_shape_dict.image.shape.height,
                model.input_shape_dict.image.shape.width,
            ]
        )
    }

    log.info(f"dummy_x.shape: {dummy_x['image'].shape}")

    _ = model.forward(dummy_x)

    module.build(
        model=model,
        input_shape_dict=model.input_shape_dict,
        output_shape_dict=task_config.output_shape_dict,
        task_config=task_config,
        modality_config=DottedDict(image=True),
    )
    optimizer = module.configure_optimizers()["optimizer"]

    dummy_input_set = DottedDict(
        image=DottedDict(
            support_set=torch.randn(
                size=[
                    1,
                    10,
                    model.input_shape_dict.image.shape.channels,
                    model.input_shape_dict.image.shape.height,
                    model.input_shape_dict.image.shape.width,
                ],
                requires_grad=False,
                device=device,
            ),
            query_set=torch.randn(
                size=[
                    1,
                    10,
                    model.input_shape_dict.image.shape.channels,
                    model.input_shape_dict.image.shape.height,
                    model.input_shape_dict.image.shape.width,
                ],
                requires_grad=False,
                device=device,
            ),
        )
    )

    dummy_label_set = DottedDict(
        image=DottedDict(
            support_set=torch.randint(
                high=5,
                size=(
                    1,
                    10,
                ),
                requires_grad=False,
                dtype=torch.int64,
                device=device,
            ),
            query_set=torch.randint(
                high=5,
                size=(
                    1,
                    10,
                ),
                requires_grad=False,
                dtype=torch.int64,
                device=device,
            ),
        )
    )

    sample = (dummy_input_set, dummy_label_set)

    module.to(device)

    output_dict, computed_task_metrics_dict, loss = module.step(
        sample,
        batch_idx=0,
        learner_metrics_dict=module.learner_metrics_dict,
        task_metrics_dict={},
        phase_name="train",
    )

    log.info(
        f"output_dict: {output_dict}, "
        f"computed_task_metrics_dict: {computed_task_metrics_dict}, "
        f"loss: {loss}"
    )
