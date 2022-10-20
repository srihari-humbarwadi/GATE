import pytest
import torch
import torch.nn.functional as F
from dotted_dict import DottedDict
from omegaconf import DictConfig

from gate.base.utils.loggers import get_logger
from gate.configs.task.image_classification import ImageClassificationTaskConfig
from gate.learners.single_layer_fine_tuning import LinearLayerFineTuningScheme
from gate.models.clip import CLIP

log = get_logger(
    __name__,
)


@pytest.mark.parametrize(
    "learner",
    [
        LinearLayerFineTuningScheme,
    ],
)
@pytest.mark.parametrize(
    "fine_tune_all_layers",
    [
        True,
    ],
)
@pytest.mark.parametrize(
    "max_epochs",
    [5],
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
@pytest.mark.parametrize(
    "model_name",
    [
        "RN50",
        "ViT-B/16",
    ],
)
@pytest.mark.parametrize("pretrained", [True])
@pytest.mark.parametrize(
    "image_shape",
    [
        DottedDict(channels=3, width=224, height=224),
    ],
)
@pytest.mark.parametrize(
    "device",
    ["cpu"],
)
def test_single_layer_fine_tuning(
    learner,
    fine_tune_all_layers,
    max_epochs,
    min_learning_rate,
    lr,
    betas,
    eps,
    weight_decay,
    amsgrad,
    model_name,
    pretrained,
    image_shape,
    device,
):
    task_config = ImageClassificationTaskConfig(
        output_shape_dict=DictConfig({"image": dict(num_classes=10)}),
    )

    module = learner(
        optimizer_config=DottedDict(_target_="torch.optim.Adam", lr=0.001),
        lr_scheduler_config=DottedDict(
            _target_="torch.optim.lr_scheduler.CosineAnnealingLR",
            T_max=max_epochs,
            eta_min=0,
            verbose=True,
            batch_size=2,
            num_train_samples=100,
        ),
        fine_tune_all_layers=fine_tune_all_layers,
        max_epochs=max_epochs,
        min_learning_rate=min_learning_rate,
        lr=lr,
        betas=betas,
        eps=eps,
        weight_decay=weight_decay,
        amsgrad=amsgrad,
    )
    model = CLIP(
        input_shape_dict=DottedDict(
            image=DottedDict(
                shape=DottedDict(
                    channels=image_shape.channels,
                    width=image_shape.width,
                    height=image_shape.height,
                ),
                dtype=torch.float32,
            ),
            text=DottedDict(
                shape=DottedDict(
                    sequence_length=77,
                ),
                dtype=torch.int32,
            ),
        ),
        model_root_dir="./pretrained_models/",
        model_name_to_download=model_name,
        pretrained=pretrained,
        device=device,
    )
    model.to(device)
    dummy_x = {
        "image": torch.randn(
            size=[
                2,
                image_shape.channels,
                image_shape.height,
                image_shape.width,
            ]
        ).to(device),
        "text": torch.randint(0, 100, size=[2, 77]).long().to(device),
    }

    log.info(f"dummy_x.shape: {dummy_x['image'].shape} {dummy_x['text'].shape}")
    _ = model.forward(dummy_x)

    module.build(
        model=model,
        input_shape_dict=model.input_shape_dict,
        output_shape_dict=task_config.output_shape_dict,
        task_config=task_config,
        modality_config=DottedDict(image=True),
    )
    optimizer = module.configure_optimizers()["optimizer"]
    module = module.to(device)
    out = module.forward(dummy_x)

    loss = F.cross_entropy(out["image"], torch.randint(0, 10, (2,)).to(device))

    loss.backward()

    optimizer.step()
