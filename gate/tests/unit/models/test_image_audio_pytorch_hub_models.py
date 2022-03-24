import pytest
import torch

from gate.base.utils.loggers import get_logger
from gate.class_configs.base import (
    ShapeConfig,
)
from gate.models.resnet import ImageResNet

log = get_logger(__name__, set_default_handler=True)


@pytest.mark.parametrize(
    "model_name",
    [
        "resnet18",
        "resnet34",
        "resnet50",
        "resnet101",
        "resnet152",
        "vit_b_16",
        "vit_b_32",
        "vit_l_16",
        "vit_l_32",
    ],
)
@pytest.mark.parametrize("pretrained", [True, False])
@pytest.mark.parametrize("image_shape", [[3, 32, 32], [3, 224, 224]])
def test_pytorch_hub_models(
    model_name,
    pretrained,
    image_shape,
):
    model = ImageResNet(
        input_modality_shape_config=ShapeConfig(image=tuple(image_shape)),
        model_name_to_download=model_name,
        pretrained=pretrained,
    )
    dummy_x = {
        "image": torch.randn(size=(2,) + model.input_modality_shape_config.image),
    }

    log.info(f"dummy_x.shape: {dummy_x['image'].shape}")

    out = model.forward(dummy_x)
    log.debug(model)

    assert out is not None
