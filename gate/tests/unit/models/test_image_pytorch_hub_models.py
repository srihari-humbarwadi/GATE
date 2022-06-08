import pytest
import torch
from dotted_dict import DottedDict

from gate.base.utils.loggers import get_logger
from gate.configs.datamodule.base import ShapeConfig
from gate.models.timm_hub import TimmImageModel

log = get_logger(__name__, set_default_handler=True)


@pytest.mark.parametrize(
    "model_name",
    [
        "resnet18",
        "resnet34",
        "resnet50",
        "resnet101",
        "resnet152",
    ],
)
@pytest.mark.parametrize("pretrained", [True, False])
@pytest.mark.parametrize(
    "image_shape",
    [
        DottedDict(channels=3, width=32, height=32),
        DottedDict(channels=3, width=224, height=224),
    ],
)
def test_pytorch_hub_models(
    model_name,
    pretrained,
    image_shape,
):
    model = TimmImageModel(
        input_shape_dict=ShapeConfig(
            image=DottedDict(
                shape=DottedDict(
                    channels=image_shape.channels,
                    width=image_shape.width,
                    height=image_shape.height,
                ),
                dtype=torch.float32,
            ),
        ),
        model_name_to_download=model_name,
        pretrained=pretrained,
    )
    dummy_x = {
        "image": torch.randn(
            size=[
                2,
                image_shape.channels,
                image_shape.height,
                image_shape.width,
            ]
        )
    }

    log.info(f"dummy_x.shape: {dummy_x['image'].shape}")

    out = model.forward(dummy_x)
    log.debug(model)

    assert out is not None
