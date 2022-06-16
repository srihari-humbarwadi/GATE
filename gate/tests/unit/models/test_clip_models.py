import pytest
import torch
from dotted_dict import DottedDict

from gate.base.utils.loggers import get_logger
from gate.models.clip import CLIP

log = get_logger(__name__, set_default_handler=True)


@pytest.mark.parametrize(
    "model_name",
    [
        "RN50",
        "RN101",
        "RN50x4",
        "RN50x16",
        "RN50x64",
        "ViT-B/32",
        "ViT-B/16",
        "ViT-L/14",
    ],
)
@pytest.mark.parametrize("pretrained", [True, False])
@pytest.mark.parametrize(
    "image_shape",
    [
        DottedDict(channels=3, width=224 * 2, height=224 * 2),
        DottedDict(channels=3, width=32, height=32),
        DottedDict(channels=3, width=224, height=224),
    ],
)
def test_clip_models(
    model_name,
    pretrained,
    image_shape,
):
    device = "cpu"
    model = CLIP(
        input_shape_dict=DottedDict(
            image=DottedDict(
                shape=DottedDict(
                    channels=image_shape.channels,
                    width=image_shape.channels,
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
    dummy_x = {
        "image": torch.randn(
            size=[
                4,
                image_shape.channels,
                image_shape.height,
                image_shape.width,
            ]
        ).to(device),
        "text": torch.randint(0, 100, size=[4, 77]).long().to(device),
    }

    log.info(f"dummy_x.shape: {dummy_x['image'].shape} {dummy_x['text'].shape}")
    out = model.forward(dummy_x)
    log.debug(model)

    assert out is not None
