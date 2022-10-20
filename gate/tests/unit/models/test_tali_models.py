import pytest
import torch
from dotted_dict import DottedDict

from gate.base.utils.loggers import get_logger
from gate.models.tali import TALIModusPrime

log = get_logger(__name__, set_default_handler=True)


@pytest.mark.parametrize(
    "model_name",
    ["tali-vasi-modality-v-1.0"],
)
@pytest.mark.parametrize("model_version", ["v0", "v1", "v2", "v3"])
@pytest.mark.parametrize("pretrained", [True, False])
@pytest.mark.parametrize(
    "image_shape",
    [
        DottedDict(channels=3, width=32, height=32),
        DottedDict(channels=3, width=224, height=224),
        DottedDict(channels=3, width=288, height=176),
    ],
)
def test_tali_models(
    model_name,
    model_version,
    pretrained,
    image_shape,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = TALIModusPrime(
        input_shape_dict=DottedDict(
            image=DottedDict(
                shape=DottedDict(
                    channels=3,
                    width=288,
                    height=176,
                ),
                dtype=torch.float32,
            ),
            video=DottedDict(
                shape=DottedDict(
                    sequence_length=8,
                    channels=3,
                    width=288,
                    height=176,
                ),
                dtype=torch.float32,
            ),
            audio=DottedDict(
                shape=DottedDict(
                    channels=2,
                    sequence_length=220500,
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
        model_root_dir=".gate/pretrained_models/tali_vita_b_16",
        model_name_to_download=model_name,
        model_version=model_version,
        project_name="machinelearningbrewery/tali-model-repo",
        pretrained=pretrained,
    )
    dummy_x = {
        "image": torch.randn(
            size=[
                2,
                image_shape.channels,
                image_shape.width,
                image_shape.height,
            ]
        ).to(device),
        "video": torch.randn(
            size=[
                2,
                8,
                image_shape.channels,
                image_shape.width,
                image_shape.height,
            ]
        ).to(device),
        "audio": torch.randn(size=[2, 2, 220500]).to(device),
        "text": torch.randint(0, 100, size=[2, 77]).long().to(device),
    }

    log.info(f"dummy_x.shape: {dummy_x['image'].shape} {dummy_x['text'].shape}")
    out = model.forward(dummy_x)
    log.debug(model)

    assert out is not None
