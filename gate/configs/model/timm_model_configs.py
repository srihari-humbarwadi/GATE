from dataclasses import dataclass, MISSING

from typing import Dict

from gate.configs import get_module_import_path
from gate.models.timm_hub import TimmImageModel

input_size_dict_224_image = dict(
    image=dict(shape=dict(channels=3, width=224, height=224), dtype="float32")
)

input_size_dict_224_audio_image = dict(
    image=dict(shape=dict(channels=3, width=224, height=224), dtype="float32"),
    audio=dict(shape=dict(channels=2, width=44100), dtype="float32"),
)


@dataclass
class TimmImageModelConfig:
    pretrained: bool = MISSING
    model_name_to_download: str = MISSING
    input_shape_dict: Dict = MISSING
    _target_: str = get_module_import_path(TimmImageModel)


TimmImageResNet18Config = TimmImageModelConfig(
    model_name_to_download="resnet18",
    pretrained=True,
    input_shape_dict=input_size_dict_224_image,
)
