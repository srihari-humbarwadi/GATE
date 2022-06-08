from dataclasses import MISSING, dataclass
from typing import Dict

from gate.configs import get_module_import_path
from gate.models.tali import TALIModusPrime

input_size_dict_tali = dict(
    image=dict(shape=dict(channels=3, width=288, height=176), dtype="float32"),
    video=dict(
        shape=dict(sequence_length=8, channels=3, width=288, height=176),
        dtype="float32",
    ),
    audio=dict(
        shape=dict(channels=2, sequence_length=22050),
        dtype="float32",
    ),
    text=dict(
        shape=dict(sequence_length=77),
        dtype="int32",
    ),
)


@dataclass
class TALIModelConfig:
    pretrained: bool = MISSING
    input_shape_dict: Dict = MISSING
    model_name_to_download: str = "base"
    model_root_dir: str = "${root_experiment_dir}/tali_models"
    _target_: str = get_module_import_path(TALIModusPrime)


TALIModelGenericPretrainedConfig = TALIModelConfig(
    input_shape_dict=input_size_dict_tali, pretrained=True
)

TALIModelGenericScratchConfig = TALIModelConfig(
    input_shape_dict=input_size_dict_tali, pretrained=False
)
