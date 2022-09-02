from dataclasses import MISSING, dataclass
from typing import Any, Dict

#
from omegaconf import DictConfig

from gate.configs import get_module_import_path
from gate.tasks.standard_classification import ImageClassificationTaskModule

# ------------------------------------------------------------------------------
# task configs


@dataclass
class TaskConfig:
    output_shape_dict: Dict = MISSING
    _target_: str = MISSING


@dataclass
class ImageClassificationTaskConfig(TaskConfig):
    output_shape_dict: DictConfig = MISSING
    _target_: Any = get_module_import_path(ImageClassificationTaskModule)


TenClassClassificationTask = ImageClassificationTaskConfig(
    output_shape_dict=DictConfig(dict(image=dict(num_classes=10)))
)

HundredClassClassificationTask = ImageClassificationTaskConfig(
    output_shape_dict=DictConfig(dict(image=dict(num_classes=100)))
)

ThousandClassClassificationTask = ImageClassificationTaskConfig(
    output_shape_dict=DictConfig(dict(image=dict(num_classes=1000)))
)

VariableClassClassificationTask = ImageClassificationTaskConfig(
    output_shape_dict=DictConfig(dict(image=dict(num_classes=-1)))
)
