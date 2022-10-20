from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import hydra
from omegaconf import DictConfig, ListConfig
from torchvision.transforms import transforms

from gate.configs import get_module_import_path
from gate.configs.datamodule.base import TransformConfig
from gate.configs.string_variables import (
    ADDITIONAL_INPUT_TRANSFORMS,
    ADDITIONAL_TARGET_TRANSFORMS,
)


def compose_with_additional_transforms(
    transforms_list, additional_transforms: Optional[Any] = None
):
    if additional_transforms is None:
        return transforms.Compose(transforms_list)

    if not isinstance(additional_transforms, (List, ListConfig)):
        additional_transforms = [additional_transforms]

    additional_transforms_functions = []

    for transform in additional_transforms:
        if isinstance(transform, (Dict, DictConfig)):
            additional_transforms_functions.append(hydra.utils.instantiate(transform))
        else:
            additional_transforms_functions.append(transform)

    return (
        transforms.Compose(transforms_list + additional_transforms_functions)
        if len(additional_transforms_functions) > 0
        else transforms.Compose(transforms_list)
    )


def generic_additional_transform(
    additional_transforms: Optional[Any] = None,
):
    if len(additional_transforms) == 0 or additional_transforms is None:
        transforms_list = []
    else:
        transforms_list = []
    return compose_with_additional_transforms(
        transforms_list=transforms_list,
        additional_transforms=additional_transforms,
    )


@dataclass
class InputTransformConfig(TransformConfig):
    _target_: str = get_module_import_path(generic_additional_transform)
    additional_transforms: Optional[List[Any]] = ADDITIONAL_INPUT_TRANSFORMS


@dataclass
class TargetTransformConfig(TransformConfig):
    _target_: str = get_module_import_path(generic_additional_transform)
    additional_transforms: Optional[List[Any]] = ADDITIONAL_TARGET_TRANSFORMS
