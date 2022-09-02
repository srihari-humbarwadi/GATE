from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import hydra.utils
from omegaconf import DictConfig, ListConfig
from torchvision.transforms import transforms

from gate.configs import get_module_import_path
from gate.configs.datamodule.transform_primitives import (
    InputTransformConfig,
    TargetTransformConfig,
    compose_with_additional_transforms,
)
from gate.datasets.transforms import RandomCropResizeCustom, SuperClassExistingLabels


def omniglot_support_set_transforms(
    additional_transforms: Optional[Any] = None,
):
    return compose_with_additional_transforms(
        [
            transforms.ToPILImage(),
            transforms.Resize(size=(28, 28)),
            transforms.ToTensor(),
        ],
        additional_transforms=additional_transforms,
    )


def omniglot_query_set_transforms(
    additional_transforms: Optional[Any] = None,
):
    return compose_with_additional_transforms(
        [
            transforms.ToPILImage(),
            transforms.Resize(size=(28, 28)),
            transforms.ToTensor(),
        ],
        additional_transforms=additional_transforms,
    )


def cub200_support_set_transforms(
    additional_transforms: Optional[Any] = None,
):
    return compose_with_additional_transforms(
        [
            transforms.ToPILImage(),
            transforms.Resize(size=(84, 84)),
            transforms.ToTensor(),
        ],
        additional_transforms=additional_transforms,
    )


def cub200_query_set_transforms(
    additional_transforms: Optional[Any] = None,
):
    return compose_with_additional_transforms(
        [
            transforms.ToPILImage(),
            transforms.Resize(size=(84, 84)),
            transforms.ToTensor(),
        ],
        additional_transforms=additional_transforms,
    )


def aircraft_support_set_transforms(
    additional_transforms: Optional[Any] = None,
):
    return compose_with_additional_transforms(
        [
            transforms.ToPILImage(),
            transforms.Resize(size=(84, 84)),
            transforms.ToTensor(),
        ],
        additional_transforms=additional_transforms,
    )


def aircraft_query_set_transforms(
    additional_transforms: Optional[Any] = None,
):
    return compose_with_additional_transforms(
        [
            transforms.ToPILImage(),
            transforms.Resize(size=(84, 84)),
            transforms.ToTensor(),
        ],
        additional_transforms=additional_transforms,
    )


def dtd_support_set_transforms(
    additional_transforms: Optional[Any] = None,
):
    return compose_with_additional_transforms(
        [
            transforms.ToPILImage(),
            transforms.Resize(size=(84, 84)),
            transforms.ToTensor(),
        ],
        additional_transforms=additional_transforms,
    )


def dtd_query_set_transforms(additional_transforms: Optional[Any] = None):
    return compose_with_additional_transforms(
        [
            transforms.ToPILImage(),
            transforms.Resize(size=(84, 84)),
            transforms.ToTensor(),
        ],
        additional_transforms=additional_transforms,
    )


def mscoco_support_set_transforms(
    additional_transforms: Optional[Any] = None,
):
    return compose_with_additional_transforms(
        [
            transforms.ToPILImage(),
            transforms.Resize(size=(84, 84)),
            transforms.ToTensor(),
        ],
        additional_transforms=additional_transforms,
    )


def mscoco_query_set_transforms(
    additional_transforms: Optional[Any] = None,
):
    return compose_with_additional_transforms(
        [
            transforms.ToPILImage(),
            transforms.Resize(size=(84, 84)),
            transforms.ToTensor(),
        ],
        additional_transforms=additional_transforms,
    )


def vgg_flowers_support_set_transforms(
    additional_transforms: Optional[Any] = None,
):
    return compose_with_additional_transforms(
        [
            transforms.ToPILImage(),
            transforms.Resize(size=(84, 84)),
            transforms.ToTensor(),
        ],
        additional_transforms=additional_transforms,
    )


def vgg_flowers_query_set_transforms(
    additional_transforms: Optional[Any] = None,
):
    return compose_with_additional_transforms(
        [
            transforms.ToPILImage(),
            transforms.Resize(size=(84, 84)),
            transforms.ToTensor(),
        ],
        additional_transforms=additional_transforms,
    )


def fungi_support_set_transforms(
    additional_transforms: Optional[Any] = None,
):
    return compose_with_additional_transforms(
        [
            transforms.Resize(size=(84, 84)),
            transforms.ToTensor(),
        ],
        additional_transforms=additional_transforms,
    )


def fungi_query_set_transforms(
    additional_transforms: Optional[Any] = None,
):
    return compose_with_additional_transforms(
        [
            transforms.Resize(size=(84, 84)),
            transforms.ToTensor(),
        ],
        additional_transforms=additional_transforms,
    )


def german_traffic_signs_support_set_transforms(
    additional_transforms: Optional[Any] = None,
):
    return compose_with_additional_transforms(
        [
            transforms.ToPILImage(),
            transforms.Resize(size=(84, 84)),
            transforms.ToTensor(),
        ],
        additional_transforms=additional_transforms,
    )


def german_traffic_signs_query_set_transforms(
    additional_transforms: Optional[Any] = None,
):
    return compose_with_additional_transforms(
        [
            transforms.ToPILImage(),
            transforms.Resize(size=(84, 84)),
            transforms.ToTensor(),
        ],
        additional_transforms=additional_transforms,
    )


def quickdraw_support_set_transforms(
    additional_transforms: Optional[Any] = None,
):
    return compose_with_additional_transforms(
        [
            transforms.ToPILImage(),
            transforms.Resize(size=(28, 28)),
            transforms.ToTensor(),
        ],
        additional_transforms=additional_transforms,
    )


def quickdraw_query_set_transforms(
    additional_transforms: Optional[Any] = None,
):
    return compose_with_additional_transforms(
        [
            transforms.ToPILImage(),
            transforms.Resize(size=(28, 28)),
            transforms.ToTensor(),
        ],
        additional_transforms=additional_transforms,
    )


def cifar10_train_transforms(additional_transforms: Optional[Any] = None):
    return compose_with_additional_transforms(
        [
            #
            transforms.ToPILImage(),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.4914, 0.4822, 0.4465],
                std=[0.2023, 0.1994, 0.2010],
            ),
        ],
        additional_transforms=additional_transforms,
    )


def cifar10_eval_transforms(additional_transforms: Optional[Any] = None):
    return compose_with_additional_transforms(
        [
            #
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.4914, 0.4822, 0.4465],
                std=[0.2023, 0.1994, 0.2010],
            ),
        ],
        additional_transforms=additional_transforms,
    )


def cifar100_train_transforms(
    additional_transforms: Optional[Any] = None,
):
    return compose_with_additional_transforms(
        [
            transforms.ToPILImage(),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5071, 0.4866, 0.4409], std=[0.2009, 0.1984, 0.2023]
            ),
        ],
        additional_transforms=additional_transforms,
    )


def cifar100_eval_transforms(additional_transforms: Optional[Any] = None):
    return compose_with_additional_transforms(
        [
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5071, 0.4866, 0.4409], std=[0.2009, 0.1984, 0.2023]
            ),
        ],
        additional_transforms=additional_transforms,
    )


def stl10_train_transforms(additional_transforms: Optional[Any] = None):
    return compose_with_additional_transforms(
        [
            transforms.Pad(4),
            transforms.RandomCrop(96),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ],
        additional_transforms=additional_transforms,
    )


def stl10_eval_transforms(additional_transforms: Optional[Any] = None):
    return compose_with_additional_transforms(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ],
        additional_transforms=additional_transforms,
    )


def omniglot_transform_config(additional_transforms=None):

    return compose_with_additional_transforms(
        [
            transforms.ToPILImage(),
            transforms.Resize(size=(28, 28)),
            transforms.ToTensor(),
        ],
        additional_transforms=additional_transforms,
    )


@dataclass
class OmniglotSupportSetTransformConfig(InputTransformConfig):
    _target_: Any = get_module_import_path(omniglot_support_set_transforms)


@dataclass
class OmniglotQuerySetTransformConfig(InputTransformConfig):
    _target_: Any = get_module_import_path(omniglot_query_set_transforms)


@dataclass
class CUB200SupportSetTransformConfig(InputTransformConfig):
    _target_: Any = get_module_import_path(cub200_support_set_transforms)


@dataclass
class CUB200QuerySetTransformConfig(InputTransformConfig):
    _target_: Any = get_module_import_path(cub200_query_set_transforms)


@dataclass
class DTDSupportSetTransformConfig(InputTransformConfig):
    _target_: Any = get_module_import_path(dtd_support_set_transforms)


@dataclass
class DTDQuerySetTransformConfig(InputTransformConfig):
    _target_: Any = get_module_import_path(dtd_query_set_transforms)


@dataclass
class GermanTrafficSignsSupportSetTransformConfig(InputTransformConfig):
    _target_: Any = get_module_import_path(german_traffic_signs_support_set_transforms)


@dataclass
class GermanTrafficSignsQuerySetTransformConfig(InputTransformConfig):
    _target_: Any = get_module_import_path(german_traffic_signs_query_set_transforms)


@dataclass
class AircraftSupportSetTransformConfig(InputTransformConfig):
    _target_: Any = get_module_import_path(aircraft_support_set_transforms)


@dataclass
class AircraftQuerySetTransformConfig(InputTransformConfig):
    _target_: Any = get_module_import_path(aircraft_query_set_transforms)


@dataclass
class VGGFlowersSupportSetTransformConfig(InputTransformConfig):
    _target_: Any = get_module_import_path(vgg_flowers_support_set_transforms)


@dataclass
class VGGFlowersQuerySetTransformConfig(InputTransformConfig):
    _target_: Any = get_module_import_path(vgg_flowers_query_set_transforms)


@dataclass
class FungiSupportSetTransformConfig(InputTransformConfig):
    _target_: Any = get_module_import_path(fungi_support_set_transforms)


@dataclass
class FungiQuerySetTransformConfig(InputTransformConfig):
    _target_: Any = get_module_import_path(fungi_query_set_transforms)


@dataclass
class QuickDrawSupportSetTransformConfig(InputTransformConfig):
    _target_: Any = get_module_import_path(quickdraw_support_set_transforms)


@dataclass
class QuickDrawQuerySetTransformConfig(InputTransformConfig):
    _target_: Any = get_module_import_path(quickdraw_query_set_transforms)


@dataclass
class MSCOCOSupportSetTransformConfig(InputTransformConfig):
    _target_: Any = get_module_import_path(mscoco_support_set_transforms)


@dataclass
class MSCOCOQuerySetTransformConfig(InputTransformConfig):
    _target_: Any = get_module_import_path(mscoco_query_set_transforms)


@dataclass
class RandomCropResizeCustomTransform:
    _target_: Any = get_module_import_path(RandomCropResizeCustom)
    size: Optional[List[int]] = None
    padding: Optional[List[int]] = None
    pad_if_needed: bool = False
    fill: float = 0
    padding_mode: str = "constant"


@dataclass
class SuperClassExistingLabelsTransform:
    _target_: Any = get_module_import_path(SuperClassExistingLabels)
    num_classes_to_group: int = 5


@dataclass
class FewShotTransformConfig:
    support_set_input_transform: Optional[Any] = InputTransformConfig()
    query_set_input_transform: Optional[Any] = InputTransformConfig()
    support_set_target_transform: Optional[Any] = TargetTransformConfig()
    query_set_target_transform: Optional[Any] = TargetTransformConfig()
