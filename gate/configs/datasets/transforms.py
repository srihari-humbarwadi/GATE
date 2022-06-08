from dataclasses import dataclass
from typing import Any, Optional

from torchvision import transforms
from torchvision.transforms import transforms

from gate.configs import get_module_import_path
from gate.configs.datamodule.base import TransformConfig


def channels_first(x):
    return x.transpose([2, 0, 1])


def omniglot_support_set_transforms():
    return transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize(size=(28, 28)),
            transforms.ToTensor(),
        ]
    )


def omniglot_query_set_transforms():
    return transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize(size=(28, 28)),
            transforms.ToTensor(),
        ]
    )


def cub200_support_set_transforms():
    return transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize(size=(84, 84)),
            transforms.ToTensor(),
        ]
    )


def cub200_query_set_transforms():
    return transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize(size=(84, 84)),
            transforms.ToTensor(),
        ]
    )


def aircraft_support_set_transforms():
    return transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize(size=(84, 84)),
            transforms.ToTensor(),
        ]
    )


def aircraft_query_set_transforms():
    return transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize(size=(84, 84)),
            transforms.ToTensor(),
        ]
    )


def dtd_support_set_transforms():
    return transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize(size=(84, 84)),
            transforms.ToTensor(),
        ]
    )


def dtd_query_set_transforms():
    return transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize(size=(84, 84)),
            transforms.ToTensor(),
        ]
    )


def mscoco_support_set_transforms():
    return transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize(size=(84, 84)),
            transforms.ToTensor(),
        ]
    )


def mscoco_query_set_transforms():
    return transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize(size=(84, 84)),
            transforms.ToTensor(),
        ]
    )


def vgg_flowers_support_set_transforms():
    return transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize(size=(84, 84)),
            transforms.ToTensor(),
        ]
    )


def vgg_flowers_query_set_transforms():
    return transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize(size=(84, 84)),
            transforms.ToTensor(),
        ]
    )


def fungi_support_set_transforms():
    return transforms.Compose(
        [
            transforms.Resize(size=(84, 84)),
            transforms.ToTensor(),
        ]
    )


def fungi_query_set_transforms():
    return transforms.Compose(
        [
            transforms.Resize(size=(84, 84)),
            transforms.ToTensor(),
        ]
    )


def german_traffic_signs_support_set_transforms():
    return transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize(size=(84, 84)),
            transforms.ToTensor(),
        ]
    )


def german_traffic_signs_query_set_transforms():
    return transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize(size=(84, 84)),
            transforms.ToTensor(),
        ]
    )


def quickdraw_support_set_transforms():
    return transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize(size=(28, 28)),
            transforms.ToTensor(),
        ]
    )


def quickdraw_query_set_transforms():
    return transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize(size=(28, 28)),
            transforms.ToTensor(),
        ]
    )


def cifar10_train_transforms():
    return transforms.Compose(
        [
            #
            transforms.ToPILImage(),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]
            ),
        ]
    )


def cifar10_eval_transforms():
    return transforms.Compose(
        [
            #
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]
            ),
        ]
    )


def cifar100_train_transforms():
    return transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5071, 0.4866, 0.4409], std=[0.2009, 0.1984, 0.2023]
            ),
        ]
    )


def cifar100_eval_transforms():
    return transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5071, 0.4866, 0.4409], std=[0.2009, 0.1984, 0.2023]
            ),
        ]
    )


def stl10_train_transforms():
    return transforms.Compose(
        [
            transforms.Pad(4),
            transforms.RandomCrop(96),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )


def stl10_eval_transforms():
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )


def omniglot_transform_config():
    return transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize(size=(28, 28)),
            transforms.ToTensor(),
        ]
    )


@dataclass
class OmniglotSupportSetTransformConfig(TransformConfig):
    _target_: Any = get_module_import_path(omniglot_support_set_transforms)


@dataclass
class OmniglotQuerySetTransformConfig(TransformConfig):
    _target_: Any = get_module_import_path(omniglot_query_set_transforms)


@dataclass
class CUB200SupportSetTransformConfig(TransformConfig):
    _target_: Any = get_module_import_path(cub200_support_set_transforms)


@dataclass
class CUB200QuerySetTransformConfig(TransformConfig):
    _target_: Any = get_module_import_path(cub200_query_set_transforms)


@dataclass
class DTDSupportSetTransformConfig(TransformConfig):
    _target_: Any = get_module_import_path(dtd_support_set_transforms)


@dataclass
class DTDQuerySetTransformConfig(TransformConfig):
    _target_: Any = get_module_import_path(dtd_query_set_transforms)


@dataclass
class GermanTrafficSignsSupportSetTransformConfig(TransformConfig):
    _target_: Any = get_module_import_path(
        german_traffic_signs_support_set_transforms
    )


@dataclass
class GermanTrafficSignsQuerySetTransformConfig(TransformConfig):
    _target_: Any = get_module_import_path(
        german_traffic_signs_query_set_transforms
    )


@dataclass
class AircraftSupportSetTransformConfig(TransformConfig):
    _target_: Any = get_module_import_path(aircraft_support_set_transforms)


@dataclass
class AircraftQuerySetTransformConfig(TransformConfig):
    _target_: Any = get_module_import_path(aircraft_query_set_transforms)


@dataclass
class VGGFlowersSupportSetTransformConfig(TransformConfig):
    _target_: Any = get_module_import_path(vgg_flowers_support_set_transforms)


@dataclass
class VGGFlowersQuerySetTransformConfig(TransformConfig):
    _target_: Any = get_module_import_path(vgg_flowers_query_set_transforms)


@dataclass
class FungiSupportSetTransformConfig(TransformConfig):
    _target_: Any = get_module_import_path(fungi_support_set_transforms)


@dataclass
class FungiQuerySetTransformConfig(TransformConfig):
    _target_: Any = get_module_import_path(fungi_query_set_transforms)


@dataclass
class QuickDrawSupportSetTransformConfig(TransformConfig):
    _target_: Any = get_module_import_path(quickdraw_support_set_transforms)


@dataclass
class QuickDrawQuerySetTransformConfig(TransformConfig):
    _target_: Any = get_module_import_path(quickdraw_query_set_transforms)


@dataclass
class MSCOCOSupportSetTransformConfig(TransformConfig):
    _target_: Any = get_module_import_path(mscoco_support_set_transforms)


@dataclass
class MSCOCOQuerySetTransformConfig(TransformConfig):
    _target_: Any = get_module_import_path(mscoco_query_set_transforms)


@dataclass
class FewShotTransformConfig:
    support_set_input_transform: Optional[Any] = None
    query_set_input_transform: Optional[Any] = None
    support_set_target_transform: Optional[Any] = None
    query_set_target_transform: Optional[Any] = None
