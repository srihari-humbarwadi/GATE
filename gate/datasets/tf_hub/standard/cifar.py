import pathlib
from typing import Any, Optional, Union

from dotted_dict import DottedDict

from gate.base.utils.loggers import get_logger
from gate.datasets.tf_hub.standard.base import ClassificationDataset

log = get_logger(__name__, set_default_handler=False)


class CIFAR10ClassificationDataset(ClassificationDataset):
    def __init__(
        self,
        dataset_root: Union[str, pathlib.Path],
        split_name: str,
        download: bool,
        input_transform: Optional[Any] = None,
        target_transform: Optional[Any] = None,
    ):
        super(CIFAR10ClassificationDataset, self).__init__(
            dataset_name="cifar10",
            input_target_keys=dict(inputs="image", targets="label"),
            dataset_root=dataset_root,
            split_name=split_name,
            download=download,
            input_transform=input_transform,
            target_transform=target_transform,
            input_shape_dict=DottedDict(
                image=DottedDict(channels=3, height=32, width=32)
            ),
            target_shape_dict=DottedDict(image=DottedDict(num_classes=10)),
        )


class CIFAR100ClassificationDataset(ClassificationDataset):
    def __init__(
        self,
        dataset_root: Union[str, pathlib.Path],
        split_name: str,
        download: bool,
        input_transform: Optional[Any] = None,
        target_transform: Optional[Any] = None,
    ):
        super(CIFAR100ClassificationDataset, self).__init__(
            dataset_name="cifar100",
            input_target_keys=dict(inputs="image", targets="label"),
            dataset_root=dataset_root,
            split_name=split_name,
            download=download,
            input_transform=input_transform,
            target_transform=target_transform,
            input_shape_dict=DottedDict(
                image=DottedDict(channels=3, height=32, width=32)
            ),
            target_shape_dict=DottedDict(image=DottedDict(num_classes=100)),
        )
