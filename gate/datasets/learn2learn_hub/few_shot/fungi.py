import pathlib
from typing import Any, List, Optional, Union

from dotted_dict import DottedDict
from learn2learn.vision.datasets import FGVCFungi
from omegaconf import DictConfig

from gate.base.utils.loggers import get_logger
from gate.configs import get_module_import_path
from gate.configs.datasets.data_splits_config import data_splits_dict
from gate.datasets.data_utils import FewShotSuperSplitSetOptions
from gate.datasets.learn2learn_hub.few_shot.base import (
    FewShotClassificationDatsetL2L,
)
from gate.datasets.tf_hub import bytes_to_string

log = get_logger(
    __name__,
)


class FungiFewShotClassificationDataset(FewShotClassificationDatsetL2L):
    def __init__(
        self,
        dataset_root: Union[str, pathlib.Path],
        split_name: str,
        download: bool,
        num_episodes: int,
        min_num_classes_per_set: int,
        min_num_samples_per_class: int,
        num_classes_per_set: Union[int, List[int]],  # n_way
        num_samples_per_class: Union[int, List[int]],  # n_shot
        variable_num_samples_per_class: bool,
        variable_num_classes_per_set: bool,
        support_set_input_transform: Optional[Any],
        query_set_input_transform: Optional[Any],
        support_set_target_transform: Optional[Any] = None,
        query_set_target_transform: Optional[Any] = None,
        support_to_query_ratio: float = 0.75,
        rescan_cache: bool = True,
    ):
        dataset_module_path = get_module_import_path(FGVCFungi)
        super(FungiFewShotClassificationDataset, self).__init__(
            modality_config=DottedDict(image=True),
            input_shape_dict=DottedDict(
                image=dict(channels=3, height=84, width=84)
            ),
            dataset_name=dataset_module_path.split(".")[-1],
            dataset_root=dataset_root,
            split_name=split_name,
            download=download,
            num_episodes=num_episodes,
            num_classes_per_set=num_classes_per_set,
            num_samples_per_class=num_samples_per_class,
            variable_num_samples_per_class=variable_num_samples_per_class,
            variable_num_classes_per_set=variable_num_classes_per_set,
            support_to_query_ratio=support_to_query_ratio,
            rescan_cache=rescan_cache,
            input_target_annotation_keys=dict(
                inputs=0,
                targets=1,
                target_annotations=1,
            ),
            support_set_input_transform=support_set_input_transform,
            query_set_input_transform=query_set_input_transform,
            support_set_target_transform=support_set_target_transform,
            query_set_target_transform=query_set_target_transform,
            label_extractor_fn=lambda x: bytes_to_string(x),
            min_num_classes_per_set=min_num_classes_per_set,
            min_num_samples_per_class=min_num_samples_per_class,
            dataset_module_path=dataset_module_path,
        )
