import pathlib
from typing import Any, List, Optional, Union

from dotted_dict import DottedDict

from gate.base.utils.loggers import get_logger
from gate.configs.datasets.data_splits_config import data_splits_dict
from gate.datasets.data_utils import FewShotSuperSplitSetOptions
from gate.datasets.tf_hub import bytes_to_string
from gate.datasets.tf_hub.few_shot.base import FewShotClassificationDatasetTFDS

log = get_logger(
    __name__,
)


class GermanTrafficSignsFewShotClassificationDataset(
    FewShotClassificationDatasetTFDS
):
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
        DATASET_NAME = "visual_domain_decathlon/gtsrb"
        split_counts = {
            key: len(value)
            for key, value in data_splits_dict["traffic_signs"].items()
        }
        super(GermanTrafficSignsFewShotClassificationDataset, self).__init__(
            modality_config=DottedDict(image=True),
            input_shape_dict=DottedDict(
                image=dict(channels=3, height=84, width=84)
            ),
            dataset_name=DATASET_NAME,
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
                inputs="image",
                targets="label",
                target_annotations="name",
            ),
            support_set_input_transform=support_set_input_transform,
            query_set_input_transform=query_set_input_transform,
            support_set_target_transform=support_set_target_transform,
            query_set_target_transform=query_set_target_transform,
            split_percentage={
                FewShotSuperSplitSetOptions.TRAIN: split_counts["train"],
                FewShotSuperSplitSetOptions.VAL: split_counts["val"],
                FewShotSuperSplitSetOptions.TEST: split_counts["test"],
            },
            split_config=data_splits_dict["traffic_signs"],
            label_extractor_fn=lambda x: str(
                int(bytes_to_string(x).split("/")[-2])
            ),
            subset_split_name_list=["train", "validation"],
            min_num_classes_per_set=min_num_classes_per_set,
            min_num_samples_per_class=min_num_samples_per_class,
        )
