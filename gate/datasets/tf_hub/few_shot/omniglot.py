import pathlib
from typing import Any, Optional, Union

from dotted_dict import DottedDict

from gate.base.utils.loggers import get_logger
from gate.datasets.data_utils import FewShotSuperSplitSetOptions
from gate.datasets.tf_hub import bytes_to_string
from gate.datasets.tf_hub.few_shot.base import FewShotClassificationDatasetTFDS

log = get_logger(
    __name__,
)


class OmniglotFewShotClassificationDataset(FewShotClassificationDatasetTFDS):
    def __init__(
        self,
        dataset_root: str,
        split_name: str,
        download: bool,
        num_episodes: int,
        min_num_classes_per_set: int,
        min_num_samples_per_class: int,
        num_classes_per_set: int,  # n_way
        num_samples_per_class: int,  # n_shot
        variable_num_samples_per_class: bool,
        variable_num_classes_per_set: bool,
        support_set_input_transform: Any = None,
        query_set_input_transform: Any = None,
        support_set_target_transform: Any = None,
        query_set_target_transform: Any = None,
        support_to_query_ratio: float = 0.75,
        rescan_cache: bool = True,
    ):
        super(OmniglotFewShotClassificationDataset, self).__init__(
            modality_config=DottedDict(image=True),
            input_shape_dict=DottedDict(
                image=dict(channels=3, height=105, width=105)
            ),
            dataset_name="omniglot",
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
                target_annotations="label",
            ),
            support_set_input_transform=support_set_input_transform,
            query_set_input_transform=query_set_input_transform,
            support_set_target_transform=support_set_target_transform,
            query_set_target_transform=query_set_target_transform,
            split_percentage={
                FewShotSuperSplitSetOptions.TRAIN: 1200,
                FewShotSuperSplitSetOptions.VAL: 200,
                FewShotSuperSplitSetOptions.TEST: 222,
            },
            label_extractor_fn=bytes_to_string,
            min_num_classes_per_set=min_num_classes_per_set,
            min_num_samples_per_class=min_num_samples_per_class,
        )
