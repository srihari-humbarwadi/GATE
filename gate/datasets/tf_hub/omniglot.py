import pathlib

from dotted_dict import DottedDict
from typing import Optional, Union, Any

from gate.base.utils.loggers import get_logger
from gate.datasets.data_utils import (
    FewShotSuperSplitSetOptions,
)
from gate.datasets.tf_hub import bytes_to_string
from gate.datasets.tf_hub.classification_dataset import ClassificationDataset
from gate.datasets.tf_hub.few_shot_base_dataset import (
    FewShotClassificationDataset,
)

log = get_logger(__name__, set_default_handler=True)


# TODO ensure the multi-set few-shot dataset splits things
#  into support/val/query for
#  consistency Make sure the two datasets have matching sets
#  for consistency again
#  Create one few-shot dataset for each of the metadataset datasets
class OmniglotClassificationDataset(ClassificationDataset):
    def __init__(
        self,
        dataset_root: Union[str, pathlib.Path],
        split_name: str,
        download: bool,
        input_transform: Optional[Any] = None,
        target_transform: Optional[Any] = None,
    ):

        super(OmniglotClassificationDataset, self).__init__(
            dataset_name="omniglot",
            input_target_keys=dict(inputs="image", targets="label"),
            dataset_root=dataset_root,
            split_name=split_name,
            download=download,
            input_transform=input_transform,
            target_transform=target_transform,
            input_shape_dict=DottedDict(
                image=DottedDict(channels=1, height=28, width=28)
            ),
            target_shape_dict=DottedDict(image=DottedDict(num_classes=1622)),
        )


class OmniglotFewShotClassificationDataset(FewShotClassificationDataset):
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
            label_extractor_fn=lambda x: bytes_to_string(x),
            min_num_classes_per_set=min_num_classes_per_set,
            min_num_samples_per_class=min_num_samples_per_class,
        )
