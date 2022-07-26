# TODO ensure the multi-set few-shot dataset splits things
#  into support/val/query for
#  consistency Make sure the two datasets have matching sets
#  for consistency again
#  Create one few-shot dataset for each of the metadataset datasets
import pathlib
from typing import Any, Optional, Union

from dotted_dict import DottedDict

from gate.datasets.tf_hub.standard.base import ClassificationDataset

# TODO:
# Tutorial for how to build a model
# Tutorial on how to build a new 'generic dataset'
# Tutorial on how to build a new 'tf hub dataset'
# Tutorial on how to build a new 'learner'


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
