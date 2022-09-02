import pathlib
from typing import Any, Callable, Dict, List, Optional

import h5py
import hydra
import numpy as np
import tensorflow_datasets as tfds
import torch
from dotted_dict import DottedDict
from omegaconf import DictConfig
from torch import Tensor
from torch.utils.data import Dataset
from tqdm import tqdm

from gate.base.utils.loggers import get_logger
from gate.datasets.data_utils import (
    FewShotSuperSplitSetOptions,
    get_class_to_idx_dict,
    get_class_to_image_idx_and_bbox,
    store_dict_as_hdf5,
)

log = get_logger(
    __name__,
)


# convert a list of dicts into a dict of lists
def list_of_dicts_to_dict_of_lists(list_of_dicts):
    return {key: [x[key] for x in list_of_dicts] for key in list_of_dicts[0].keys()}


def apply_input_transforms(inputs, transforms):
    inputs = [transforms(x) for x in inputs]

    if isinstance(inputs[0], (Dict, DictConfig)):

        inputs = list_of_dicts_to_dict_of_lists(inputs)

        inputs = {key: torch.stack(value, dim=0) for key, value in inputs.items()}
    else:
        if isinstance(inputs[0], Tensor):
            inputs = torch.stack(inputs, dim=0)

    return inputs


def apply_target_transforms(targets, transforms):
    targets = transforms(targets)
    return targets


class FewShotClassificationDatasetTFDS(Dataset):
    def __init__(
        self,
        dataset_name: str,
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
        modality_config: Dict,
        input_shape_dict: Dict,
        input_target_annotation_keys: Dict,
        subset_split_name_list: Optional[List[str]] = None,
        split_percentage: Optional[Dict[str, float]] = None,
        split_config: Optional[DictConfig] = None,
        support_set_input_transform: Any = None,
        query_set_input_transform: Any = None,
        support_set_target_transform: Any = None,
        query_set_target_transform: Any = None,
        support_to_query_ratio: float = 0.75,
        rescan_cache: bool = True,
        label_extractor_fn: Optional[Any] = None,
    ):
        super(FewShotClassificationDatasetTFDS, self).__init__()

        self.dataset_name = dataset_name
        self.dataset_root = dataset_root
        self.input_target_annotation_keys = input_target_annotation_keys
        self.input_shape_dict = input_shape_dict
        self.modality_config = modality_config

        self.num_episodes = num_episodes

        assert min_num_samples_per_class < num_samples_per_class, (
            f"min_num_samples_per_class {min_num_samples_per_class} "
            f"must be less than "
            f"num_samples_per_class {num_samples_per_class}"
        )

        assert min_num_classes_per_set < num_classes_per_set, (
            f"min_num_classes_per_set {min_num_classes_per_set} "
            f"must be less than "
            f"num_classes_per_set {num_classes_per_set}"
        )

        self.min_num_classes_per_set = min_num_classes_per_set
        self.min_num_samples_per_class = min_num_samples_per_class
        self.num_classes_per_set = num_classes_per_set
        self.num_samples_per_class = num_samples_per_class
        self.variable_num_samples_per_class = variable_num_samples_per_class
        self.variable_num_classes_per_set = variable_num_classes_per_set
        self.split_config = split_config
        self.print_info = True

        self.support_set_input_transform = (
            hydra.utils.instantiate(support_set_input_transform)
            if isinstance(support_set_input_transform, Dict)
            or isinstance(support_set_input_transform, DictConfig)
            else support_set_input_transform
        )
        self.query_set_input_transform = (
            hydra.utils.instantiate(query_set_input_transform)
            if isinstance(query_set_input_transform, Dict)
            or isinstance(query_set_input_transform, DictConfig)
            else query_set_input_transform
        )

        self.support_set_target_transform = (
            hydra.utils.instantiate(support_set_target_transform)
            if isinstance(support_set_target_transform, Dict)
            or isinstance(support_set_input_transform, DictConfig)
            else support_set_target_transform
        )

        self.query_set_target_transform = (
            hydra.utils.instantiate(query_set_target_transform)
            if isinstance(query_set_target_transform, Dict)
            or isinstance(query_set_target_transform, DictConfig)
            else query_set_target_transform
        )

        self.support_to_query_ratio = support_to_query_ratio

        self.split_name = split_name
        self.split_percentage = split_percentage
        self.subsets = []

        if subset_split_name_list is None:
            subset_split_name_list = ["train", "test"]

        for subset_name in subset_split_name_list:

            subset, subset_info = tfds.load(
                self.dataset_name,
                split=subset_name,
                shuffle_files=False,
                download=download,
                as_supervised=False,
                data_dir=self.dataset_root,
                with_info=True,
            )

            log.info(f"Loading into memory {subset_name} info: {subset_info}")
            subset_samples = []
            with tqdm(total=len(subset)) as pbar:
                for sample in subset:
                    sample = {key: sample[key].numpy() for key in sample.keys()}
                    subset_samples.append(sample)
                    pbar.update(1)
                self.subsets.append(subset_samples)

            if self.print_info:
                log.info(f"Loaded two subsets with info: {subset_info}")

        self.class_to_address_dict = get_class_to_idx_dict(
            self.subsets,
            class_name_key=self.input_target_annotation_keys["target_annotations"],
            label_extractor_fn=label_extractor_fn,
        )

        self.label_extractor_fn = label_extractor_fn

        if self.split_config is None:
            if split_name == FewShotSuperSplitSetOptions.TRAIN:
                self.current_class_to_address_dict = {
                    key: value
                    for idx, (key, value) in enumerate(
                        self.class_to_address_dict.items()
                    )
                    if idx < split_percentage[FewShotSuperSplitSetOptions.TRAIN]
                }
            elif split_name == FewShotSuperSplitSetOptions.VAL:
                self.current_class_to_address_dict = {
                    key: value
                    for idx, (key, value) in enumerate(
                        self.class_to_address_dict.items()
                    )
                    if split_percentage[FewShotSuperSplitSetOptions.TRAIN]
                    < idx
                    < split_percentage[FewShotSuperSplitSetOptions.TRAIN]
                    + split_percentage[FewShotSuperSplitSetOptions.VAL]
                }
            elif split_name == FewShotSuperSplitSetOptions.TEST:
                self.current_class_to_address_dict = {
                    key: value
                    for idx, (key, value) in enumerate(
                        self.class_to_address_dict.items()
                    )
                    if split_percentage[FewShotSuperSplitSetOptions.TRAIN]
                    + split_percentage[FewShotSuperSplitSetOptions.VAL]
                    < idx
                    < split_percentage[FewShotSuperSplitSetOptions.TRAIN]
                    + split_percentage[FewShotSuperSplitSetOptions.VAL]
                    + split_percentage[FewShotSuperSplitSetOptions.TEST]
                }
        else:
            if self.print_info:
                log.info(self.split_config)
            self.current_class_to_address_dict = {
                label_name: self.class_to_address_dict[label_name]
                for label_name in self.split_config[split_name]
            }

        self.print_info = False

    def __len__(self):
        return self.num_episodes

    def __getitem__(self, index):
        rng = np.random.RandomState(index)

        support_set_inputs = []
        support_set_labels = []

        query_set_inputs = []
        query_set_labels = []

        num_classes_per_set = (
            rng.choice(range(self.min_num_classes_per_set, self.num_classes_per_set))
            if self.variable_num_classes_per_set
            else self.num_classes_per_set
        )

        available_class_labels = list(self.current_class_to_address_dict.keys())
        select_classes_for_set = rng.choice(
            available_class_labels,
            size=min(num_classes_per_set, len(available_class_labels)),
        )

        label_idx = set(select_classes_for_set)
        label_idx = list(label_idx)

        # shuffle label idx
        label_idx = rng.permutation(label_idx)

        label_idx_to_local_label_idx = {
            label_name: i for i, label_name in enumerate(label_idx)
        }

        for class_name in select_classes_for_set:
            if self.variable_num_samples_per_class:
                num_samples_per_class = (
                    self.num_samples_per_class
                    if isinstance(self.num_samples_per_class, int)
                    else rng.choice(
                        range(
                            self.min_num_samples_per_class,
                            self.num_samples_per_class,
                        )
                    )
                )
            else:
                num_samples_per_class = self.num_samples_per_class

            selected_samples_addresses_idx = rng.choice(
                range(
                    len(self.current_class_to_address_dict[class_name]),
                ),
                size=min(
                    len(self.current_class_to_address_dict[class_name]),
                    num_samples_per_class,
                ),
                replace=False,
            )

            selected_samples_addresses = [
                self.current_class_to_address_dict[class_name][sample_address_idx]
                for sample_address_idx in selected_samples_addresses_idx
            ]

            data_inputs = [
                self.subsets[subset_idx][idx][
                    self.input_target_annotation_keys["inputs"]
                ]
                for (subset_idx, idx) in selected_samples_addresses
            ]

            data_labels = [
                self.subsets[subset_idx][idx][
                    self.input_target_annotation_keys["target_annotations"]
                ]
                for (subset_idx, idx) in selected_samples_addresses
            ]

            data_labels = [
                label_idx_to_local_label_idx[self.label_extractor_fn(item)]
                for item in data_labels
            ]

            shuffled_idx = rng.permutation(len(data_inputs))

            data_inputs = [data_inputs[i] for i in shuffled_idx]

            if isinstance(data_inputs[0], np.ndarray):
                data_inputs = [
                    torch.tensor(sample).permute(2, 0, 1) for sample in data_inputs
                ]

            data_labels = [data_labels[i] for i in shuffled_idx]

            train_set_range = int(
                np.floor(len(data_inputs) * self.support_to_query_ratio)
            )

            support_set_inputs.extend(data_inputs[:train_set_range])
            support_set_labels.extend(data_labels[:train_set_range])

            query_set_inputs.extend(data_inputs[train_set_range:])
            query_set_labels.extend(data_labels[train_set_range:])

        if self.support_set_input_transform:
            support_set_inputs = apply_input_transforms(
                inputs=support_set_inputs,
                transforms=self.support_set_input_transform,
            )

        if self.support_set_target_transform:
            support_set_labels = apply_target_transforms(
                targets=support_set_labels,
                transforms=self.support_set_target_transform,
            )

        if self.query_set_input_transform:
            query_set_inputs = apply_input_transforms(
                inputs=query_set_inputs,
                transforms=self.query_set_input_transform,
            )

        if self.query_set_target_transform:
            query_set_labels = apply_target_transforms(
                targets=query_set_labels,
                transforms=self.query_set_target_transform,
            )

        support_set_inputs = (
            torch.stack(support_set_inputs, dim=0)
            if isinstance(support_set_inputs, list)
            else support_set_inputs
        )
        support_set_labels = (
            torch.tensor(support_set_labels)
            if isinstance(support_set_labels, list)
            else support_set_labels
        )
        query_set_inputs = (
            torch.stack(query_set_inputs, dim=0)
            if isinstance(query_set_inputs, list)
            else query_set_inputs
        )
        query_set_labels = (
            torch.tensor(query_set_labels)
            if isinstance(query_set_labels, list)
            else query_set_labels
        )

        if not isinstance(support_set_inputs, (Dict, DictConfig)):

            input_dict = DottedDict(
                image=DottedDict(
                    support_set=support_set_inputs,
                    query_set=query_set_inputs,
                ),
            )
        else:

            input_dict = DottedDict(
                image=DottedDict(
                    support_set=support_set_inputs["image"],
                    query_set=query_set_inputs["image"],
                    support_set_extras={
                        key: value
                        for key, value in support_set_inputs.items()
                        if key != "image"
                    },
                    query_set_extras={
                        key: value
                        for key, value in query_set_inputs.items()
                        if key != "image"
                    },
                ),
            )

        if not isinstance(support_set_labels, (Dict, DictConfig)):

            label_dict = DottedDict(
                image=DottedDict(
                    support_set=support_set_labels,
                    query_set=query_set_labels,
                )
            )

        else:

            label_dict = DottedDict(
                image=DottedDict(
                    support_set=support_set_labels["image"],
                    query_set=query_set_labels["image"],
                    support_set_extras={
                        key: value
                        for key, value in support_set_labels.items()
                        if key != "image"
                    },
                    query_set_extras={
                        key: value
                        for key, value in query_set_labels.items()
                        if key != "image"
                    },
                ),
            )
        return input_dict, label_dict


class MSCOCOFewShotClassificationDatasetTFDS(Dataset):
    def __init__(
        self,
        dataset_name: str,
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
        modality_config: Dict,
        input_shape_dict: Dict,
        subset_split_name_list: Optional[List[str]] = None,
        split_percentage: Optional[Dict[str, float]] = None,
        split_config: Optional[DictConfig] = None,
        support_set_input_transform: Any = None,
        query_set_input_transform: Any = None,
        support_set_target_transform: Any = None,
        query_set_target_transform: Any = None,
        support_to_query_ratio: float = 0.75,
        rescan_cache: bool = True,
        label_extractor_fn: Optional[Callable] = None,
    ):
        super(MSCOCOFewShotClassificationDatasetTFDS, self).__init__()

        self.dataset_name = dataset_name
        self.dataset_root = dataset_root
        self.input_shape_dict = input_shape_dict
        self.modality_config = modality_config

        self.num_episodes = num_episodes

        assert min_num_samples_per_class < num_samples_per_class, (
            f"min_num_samples_per_class {min_num_samples_per_class} "
            f"must be less than "
            f"num_samples_per_class {num_samples_per_class}"
        )

        assert min_num_classes_per_set < num_classes_per_set, (
            f"min_num_classes_per_set {min_num_classes_per_set} "
            f"must be less than "
            f"num_classes_per_set {num_classes_per_set}"
        )

        self.min_num_classes_per_set = min_num_classes_per_set
        self.min_num_samples_per_class = min_num_samples_per_class
        self.num_classes_per_set = num_classes_per_set
        self.num_samples_per_class = num_samples_per_class
        self.variable_num_samples_per_class = variable_num_samples_per_class
        self.variable_num_classes_per_set = variable_num_classes_per_set
        self.split_config = split_config
        self.print_info = True

        self.support_set_input_transform = (
            hydra.utils.instantiate(support_set_input_transform)
            if isinstance(support_set_input_transform, Dict)
            or isinstance(support_set_input_transform, DictConfig)
            else support_set_input_transform
        )
        self.query_set_input_transform = (
            hydra.utils.instantiate(query_set_input_transform)
            if isinstance(query_set_input_transform, Dict)
            or isinstance(support_set_input_transform, DictConfig)
            else query_set_input_transform
        )

        self.support_set_target_transform = (
            hydra.utils.instantiate(support_set_target_transform)
            if isinstance(support_set_target_transform, Dict)
            or isinstance(support_set_input_transform, DictConfig)
            else support_set_target_transform
        )

        self.query_set_target_transform = (
            hydra.utils.instantiate(query_set_target_transform)
            if isinstance(query_set_target_transform, Dict)
            or isinstance(support_set_input_transform, DictConfig)
            else query_set_target_transform
        )

        self.support_to_query_ratio = support_to_query_ratio

        self.split_name = split_name
        self.split_percentage = split_percentage
        self.subsets = []

        if subset_split_name_list is None:
            subset_split_name_list = ["train", "test"]

        for subset_name in subset_split_name_list:

            subset, subset_info = tfds.load(
                "coco_captions",
                split=subset_name,
                shuffle_files=False,
                download=download,
                as_supervised=False,
                data_dir=self.dataset_root,
                with_info=True,
            )

            self.subsets.append(list(subset.as_numpy_iterator()))

            if self.print_info:
                log.info(f"Loaded two subsets with info: {subset_info}")

        self.class_to_address_dict = get_class_to_image_idx_and_bbox(
            self.subsets,
            label_extractor_fn=label_extractor_fn,
        )

        self.label_extractor_fn = label_extractor_fn
        # dataset_root = (
        #     pathlib.Path(self.dataset_root)
        #     if isinstance(self.dataset_root, str)
        #     else self.dataset_root
        # )

        # hdf5_filepath = (
        #     dataset_root
        #     / f"{self.dataset_name}_few_shot_classification_dataset.h5"
        # )
        #
        # if hdf5_filepath.exists() and not rescan_cache:
        #     self.class_to_address_dict = h5py.File(hdf5_filepath, "r")
        # else:
        #     self.class_to_address_dict = store_dict_as_hdf5(
        #         self.class_to_address_dict, hdf5_filepath
        #     )

        self.current_class_to_address_dict = self.class_to_address_dict

        if self.split_config is None:
            if split_name == FewShotSuperSplitSetOptions.TRAIN:
                self.current_class_to_address_dict = {
                    key: value
                    for idx, (key, value) in enumerate(
                        self.class_to_address_dict.items()
                    )
                    if idx < split_percentage[FewShotSuperSplitSetOptions.TRAIN]
                }
            elif split_name == FewShotSuperSplitSetOptions.VAL:
                self.current_class_to_address_dict = {
                    key: value
                    for idx, (key, value) in enumerate(
                        self.class_to_address_dict.items()
                    )
                    if split_percentage[FewShotSuperSplitSetOptions.TRAIN]
                    < idx
                    < split_percentage[FewShotSuperSplitSetOptions.TRAIN]
                    + split_percentage[FewShotSuperSplitSetOptions.VAL]
                }
            elif split_name == FewShotSuperSplitSetOptions.TEST:
                self.current_class_to_address_dict = {
                    key: value
                    for idx, (key, value) in enumerate(
                        self.class_to_address_dict.items()
                    )
                    if split_percentage[FewShotSuperSplitSetOptions.TRAIN]
                    + split_percentage[FewShotSuperSplitSetOptions.VAL]
                    < idx
                    < split_percentage[FewShotSuperSplitSetOptions.TRAIN]
                    + split_percentage[FewShotSuperSplitSetOptions.VAL]
                    + split_percentage[FewShotSuperSplitSetOptions.TEST]
                }
        else:
            if self.print_info:
                log.info(self.split_config)
            self.current_class_to_address_dict = {
                label_name: self.class_to_address_dict[label_name]
                for idx, label_name in enumerate(self.split_config[split_name])
            }
        self.print_info = False

    def __len__(self):

        return self.num_episodes

    def __getitem__(self, index):
        rng = np.random.RandomState(index)

        support_set_inputs = []
        support_set_labels = []

        query_set_inputs = []
        query_set_labels = []

        # log.info(
        #     f"Check {self.min_num_classes_per_set} {self.num_classes_per_set}"
        # )
        num_classes_per_set = (
            rng.choice(range(self.min_num_classes_per_set, self.num_classes_per_set))
            if self.variable_num_classes_per_set
            else self.num_classes_per_set
        )

        available_class_labels = list(self.current_class_to_address_dict.keys())
        select_classes_for_set = rng.choice(
            available_class_labels,
            size=min(num_classes_per_set, len(available_class_labels)),
        )

        label_idx = set(select_classes_for_set)
        label_idx = list(label_idx)

        # shuffle label idx
        label_idx = rng.permutation(label_idx)

        label_idx_to_local_label_idx = {
            label_name: i for i, label_name in enumerate(label_idx)
        }

        for class_name in select_classes_for_set:
            if self.variable_num_samples_per_class:
                num_samples_per_class = (
                    self.num_samples_per_class
                    if isinstance(self.num_samples_per_class, int)
                    else rng.choice(
                        range(
                            self.min_num_samples_per_class,
                            self.num_samples_per_class,
                        )
                    )
                )
            else:
                num_samples_per_class = self.num_samples_per_class

            selected_samples_addresses_idx = rng.choice(
                range(
                    len(self.current_class_to_address_dict[class_name]),
                ),
                size=min(
                    len(self.current_class_to_address_dict[class_name]),
                    num_samples_per_class,
                ),
                replace=False,
            )

            selected_samples_addresses = [
                self.current_class_to_address_dict[class_name][sample_address_idx]
                for sample_address_idx in selected_samples_addresses_idx
            ]

            # log.info("HERE HERE")

            data_inputs = [
                self.subsets[object_dict["subset_idx"]][object_dict["sample_idx"]][
                    "image"
                ][
                    object_dict["bbox"]["x_min"] : object_dict["bbox"]["x_max"],
                    object_dict["bbox"]["y_min"] : object_dict["bbox"]["y_max"],
                ]
                for object_dict in selected_samples_addresses
            ]

            data_labels = [
                object_dict["label"] for object_dict in selected_samples_addresses
            ]

            # log.info("HERE HERE HERE")

            # log.info(
            #     f"label idx to local label idx:
            #  {label_idx_to_local_label_idx}, data labels {data_labels}"
            # )

            # log.info(data_labels)
            #
            # log.info(label_idx_to_local_label_idx)
            #
            # log.info(self.current_class_to_address_dict)

            # log.info(data_inputs)

            data_labels = [
                label_idx_to_local_label_idx[self.label_extractor_fn(item)]
                for item in data_labels
            ]

            shuffled_idx = rng.permutation(len(data_inputs))

            data_inputs = [data_inputs[i] for i in shuffled_idx]

            if isinstance(data_inputs[0], np.ndarray):
                data_inputs = [
                    torch.tensor(sample).permute(2, 0, 1) for sample in data_inputs
                ]

            data_labels = [data_labels[i] for i in shuffled_idx]

            train_set_range = int(
                np.floor(len(data_inputs) * self.support_to_query_ratio)
            )

            support_set_inputs.extend(data_inputs[:train_set_range])
            support_set_labels.extend(data_labels[:train_set_range])

            query_set_inputs.extend(data_inputs[train_set_range:])
            query_set_labels.extend(data_labels[train_set_range:])

        if self.support_set_input_transform:
            support_set_inputs = torch.stack(
                [
                    self.support_set_input_transform(input)
                    for input in support_set_inputs
                ],
                dim=0,
            )

        if self.support_set_target_transform:
            support_set_labels = torch.stack(
                [
                    self.support_set_target_transform(label)
                    for label in support_set_labels
                ],
                dim=0,
            )

        if self.query_set_input_transform:
            query_set_inputs = torch.stack(
                [self.query_set_input_transform(input) for input in query_set_inputs],
                dim=0,
            )

        if self.query_set_target_transform:
            query_set_labels = torch.stack(
                [self.query_set_target_transform(label) for label in query_set_labels],
                dim=0,
            )

        support_set_inputs = (
            torch.stack(support_set_inputs, dim=0)
            if isinstance(support_set_inputs, list)
            else support_set_inputs
        )
        support_set_labels = (
            torch.tensor(support_set_labels)
            if isinstance(support_set_labels, list)
            else support_set_labels
        )
        query_set_inputs = (
            torch.stack(query_set_inputs, dim=0)
            if isinstance(query_set_inputs, list)
            else query_set_inputs
        )
        query_set_labels = (
            torch.tensor(query_set_labels)
            if isinstance(query_set_labels, list)
            else query_set_labels
        )

        input_dict = DottedDict(
            image=DottedDict(
                support_set=support_set_inputs, query_set=query_set_inputs
            ),
        )

        label_dict = DottedDict(
            image=DottedDict(support_set=support_set_labels, query_set=query_set_labels)
        )

        return input_dict, label_dict
