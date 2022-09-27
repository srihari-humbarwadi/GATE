import multiprocessing
import pathlib
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from typing import Any, Callable, Dict, Optional

import h5py
import hydra
import torch
from PIL.Image import Image
from omegaconf import DictConfig
from torchvision.transforms import transforms
from tqdm import tqdm

from gate.base.utils.loggers import get_logger
from gate.datasets.data_utils import get_class_to_idx_dict, store_dict_as_hdf5
from gate.datasets.tf_hub.few_shot.base import FewShotClassificationDatasetTFDS

logger = get_logger(__name__)


def data_load(item):
    image, label = item

    return transforms.ToTensor()(image).int8, label


class FewShotClassificationDatsetL2L(FewShotClassificationDatasetTFDS):
    def __init__(
        self,
        dataset_name: str,
        dataset_root: str,
        split_name: str,
        download: bool,
        num_episodes: int,
        min_num_classes_per_set: int,
        min_num_samples_per_class: int,
        min_num_queries_per_class: int,
        num_classes_per_set: int,  # n_way
        num_samples_per_class: int,  # n_shot
        num_queries_per_class: int,
        variable_num_samples_per_class: bool,
        variable_num_queries_per_class: bool,
        variable_num_classes_per_set: bool,
        modality_config: Dict,
        input_shape_dict: Dict,
        input_target_annotation_keys: Dict,
        dataset_module_path: str,
        support_set_input_transform: Any = None,
        query_set_input_transform: Any = None,
        support_set_target_transform: Any = None,
        query_set_target_transform: Any = None,
        support_to_query_ratio: float = 0.75,
        rescan_cache: bool = True,
        label_extractor_fn: Optional[Callable] = None,
    ):
        super(FewShotClassificationDatasetTFDS, self).__init__()

        self.dataset_name = dataset_name
        self.dataset_root = dataset_root
        self.input_target_annotation_keys = input_target_annotation_keys
        self.input_shape_dict = input_shape_dict
        self.modality_config = modality_config

        self.num_episodes = num_episodes

        if variable_num_samples_per_class:
            assert min_num_samples_per_class < num_samples_per_class, (
                f"min_num_samples_per_class {min_num_samples_per_class} "
                f"must be less than "
                f"num_samples_per_class {num_samples_per_class}"
            )

        if variable_num_classes_per_set:
            assert min_num_classes_per_set < num_classes_per_set, (
                f"min_num_classes_per_set {min_num_classes_per_set} "
                f"must be less than "
                f"num_classes_per_set {num_classes_per_set}"
            )

        self.min_num_classes_per_set = min_num_classes_per_set
        self.min_num_samples_per_class = min_num_samples_per_class
        self.min_num_queries_per_class = min_num_queries_per_class
        self.num_classes_per_set = num_classes_per_set
        self.num_samples_per_class = num_samples_per_class
        self.num_queries_per_class = num_queries_per_class
        self.variable_num_samples_per_class = variable_num_samples_per_class
        self.variable_num_queries_per_class = variable_num_queries_per_class
        self.variable_num_classes_per_set = variable_num_classes_per_set
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

        dataset_config = {"_target_": dataset_module_path}
        split_names_mapper = {"train": "train", "val": "validation", "test": "test"}

        dataset = hydra.utils.instantiate(
            config=dataset_config,
            root=self.dataset_root,
            mode=split_names_mapper[self.split_name],
            transform=None,
            target_transform=None,
            download=download,
        )

        dataset_new = []

        logger.info(
            f"Loading the {split_name} set of the {dataset_name} dataset into memory 💿"
        )
        with tqdm(total=len(dataset)) as pbar:
            with ThreadPoolExecutor(
                max_workers=multiprocessing.cpu_count()
            ) as executor:
                for i, (image, label) in enumerate(executor.map(data_load, dataset)):
                    dataset_new.append((image, label))
                    pbar.update(1)

        self.subsets = [dataset_new]

        # self.subsets = [dataset]

        self.class_to_address_dict = get_class_to_idx_dict(
            self.subsets,
            class_name_key=1,
            label_extractor_fn=None,
        )

        self.label_extractor_fn = label_extractor_fn
        dataset_root = (
            pathlib.Path(self.dataset_root)
            if isinstance(self.dataset_root, str)
            else self.dataset_root
        )

        hdf5_filepath = (
            dataset_root / f"{split_name}_{self.dataset_name}_"
            f"few_shot_classification_dataset.h5"
        )

        if hdf5_filepath.exists() and not rescan_cache:
            self.class_to_address_dict = h5py.File(hdf5_filepath, "r")
        else:
            self.class_to_address_dict = store_dict_as_hdf5(
                self.class_to_address_dict, hdf5_filepath
            )
        self.current_class_to_address_dict = self.class_to_address_dict
        self.print_info = False
