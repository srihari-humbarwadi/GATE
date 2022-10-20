import pathlib
from typing import Any, Callable, Dict, Optional, Union

import hydra
import tensorflow_datasets as tfds
import torch
from omegaconf import DictConfig
from torch.utils.data import Dataset

from gate.base.utils.loggers import get_logger

log = get_logger(__name__, set_default_handler=False)


class ClassificationDataset(Dataset):
    def __init__(
        self,
        dataset_name: str,
        dataset_root: Union[str, pathlib.Path],
        input_target_keys: Dict[str, str],
        split_name: str,
        download: bool,
        input_shape_dict: Dict[str, int],
        target_shape_dict: Dict[str, int],
        input_transform: Optional[Any] = None,
        target_transform: Optional[Any] = None,
    ):
        super(ClassificationDataset, self).__init__()
        self.tf_dataset, info = tfds.load(
            dataset_name,
            split=split_name,
            shuffle_files=False,
            download=download,
            as_supervised=False,
            data_dir=dataset_root,
            with_info=True,
        )
        log.info(f"Loaded {split_name} set with info: {info}")
        self.dataset = list(self.tf_dataset.as_numpy_iterator())

        self.input_target_keys = input_target_keys

        self.input_transform = (
            hydra.utils.instantiate(input_transform)
            if isinstance(input_transform, Dict)
            or isinstance(input_transform, DictConfig)
            else input_transform
        )
        self.target_transform = (
            hydra.utils.instantiate(target_transform)
            if isinstance(target_transform, Dict)
            or isinstance(input_transform, DictConfig)
            else target_transform
        )

        self.input_shape_dict = input_shape_dict
        self.target_shape_dict = target_shape_dict

    def __len__(self):
        return len(self.tf_dataset)

    def __getitem__(self, index):
        sample = self.dataset[index]
        x = sample[self.input_target_keys["inputs"]]
        y = sample[self.input_target_keys["targets"]]

        if self.input_transform:
            x = self.input_transform(x)

        if self.target_transform:
            y = self.target_transform(y)
        # log.info(
        #     f"Returning sample: {torch.tensor(x).type(torch.FloatTensor).shape}"
        # )
        return {"image": torch.tensor(x).type(torch.FloatTensor)}, {
            "image": torch.ones(size=(1,)).type(torch.LongTensor) * y
        }
