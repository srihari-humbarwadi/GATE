from typing import Any, Dict, Optional, Union

import hydra.utils
import torch.utils.data
from torch.utils.data import DataLoader

from gate.configs.datamodule.base import DataLoaderConfig
from gate.configs.datasets.standard_classification import (
    CIFAR10DatasetConfig,
    CIFAR100DatasetConfig,
    PreSplitDatasetConfig,
)
from gate.datamodules.base import DataModule


class TwoSplitDataModule(DataModule):
    def __init__(
        self,
        dataset_config: Union[CIFAR10DatasetConfig, CIFAR100DatasetConfig],
        data_loader_config: DataLoaderConfig,
        transform_train: Any,
        transform_eval: Any,
    ):
        super(TwoSplitDataModule, self).__init__(
            dataset_config, data_loader_config
        )

        self.transform_train = transform_train
        self.transform_eval = transform_eval

    def setup(self, stage: Optional[str] = None):

        if stage == "fit" or stage is None:
            train_set = hydra.utils.instantiate(
                config=self.dataset_config,
                dataset_root=self.dataset_root,
                train=True,
                input_transform=self.transform_train.input_transform,
            )

            num_training_items = int(
                len(train_set) * (1.0 - self.dataset_config.val_set_percentage)
            )

            num_val_items = len(train_set) - num_training_items

            self.train_set, self.val_set = torch.utils.data.random_split(
                train_set,
                [num_training_items, num_val_items],
                generator=torch.Generator().manual_seed(self.seed),
            )

            self.input_shape_dict = self.train_set.input_shape_dict
            self.target_shape_dict = self.train_set.target_shape_dict

        elif stage == "validate":
            train_set = hydra.utils.instantiate(
                config=self.dataset_config,
                dataset_root=self.dataset_root,
                train=True,
                input_transform=self.transform_eval.input_transform,
            )

            num_training_items = int(
                len(train_set) * (1.0 - self.dataset_config.val_set_percentage)
            )

            num_val_items = len(train_set) - num_training_items

            _, self.val_set = torch.utils.data.random_split(
                train_set,
                [num_training_items, num_val_items],
                generator=torch.Generator().manual_seed(self.seed),
            )

            self.input_shape_dict = self.val_set.input_shape_dict
            self.target_shape_dict = self.val_set.target_shape_dict

        elif stage == "test":
            self.test_set = hydra.utils.instantiate(
                config=self.dataset_config,
                dataset_root=self.dataset_root,
                train=False,
                input_transform=self.transform_eval.input_transform,
            )

            self.input_shape_dict = self.test_set.input_shape_dict
            self.target_shape_dict = self.test_set.target_shape_dict

        else:
            raise ValueError(f"Invalid stage name passed {stage}")

    def dummy_batch(self):

        if getattr(self, "target_shape_dict") is None:
            self.setup(stage="fit")

        return (
            {
                "image": torch.randn(
                    2,
                    self.input_shape_dict["image"]["channels"],
                    self.input_shape_dict["image"]["height"],
                    self.input_shape_dict["image"]["width"],
                )
            },
            {
                "image": torch.randint(
                    0, self.target_shape_dict["image"]["num_classes"], (2,)
                )
            },
        )

    def train_dataloader(self):

        return DataLoader(
            self.train_set,
            batch_size=self.data_loader_config.train_batch_size,
            shuffle=self.data_loader_config.train_shuffle,
            num_workers=self.data_loader_config.num_workers,
            pin_memory=self.data_loader_config.pin_memory,
            prefetch_factor=self.data_loader_config.prefetch_factor,
            persistent_workers=self.data_loader_config.persistent_workers,
            drop_last=self.data_loader_config.train_drop_last,
        )

    def val_dataloader(self):

        return DataLoader(
            self.val_set,
            batch_size=self.data_loader_config.val_batch_size,
            shuffle=self.data_loader_config.eval_shuffle,
            num_workers=self.data_loader_config.num_workers,
            pin_memory=self.data_loader_config.pin_memory,
            prefetch_factor=self.data_loader_config.prefetch_factor,
            persistent_workers=self.data_loader_config.persistent_workers,
            drop_last=self.data_loader_config.eval_drop_last,
        )

    def test_dataloader(self):

        return DataLoader(
            self.test_set,
            batch_size=self.data_loader_config.test_batch_size,
            shuffle=self.data_loader_config.eval_shuffle,
            num_workers=self.data_loader_config.num_workers,
            pin_memory=self.data_loader_config.pin_memory,
            prefetch_factor=self.data_loader_config.prefetch_factor,
            persistent_workers=self.data_loader_config.persistent_workers,
            drop_last=self.data_loader_config.eval_drop_last,
        )

    def predict_dataloader(self):
        return self.test_dataloader()


class PreSplitDataModule(TwoSplitDataModule):
    def __init__(
        self,
        dataset_config: PreSplitDatasetConfig,
        data_loader_config: DataLoaderConfig,
        transform_train: Any,
        transform_eval: Any,
        split_name_to_phase_dict: Dict[str, str],
    ):
        self.split_name_to_phase_dict = split_name_to_phase_dict
        super(PreSplitDataModule, self).__init__(
            dataset_config=dataset_config,
            data_loader_config=data_loader_config,
            transform_train=transform_train,
            transform_eval=transform_eval,
        )

    def setup(self, stage: Optional[str] = None):

        if stage == "fit" or stage is None:
            self.train_set = hydra.utils.instantiate(
                config=self.dataset_config,
                dataset_root=self.dataset_root,
                split_name=self.split_name_to_phase_dict["train"],
                input_transform=self.transform_train.input_transform,
            )

            self.val_set = hydra.utils.instantiate(
                config=self.dataset_config,
                dataset_root=self.dataset_root,
                split_name=self.split_name_to_phase_dict["val"],
                input_transform=self.transform_eval.input_transform,
            )

            self.input_shape_dict = self.train_set.input_shape_dict
            self.target_shape_dict = self.train_set.target_shape_dict

        elif stage == "validate":
            self.val_set = hydra.utils.instantiate(
                config=self.dataset_config,
                dataset_root=self.dataset_root,
                split_name=self.split_name_to_phase_dict["val"],
                input_transform=self.transform_eval.input_transform,
            )

            self.input_shape_dict = self.val_set.input_shape_dict
            self.target_shape_dict = self.val_set.target_shape_dict

        # Assign test dataset for use in dataloader(s)
        elif stage == "test":
            self.test_set = hydra.utils.instantiate(
                config=self.dataset_config,
                dataset_root=self.dataset_root,
                split_name=self.split_name_to_phase_dict["test"],
                input_transform=self.transform_eval.input_transform,
            )

            self.input_shape_dict = self.test_set.input_shape_dict
            self.target_shape_dict = self.test_set.target_shape_dict

        else:
            raise ValueError(f"Invalid stage name passed {stage}")
