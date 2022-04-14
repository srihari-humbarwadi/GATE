import functools
from typing import Optional, Union

import torch.utils.data
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from gate.class_configs.base import (
    DataLoaderConfig,
    CIFAR10DatasetConfig,
    CIFAR100DatasetConfig,
)
from gate.datamodules.base import DataModule
from gate.datasets.cifar import (
    CIFAR10ClassificationDataset,
    CIFAR100ClassificationDataset,
)
from gate.datasets.data_utils import collate_fn_replace_corrupted


class CIFAR10DataModule(DataModule):
    def __init__(
        self,
        dataset_config: Union[CIFAR10DatasetConfig, CIFAR100DatasetConfig],
        data_loader_config: DataLoaderConfig,
    ):
        super(CIFAR10DataModule, self).__init__(dataset_config, data_loader_config)
        normalize = transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]
        )

        self.output_shape_dict = {"image": dict(num_classes=10)}
        self.val_set_percentage = self.dataset_config.val_set_percentage
        self.download = self.dataset_config.download

        self.transform_train = transforms.Compose(
            [
                transforms.RandomCrop(
                    self.input_shape_dict["image"]["width"], padding=4
                ),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]
        )

        self.transform_validate = transforms.Compose(
            [
                transforms.ToTensor(),
                normalize,
            ]
        )
        self.name = self.__class__.__name__
        self.save_hyperparameters()

    def prepare_data(self, **kwargs):
        # download
        pass

    def setup(self, stage: Optional[str] = None):

        if stage == "fit" or stage is None:
            train_set = CIFAR10ClassificationDataset(
                root=self.dataset_root,
                train=True,
                download=self.download,
                input_transform=self.transform_train,
            )

            num_training_items = int(len(train_set) * (1.0 - self.val_set_percentage))
            num_val_items = len(train_set) - num_training_items

            self.train_set, self.val_set = torch.utils.data.random_split(
                train_set,
                [num_training_items, num_val_items],
                generator=torch.Generator().manual_seed(self.seed),
            )

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.test_set = CIFAR10ClassificationDataset(
                root=self.dataset_root,
                train=False,
                download=self.download,
                input_transform=self.transform_validate,
            )

    def dummy_batch(self):
        return {
            "image": torch.randn(
                2,
                self.input_shape_dict["image"]["channels"],
                self.input_shape_dict["image"]["width"],
                self.input_shape_dict["image"]["height"],
            )
        }, {
            "image": torch.randint(
                0, self.output_shape_dict["image"]["num_classes"], (2,)
            )
        }

    def train_dataloader(self):

        collate_fn = functools.partial(
            collate_fn_replace_corrupted, dataset=self.train_set
        )

        return DataLoader(
            self.train_set,
            batch_size=self.data_loader_config.train_batch_size,
            shuffle=self.data_loader_config.train_shuffle,
            num_workers=self.data_loader_config.num_workers,
            pin_memory=self.data_loader_config.pin_memory,
            prefetch_factor=self.data_loader_config.prefetch_factor,
            collate_fn=collate_fn,
            persistent_workers=self.data_loader_config.persistent_workers,
            drop_last=self.data_loader_config.train_drop_last,
        )

    def val_dataloader(self):

        collate_fn = functools.partial(
            collate_fn_replace_corrupted, dataset=self.val_set
        )

        return DataLoader(
            self.val_set,
            batch_size=self.data_loader_config.val_batch_size,
            shuffle=self.data_loader_config.eval_shuffle,
            num_workers=self.data_loader_config.num_workers,
            pin_memory=self.data_loader_config.pin_memory,
            prefetch_factor=self.data_loader_config.prefetch_factor,
            collate_fn=collate_fn,
            persistent_workers=self.data_loader_config.persistent_workers,
            drop_last=self.data_loader_config.eval_drop_last,
        )

    def test_dataloader(self):

        collate_fn = functools.partial(
            collate_fn_replace_corrupted, dataset=self.test_set
        )

        return DataLoader(
            self.test_set,
            batch_size=self.data_loader_config.test_batch_size,
            shuffle=self.data_loader_config.eval_shuffle,
            num_workers=self.data_loader_config.num_workers,
            pin_memory=self.data_loader_config.pin_memory,
            prefetch_factor=self.data_loader_config.prefetch_factor,
            collate_fn=collate_fn,
            persistent_workers=self.data_loader_config.persistent_workers,
            drop_last=self.data_loader_config.eval_drop_last,
        )

    def predict_dataloader(self):
        return self.test_dataloader()


class CIFAR100DataModule(CIFAR10DataModule):
    def __init__(
        self,
        dataset_config: CIFAR100DatasetConfig,
        data_loader_config: DataLoaderConfig,
    ):
        super(CIFAR100DataModule, self).__init__(
            dataset_config=dataset_config, data_loader_config=data_loader_config
        )

        normalize = transforms.Normalize(
            mean=[0.5071, 0.4866, 0.4409], std=[0.2009, 0.1984, 0.2023]
        )
        self.output_shape_dict = {"image": dict(num_classes=100)}
        self.transform_train = transforms.Compose(
            [
                transforms.RandomCrop(
                    self.input_shape_dict["image"]["width"], padding=4
                ),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]
        )

        self.transform_validate = transforms.Compose(
            [
                transforms.ToTensor(),
                normalize,
            ]
        )
        self.name = self.__class__.__name__
        self.save_hyperparameters()

    def setup(self, stage: Optional[str] = None):

        if stage == "fit" or stage is None:
            train_set = CIFAR100ClassificationDataset(
                root=self.dataset_root,
                train=True,
                download=self.download,
                input_transform=self.transform_train,
            )

            num_training_items = int(len(train_set) * (1.0 - self.val_set_percentage))
            num_val_items = len(train_set) - num_training_items

            self.train_set, self.val_set = torch.utils.data.random_split(
                train_set,
                [num_training_items, num_val_items],
                generator=torch.Generator().manual_seed(self.seed),
            )

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.test_set = CIFAR100ClassificationDataset(
                root=self.dataset_root,
                train=False,
                download=self.download,
                input_transform=self.transform_validate,
            )
