import logging
from collections import namedtuple
from typing import Optional

import torch.utils.data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Subset

from gate.datasets.data_utils import collate_resample_none
from gate.datasets.datasets import CIFAR10ClassificationsDict, \
    CIFAR100ClassificationDict
from gate.utils.arg_parsing import DictWithDotNotation


class BaseDataModule(LightningDataModule):
    def __init__(self, **kwargs):
        super(BaseDataModule, self).__init__()

    def prepare_data(self, **kwargs):
        raise NotImplementedError

    def configure_dataloaders(self, **kwargs):
        raise NotImplementedError

    def setup(self, stage: Optional[str] = None):
        raise NotImplementedError

    @staticmethod
    def add_dataset_specific_args(self):
        raise NotImplementedError

    def dummy_dataloader(self):
        raise NotImplementedError

    def train_dataloader(self):
        raise NotImplementedError

    def val_dataloader(self):
        raise NotImplementedError

    def test_dataloader(self):
        raise NotImplementedError


class CIFAR10DataModule(BaseDataModule):
    def __init__(self, data_args, general_args, task_args, **kwargs):
        super(CIFAR10DataModule, self).__init__()
        normalize = transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]
        )
        self.dataset_name = data_args.name
        self.data_filepath = data_args.data_filepath
        self.seed = general_args.seed
        self.data_args = data_args

        self.input_shape_dict = {"image": (3, 32, 32)}
        self.output_shape_dict = {"image": (10,)}
        self.val_set_percentage = self.data_args.val_set_percentage
        self.download = self.data_args.download

        self.transform_train = transforms.Compose(
            [
                transforms.RandomCrop(self.input_shape_dict["image"][1], padding=4),
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

        self.save_hyperparameters()

    @staticmethod
    def add_dataset_specific_args(parser):
        parser.add_argument("--dataset.val_set_percentage", type=float, default=0.1)
        parser.add_argument("--dataset.download", default=False, action="store_true")

        return parser

    def configure_dataloaders(
            self,
            batch_size=128,
            eval_batch_size=128,
            num_workers=0,
            prefetch_factor=2,
            collate_fn=collate_resample_none,
    ):

        self.data_loader_config = DictWithDotNotation(
            dict(
                batch_size=batch_size,
                train_shuffle=True,
                eval_shuffle=False,
                num_workers=num_workers,
                pin_memory=False,
                prefetch_factor=prefetch_factor,
                collate_fn=collate_fn,
                persistent_workers=False,
                train_drop_last=True,
                eval_drop_last=False,
            )
        )

    def prepare_data(self, **kwargs):
        # download
        pass

    def setup(self, stage: Optional[str] = None):

        if stage == "fit" or stage is None:
            train_set = CIFAR10ClassificationsDict(
                root=self.data_filepath,
                train=True,
                download=self.download,
                transform=self.transform_train,
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
            self.test_set = CIFAR10ClassificationsDict(
                root=self.data_filepath,
                train=False,
                download=self.download,
                transform=self.transform_validate,
            )

    def dummy_dataloader(self):
        return DataLoader(
            self.val_set,
            batch_size=2,
            shuffle=self.data_loader_config.eval_shuffle,
            num_workers=self.data_loader_config.num_workers,
            pin_memory=self.data_loader_config.pin_memory,
            prefetch_factor=self.data_loader_config.prefetch_factor,
            collate_fn=self.data_loader_config.collate_fn,
            persistent_workers=self.data_loader_config.persistent_workers,
            drop_last=self.data_loader_config.eval_drop_last,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_set,
            batch_size=self.data_loader_config.batch_size,
            shuffle=self.data_loader_config.train_shuffle,
            num_workers=self.data_loader_config.num_workers,
            pin_memory=self.data_loader_config.pin_memory,
            prefetch_factor=self.data_loader_config.prefetch_factor,
            collate_fn=self.data_loader_config.collate_fn,
            persistent_workers=self.data_loader_config.persistent_workers,
            drop_last=self.data_loader_config.train_drop_last,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_set,
            batch_size=self.data_loader_config.batch_size,
            shuffle=self.data_loader_config.eval_shuffle,
            num_workers=self.data_loader_config.num_workers,
            pin_memory=self.data_loader_config.pin_memory,
            prefetch_factor=self.data_loader_config.prefetch_factor,
            collate_fn=self.data_loader_config.collate_fn,
            persistent_workers=self.data_loader_config.persistent_workers,
            drop_last=self.data_loader_config.eval_drop_last,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_set,
            batch_size=self.data_loader_config.batch_size,
            shuffle=self.data_loader_config.eval_shuffle,
            num_workers=self.data_loader_config.num_workers,
            pin_memory=self.data_loader_config.pin_memory,
            prefetch_factor=self.data_loader_config.prefetch_factor,
            collate_fn=self.data_loader_config.collate_fn,
            persistent_workers=self.data_loader_config.persistent_workers,
            drop_last=self.data_loader_config.eval_drop_last,
        )


class CIFAR100DataModule(CIFAR10DataModule):
    def __init__(self, data_args, general_args, task_args, **kwargs):
        super(CIFAR100DataModule, self).__init__(
            data_args, general_args, task_args, **kwargs
        )

        normalize = transforms.Normalize(
            mean=[0.5071, 0.4866, 0.4409], std=[0.2009, 0.1984, 0.2023]
        )
        self.output_shape_dict = {"image": (100,)}
        self.transform_train = transforms.Compose(
            [
                transforms.RandomCrop(self.input_shape_dict["image"][1], padding=4),
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

        self.val_set_percentage = self.data_args.val_set_percentage
        self.download = self.data_args.download

    def setup(self, stage: Optional[str] = None):

        if stage == "fit" or stage is None:
            train_set = CIFAR100ClassificationDict(
                root=self.data_filepath,
                train=True,
                download=self.download,
                transform=self.transform_train,
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
            self.test_set = CIFAR100ClassificationDict(
                root=self.data_filepath,
                train=False,
                download=self.download,
                transform=self.transform_validate,
            )
