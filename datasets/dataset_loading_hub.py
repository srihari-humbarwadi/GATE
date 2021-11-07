import logging
import os
from collections import namedtuple

import torch.utils.data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from rich import print
from rich.logging import RichHandler
from torch import nn
from torch.utils.data import Subset

from datasets.custom_transforms import SimCLRTransform
from datasets.TALI_dataset import (MultiModalDataset, SubSampleAudioFrames,
                                   SubSampleVideoFrames)
from models.tokenizers import SimpleTokenizer, tokenize
from utils.cinic_utils import download_cinic, extend_cinic_10

ImageShape = namedtuple("ImageShape", ["channels", "width", "height"])

FORMAT = "%(message)s"
logging.basicConfig(
    level=logging.INFO, format=FORMAT, datefmt="[%X]", handlers=[RichHandler()]
)

logging = logging.getLogger("rich")


class BPETokenizer(nn.Module):
    def __init__(self, context_length):
        super(BPETokenizer, self).__init__()
        self.tokenizer = SimpleTokenizer()
        self.context_length = context_length

    def forward(self, x):
        return tokenize(x, tokenizer=self.tokenizer, context_length=self.context_length)


class MultiModalLoader:
    def __init__(
            self,
            image_width,
            image_height,
            num_video_frames_per_datapoint,
            num_audio_frames_per_datapoint=44000,
            text_context_length=77,
            rescan_paths=False,
            restrict_train_set_size=None,
    ):
        self.rescan_paths = rescan_paths
        self.image_shape = ImageShape(3, image_height, image_width)
        self.restrict_train_set_size = restrict_train_set_size

        # normalize = transforms.Normalize(
        #     mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        # )
        self.transform_train = {
            "image": transforms.Compose(
                [
                    transforms.Resize(size=(image_height, image_width)),
                ]
            ),
            "audio": transforms.Compose(
                [
                    SubSampleAudioFrames(num_frames=num_audio_frames_per_datapoint),
                ]
            ),
            "video": transforms.Compose(
                [
                    SubSampleVideoFrames(num_frames=num_video_frames_per_datapoint),
                    transforms.Resize(size=(image_height, image_width)),
                ]
            ),
            "text": transforms.Compose(
                [
                    BPETokenizer(context_length=text_context_length),
                ]
            ),
        }

        self.transform_validate = {
            "image": transforms.Compose(
                [
                    transforms.Resize(size=(image_height, image_width)),
                ]
            ),
            "audio": transforms.Compose(
                [
                    SubSampleAudioFrames(num_frames=num_audio_frames_per_datapoint),
                ]
            ),
            "video": transforms.Compose(
                [
                    SubSampleVideoFrames(num_frames=num_video_frames_per_datapoint),
                    transforms.Resize(size=(image_height, image_width)),
                ]
            ),
            "text": transforms.Compose(
                [
                    BPETokenizer(context_length=text_context_length),
                ]
            ),
        }

    def get_data(self, data_filepath, set_cutoff_size, exclude_modalities, **kwargs):
        train_set = MultiModalDataset(
            dataset_root=data_filepath,
            set_name="train",
            restrict_train_set_size=self.restrict_train_set_size,
            rescan_paths=self.rescan_paths,
            transforms=self.transform_train,
            download=False,
            exclude_modalities=exclude_modalities,
        )

        val_set = MultiModalDataset(
            dataset_root=data_filepath,
            set_name="val",
            restrict_train_set_size=self.restrict_train_set_size,
            rescan_paths=self.rescan_paths,
            transforms=self.transform_validate,
            download=False,
            exclude_modalities=exclude_modalities,
        )

        test_set = MultiModalDataset(
            dataset_root=data_filepath,
            set_name="test",
            restrict_train_set_size=self.restrict_train_set_size,
            rescan_paths=self.rescan_paths,
            transforms=self.transform_validate,
            download=False,
            exclude_modalities=exclude_modalities,
        )

        return train_set, val_set, test_set


# build multi view augmentations
class CIFAR100Loader:
    def __init__(self):
        normalize = transforms.Normalize(
            mean=[0.5071, 0.4866, 0.4409], std=[0.2009, 0.1984, 0.2023]
        )
        self.image_shape = ImageShape(3, 32, 32)
        self.transform_train = transforms.Compose(
            [
                transforms.RandomCrop(self.image_shape.width, padding=4),
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

    def get_data(
            self, data_filepath, val_set_percentage, random_split_seed, download=False
    ):
        train_set = datasets.CIFAR100(
            root=data_filepath,
            train=True,
            download=download,
            transform=self.transform_train,
        )

        num_training_items = int(len(train_set) * (1.0 - val_set_percentage))
        num_val_items = len(train_set) - num_training_items

        train_set, val_set = torch.utils.data.random_split(
            train_set,
            [num_training_items, num_val_items],
            generator=torch.Generator().manual_seed(random_split_seed),
        )

        test_set = datasets.CIFAR100(
            root=data_filepath,
            train=False,
            download=download,
            transform=self.transform_validate,
        )

        num_labels = 100
        return train_set, val_set, test_set, num_labels


class CIFAR10Loader:
    def __init__(self):
        normalize = transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]
        )
        self.image_shape = ImageShape(3, 32, 32)
        self.transform_train = transforms.Compose(
            [
                transforms.RandomCrop(self.image_shape.width, padding=4),
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

    def get_data(
            self, data_filepath, val_set_percentage, random_split_seed, download=False
    ):
        train_set = datasets.CIFAR10(
            root=data_filepath,
            train=True,
            download=download,
            transform=self.transform_train,
        )

        num_training_items = int(len(train_set) * (1.0 - val_set_percentage))
        num_val_items = len(train_set) - num_training_items

        train_set, val_set = torch.utils.data.random_split(
            train_set,
            [num_training_items, num_val_items],
            generator=torch.Generator().manual_seed(random_split_seed),
        )

        test_set = datasets.CIFAR10(
            root=data_filepath,
            train=False,
            download=download,
            transform=self.transform_validate,
        )

        num_labels = 10
        return train_set, val_set, test_set, num_labels


def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    # logging.info(len(batch))
    return torch.utils.data.dataloader.default_collate(batch)


def load_dataset(
        dataset,
        data_filepath,
        image_height,
        image_width,
        exclude_modalities,
        batch_size=128,
        test_batch_size=128,
        num_workers=0,
        num_video_frames_per_datapoint=30,
        num_audio_frames_per_datapoint=44000,
        text_context_length=77,
        download=False,
        prefetch_factor=2,
        rescan_dataset_files=False,
        restrict_train_set_size=None,
):
    datasets = {
        "tali": MultiModalLoader(
            image_height=image_height,
            image_width=image_width,
            num_video_frames_per_datapoint=num_video_frames_per_datapoint,
            num_audio_frames_per_datapoint=num_audio_frames_per_datapoint,
            text_context_length=text_context_length,
            rescan_paths=rescan_dataset_files,
            restrict_train_set_size=restrict_train_set_size,
        ),
    }

    dataloader = datasets[dataset.lower()]

    train_set, val_set, test_set = dataloader.get_data(
        set_cutoff_size=0,
        data_filepath=data_filepath,
        exclude_modalities=exclude_modalities,
    )

    dummy_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=2,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,
        prefetch_factor=prefetch_factor,
        collate_fn=collate_fn,
        persistent_workers=False,
        drop_last=True,
    )

    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,
        prefetch_factor=prefetch_factor,
        collate_fn=collate_fn,
        persistent_workers=False,
        drop_last=True,
    )

    val_loader = torch.utils.data.DataLoader(
        val_set,
        batch_size=test_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
        prefetch_factor=prefetch_factor,
        collate_fn=collate_fn,
        persistent_workers=False,
        drop_last=True,
    )

    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=test_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
        prefetch_factor=prefetch_factor,
        collate_fn=collate_fn,
        persistent_workers=False,
        drop_last=True,
    )

    return (
        dummy_loader,
        train_loader,
        val_loader,
        test_loader,
        train_set,
        val_set,
        test_set,
        dataloader.image_shape,
    )


def load_split_datasets(dataset, split_tuple):
    total_length = len(dataset)
    total_idx = [i for i in range(total_length)]

    start_end_index_tuples = [
        (
            int(len(total_idx) * sum(split_tuple[: i - 1])),
            int(len(total_idx) * split_tuple[i]),
        )
        for i in range(len(split_tuple))
    ]

    set_selection_index_lists = [
        total_idx[start_idx:end_idx] for (start_idx, end_idx) in start_end_index_tuples
    ]

    return (Subset(dataset, set_indices) for set_indices in set_selection_index_lists)
