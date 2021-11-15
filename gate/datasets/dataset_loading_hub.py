from collections import namedtuple

import torch.utils.data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import Subset

ImageShape = namedtuple("ImageShape", ["channels", "width", "height"])


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

    @staticmethod
    def add_dataset_specific_args(parser):
        parser.add_argument("--data.download", default=False, action="store_true")
        parser.add_argument("--data.val_set_percentage", type=float, default=0.1)
        return parser

    def get_data(
            self,
            data_filepath,
            val_set_percentage,
            random_split_seed,
            download=False,
            **kwargs
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

    @staticmethod
    def add_dataset_specific_args(parser):
        parser.add_argument("--data.download", default=False, action="store_true")
        parser.add_argument("--data.val_set_percentage", type=float, default=0.1)
        return parser

    def get_data(
            self,
            data_filepath,
            val_set_percentage,
            random_split_seed,
            download=False,
            **kwargs
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


def add_dataset_args(parser):
    datasets = {
        "cifar10": CIFAR10Loader,
        "cifar100": CIFAR100Loader,
    }

    parser = datasets[parser.parse_args().dataset].add_dataset_specific_args(parser)

    return parser


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
