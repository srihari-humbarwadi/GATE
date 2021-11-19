from collections import namedtuple

import torch.utils.data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import Subset


def collate_resample_none(batch):
    batch = list(filter(lambda x: x is not None, batch))
    # logging.info(len(batch))
    return torch.utils.data.dataloader.default_collate(batch)


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
