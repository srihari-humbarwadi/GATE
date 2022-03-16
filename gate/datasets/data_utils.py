import torch.utils.data
from numpy import random
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


def collate_fn_replace_corrupted(batch, dataset):
    """Collate function that allows to replace corrupted examples in the
    dataloader. It expect that the dataloader returns 'None' when that occurs.
    The 'None's in the batch are replaced with another examples sampled randomly.

    Args:
        batch (torch.Tensor): batch from the DataLoader.
        dataset (torch.utils.data.Dataset): dataset which the DataLoader is loading.
            Specify it with functools.partial and pass the resulting partial function that only
            requires 'batch' argument to DataLoader's 'collate_fn' option.

    Returns:
        torch.Tensor: batch with new examples instead of corrupted ones.
    """
    # Idea from https://stackoverflow.com/a/57882783

    original_batch_len = len(batch)
    # Filter out all the Nones (corrupted examples)
    batch = list(filter(lambda x: x is not None, batch))
    filtered_batch_len = len(batch)
    # Num of corrupted examples
    diff = original_batch_len - filtered_batch_len
    if diff > 0:
        # Replace corrupted examples with another examples randomly
        batch.extend([dataset[random.randint(0, len(dataset))] for _ in range(diff)])
        # Recursive call to replace the replacements if they are corrupted
        return collate_fn_replace_corrupted(batch, dataset)
    # Finally, when the whole batch is fine, return it
    return torch.utils.data.dataloader.default_collate(batch)
