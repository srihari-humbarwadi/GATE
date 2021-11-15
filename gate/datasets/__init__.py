import torch
from torch.utils.data import DataLoader

from gate.datasets.dataset_loading_hub import CIFAR10Loader, CIFAR100Loader, collate_fn

datasets = {
    "cifar10": CIFAR10Loader,
    "cifar100": CIFAR100Loader,
}


def load_dataset(
        dataset_name,
        data_filepath,
        seed,
        data_args,
        batch_size=128,
        eval_batch_size=128,
        num_workers=0,
        prefetch_factor=2,
):
    dataloader = datasets[dataset_name.lower()]

    train_set, val_set, test_set = dataloader.get_data(
        data_filepath=data_filepath, random_split_seed=seed, **data_args
    )

    dummy_loader = DataLoader(
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

    train_loader = DataLoader(
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

    val_loader = DataLoader(
        val_set,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
        prefetch_factor=prefetch_factor,
        collate_fn=collate_fn,
        persistent_workers=False,
        drop_last=True,
    )

    test_loader = DataLoader(
        test_set,
        batch_size=eval_batch_size,
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
