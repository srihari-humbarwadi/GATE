import logging

import torch
from torch.utils.data import DataLoader

from gate.datasets.dataset_hub import CIFAR10DataModule, CIFAR100DataModule

datasets_library_dict = {
    "cifar10": CIFAR10DataModule,
    "cifar100": CIFAR100DataModule,
}
