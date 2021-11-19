import torch
from torchvision.datasets import CIFAR10, CIFAR100


class CIFAR10Dict(CIFAR10):
    def __init__(self, root, train, download, transform):
        super(CIFAR10Dict, self).__init__(
            root=root, train=train, download=download, transform=transform
        )

    def __getitem__(self, index):
        x, y = super().__getitem__(index)
        return {"image": x}, {"image": torch.Tensor([y]).type(torch.LongTensor)}


class CIFAR100Dict(CIFAR100):
    def __init__(self, root, train, download, transform):
        super(CIFAR100Dict, self).__init__(
            root=root, train=train, download=download, transform=transform
        )

    def __getitem__(self, index):
        x, y = super().__getitem__(index)
        return {"image": x}, {"image": torch.Tensor([y]).type(torch.LongTensor)}
