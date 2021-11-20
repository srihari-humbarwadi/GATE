import torch
from torchvision.datasets import CIFAR10, CIFAR100


class CIFAR10ClassificationsDict(CIFAR10):
    def __init__(self, root, train, download, transform):
        super(CIFAR10ClassificationsDict, self).__init__(
            root=root, train=train, download=download, transform=transform
        )

    def __getitem__(self, index):
        x, y = super().__getitem__(index)
        return {"image": x}, {"image": torch.Tensor([y]).type(torch.LongTensor)}


class CIFAR100ClassificationDict(CIFAR100):
    def __init__(self, root, train, download, transform):
        super(CIFAR100ClassificationDict, self).__init__(
            root=root, train=train, download=download, transform=transform
        )

    def __getitem__(self, index):
        x, y = super().__getitem__(index)
        return {"image": x}, {"image": torch.Tensor([y]).type(torch.LongTensor)}


class CIFAR10ReconstructionDict(CIFAR10):
    def __init__(self, root, train, download, input_transform, target_transform):
        super(CIFAR10ReconstructionDict, self).__init__(
            root=root, train=train, download=download, transform=input_transform
        )
        self.target_transform = target_transform

    def __getitem__(self, index):
        x, y = super().__getitem__(index)
        input_images = x.copy()
        target_images = x.copy()
        if isinstance(self.target_transform, list):
            for transform in self.target_transform:
                target_images = transform(target_images)
        else:
            target_images = self.target_transform(target_images)

        return {"image": input_images}, {"image": target_images}


class CIFAR100ReconstructionDict(CIFAR100):
    def __init__(self, root, train, download, input_transform, target_transform):
        super(CIFAR100ReconstructionDict, self).__init__(
            root=root, train=train, download=download, transform=input_transform
        )
        self.target_transform = target_transform

    def __getitem__(self, index):
        x, y = super().__getitem__(index)
        input_images = x.copy()
        target_images = x.copy()
        if isinstance(self.target_transform, list):
            for transform in self.target_transform:
                target_images = transform(target_images)
        else:
            target_images = self.target_transform(target_images)

        return {"image": input_images}, {"image": target_images}
