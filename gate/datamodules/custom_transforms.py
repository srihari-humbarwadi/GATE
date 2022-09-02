import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision.transforms import ToTensor


class Cutout:
    """
    Taken primarily from: https://github.com/uoguelph-mlrg/Cutout
    Randomly mask out one or more patches from an image.
    """

    def __init__(self, n_holes, length):
        """
        :param n_holes: Number of patches to cut out of each image.
        :param length: The length (in pixels) of each square patch.
        """
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        """
        :param img: Tensor image of size (C, H, W).
        :return: Image with n_holes of dimension length x length cut out of it.
        """
        h = img.size(1)
        w = img.size(2)

        mask = np.ones((h, w), np.float32)

        for _ in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1:y2, x1:x2] = 0.0

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img


class SimCLRTransform:
    """
    credit: https://github.com/sthalles/SimCLR/
    """

    def __init__(self, size, s=1, n_views=2):
        color_jitter = transforms.ColorJitter(
            0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s
        )
        base_transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(size=size),
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply([color_jitter], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.GaussianBlur(kernel_size=int(0.1 * size)),
                transforms.ToTensor(),
            ]
        )
        self.base_transform = base_transform
        self.n_views = n_views

    def __call__(self, x):
        return [self.base_transform(x) for _ in range(self.n_views)]


class UnNormalize:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, x):
        return transforms.Compose(
            [
                transforms.Normalize(
                    mean=[-m / s for m, s in zip(self.mean, self.std)],
                    std=[1.0 / s for s in self.std],
                ),
                lambda x: x * 255,
            ]
        )(x)


class LoggingTransform(torch.nn.Module):
    def __init__(self, level="INFO"):
        super().__init__()
        self.level = level

    def __call__(self, x):
        if hasattr(x, "shape"):
            print(f"Shape going in {x.shape}")
        else:
            print(f"Shape going in {ToTensor()(x).shape}")

        print(f"Type going in {type(x)}")
        return x


class ChannelCloneIfNecessary(object):
    def __init__(self, num_channels=3):
        self.num_channels = num_channels

    def __call__(self, x):
        if x.shape[0] != self.num_channels:
            return x.repeat(self.num_channels, 1, 1)
        return x
