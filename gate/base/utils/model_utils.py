import numpy as np
import torch.nn.functional as F
from rich import print


def resize_custom(
    x_image, target_image_shape, interpolation="bilinear", debug=False
):
    """
    Resize an image to a target size.
    Parameters
    ----------
    x_image
    target_image_shape
    interpolation

    Returns
    -------

    """
    target_w = target_image_shape[1]
    target_h = target_image_shape[2]

    current_w = x_image.shape[2]
    current_h = x_image.shape[3]

    if current_w > target_w:
        x_image = x_image[:, :, :target_w]
        if debug:
            print(
                f"Condition met: current_w > target_w: Resized image from {current_w} to {target_w} == {x_image.shape}"
            )

    if current_h > target_h:
        x_image = x_image[:, :, :, :target_h]
        if debug:
            print(
                f"Condition met: current_h > target_h: Resized image from {current_h} to {target_h} == {x_image.shape}"
            )

    if current_w < target_w:
        pad_size = int(np.floor((target_w - current_w) / 2))
        p2dw = (0, 0, pad_size, pad_size)
        x_image = F.pad(x_image, p2dw, "constant", 0)
        if debug:
            print(
                f"Condition met: current_w < target_w: Resized image from {current_w} to {target_w} == {x_image.shape}"
            )

    if current_h < target_h:
        pad_size = int(np.floor((target_h - current_h) / 2))
        p2dh = (pad_size, pad_size, 0, 0)
        x_image = F.pad(x_image, p2dh, "constant", 0)
        if debug:
            print(
                f"Condition met: current_h < target_h: Resized image from {current_h} to {target_h} == {x_image.shape}"
            )

    return x_image


# import matplotlib.pyplot as plt
#
# x = torch.randn(16, 3, 512, 512) * 255.0
# x = resize_custom(x, (3, 128, 1024))
# x_grid = make_grid(x, nrow=4, padding=0, normalize=False)
# plt.imshow(x_grid.permute(1, 2, 0).cpu().numpy())
# plt.show()
