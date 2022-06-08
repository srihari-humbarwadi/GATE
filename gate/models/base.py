from typing import Union

import torch.nn as nn
from dotted_dict import DottedDict

from gate.configs.datamodule.base import ShapeConfig


class MissingModalityForward(Exception):
    pass


def generic_missing_forward(module, modality_name):
    raise MissingModalityForward(
        f"forward_{modality_name} method not implemented in model: "
        f"{module.__class__.__name__} "
    )


class ModelModule(nn.Module):
    def __init__(
        self,
        input_shape_dict: Union[ShapeConfig, DottedDict],
    ):
        super().__init__()
        self.input_shape_dict = input_shape_dict
