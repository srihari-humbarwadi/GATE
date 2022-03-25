from typing import Dict, Any, Union

import torch.nn as nn

from gate.class_configs.base import ShapeConfig


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
        input_modality_shape_config: ShapeConfig,
    ):
        super().__init__()
        self.input_modality_shape_config = input_modality_shape_config
