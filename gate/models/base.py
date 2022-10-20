import torch.nn as nn
from omegaconf import DictConfig


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
        input_shape_dict: DictConfig,
    ):
        super().__init__()
        self.input_shape_dict = input_shape_dict
