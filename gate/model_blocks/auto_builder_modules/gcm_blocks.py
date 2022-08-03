from typing import Any, Callable, Dict, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

import gate.base.utils.loggers as loggers

log = loggers.get_logger(__name__)


class HeadConv(nn.Module):
    def __init__(
        self,
        num_output_filters: int,
        num_layers: int,
        num_hidden_filters: int,
        input_avg_pool_size: int = 2,
        output_activation_fn: Callable = None,
    ):
        super(HeadConv, self).__init__()
        self.layer_dict = nn.ModuleDict()
        self.num_output_filters = num_output_filters
        self.num_layers = num_layers
        self.num_hidden_filters = num_hidden_filters
        self.output_activation_fn = output_activation_fn
        self.input_avg_pool_size = input_avg_pool_size

    def build(self, input_shape: Tuple[int, ...]):

        dummy_x = torch.zeros(input_shape)  # this should be b, f; or b, c, h, w
        dummy_x = F.adaptive_avg_pool2d(dummy_x, output_size=self.input_avg_pool_size)

        # out = dummy_x.view(dummy_x.shape[0], -1)
        # self.embedding_size = out.shape[-1]
        out = dummy_x

        self.layer_dict["input_layer"] = nn.Conv2d(
            in_channels=out.shape[1],
            out_channels=self.num_hidden_filters,
            kernel_size=3,
            padding=1,
        )
        self.layer_dict["input_layer_norm"] = nn.InstanceNorm2d(
            num_features=self.num_hidden_filters
        )

        self.layer_dict["input_layer_act"] = nn.LeakyReLU()

        out = self.layer_dict["input_layer"](out)
        out = self.layer_dict["input_layer_norm"](out)
        out = self.layer_dict["input_layer_act"](out)

        for i in range(self.num_layers - 2):
            self.layer_dict[f"hidden_layer_{i}"] = nn.Conv2d(
                in_channels=self.num_hidden_filters,
                out_channels=self.num_hidden_filters,
                kernel_size=3,
                padding=1,
            )
            self.layer_dict[f"hidden_layer_norm_{i}"] = nn.InstanceNorm2d(
                num_features=self.num_hidden_filters
            )

            self.layer_dict[f"hidden_layer_act_{i}"] = nn.LeakyReLU()

            out = self.layer_dict[f"hidden_layer_{i}"](out)
            out = self.layer_dict[f"hidden_layer_norm_{i}"](out)
            out = self.layer_dict[f"hidden_layer_act_{i}"](out)

        self.layer_dict["output_pooling"] = nn.AvgPool2d(kernel_size=out.shape[-1])

        self.layer_dict["output_layer"] = nn.Linear(
            in_features=self.num_hidden_filters,
            out_features=self.num_hidden_filters,
        )

        out = self.layer_dict["output_pooling"](out).squeeze(dim=2).squeeze(dim=2)

        out = self.layer_dict["output_layer"](out)

        log.info(
            f"Built HeadConv with input_shape {dummy_x.shape} "
            f"output shape {out.shape} 👍"
        )

    def forward(self, input_dict: Dict[str, torch.Tensor]) -> Dict[str, Tensor]:

        out = input_dict["image"]
        out = F.adaptive_avg_pool2d(out, output_size=self.input_avg_pool_size)

        # log.info(out.shape)
        out = self.layer_dict["input_layer_norm"](self.layer_dict["input_layer"](out))
        out = self.layer_dict["input_layer_act"](out)

        for i in range(self.num_layers - 2):
            out = self.layer_dict[f"hidden_layer_{i}"](out)
            out = self.layer_dict[f"hidden_layer_norm_{i}"](out)
            out = self.layer_dict[f"hidden_layer_act_{i}"](out)

        out = self.layer_dict["output_pooling"](out).squeeze(dim=2).squeeze(dim=2)

        out = self.layer_dict["output_layer"](out)
        # log.info(self.output_activation_fn)
        if self.output_activation_fn is not None:
            out = self.output_activation_fn(out)

        log.debug(
            f"mean {out.mean()} std {out.std()} " f"min {out.min()} max {out.max()} "
        )

        return {"image": out}


class HeadMLP(nn.Module):
    def __init__(
        self,
        num_output_filters: int,
        num_layers: int,
        num_hidden_filters: int,
        input_avg_pool_size: int,
        output_activation_fn: Callable = None,
    ):
        super(HeadMLP, self).__init__()
        self.layer_dict = nn.ModuleDict()
        self.num_output_filters = num_output_filters
        self.num_layers = num_layers
        self.num_hidden_filters = num_hidden_filters
        self.output_activation_fn = output_activation_fn
        self.input_avg_pool_size = input_avg_pool_size

    def build(self, input_shape: Tuple[int, ...]):

        dummy_x = torch.zeros(input_shape)  # this should be b, f; or b, c, h, w
        out = F.adaptive_avg_pool2d(dummy_x, output_size=self.input_avg_pool_size)
        out = out.view(out.shape[0], -1)

        self.layer_dict["input_layer"] = nn.Linear(
            in_features=out.shape[1], out_features=self.num_hidden_filters
        )
        self.layer_dict["input_layer_norm"] = nn.InstanceNorm1d(
            num_features=self.num_hidden_filters
        )

        self.layer_dict["input_layer_act"] = nn.LeakyReLU()

        out = self.layer_dict["input_layer"](out)
        out = self.layer_dict["input_layer_norm"](out)
        out = self.layer_dict["input_layer_act"](out)

        for i in range(self.num_layers - 2):
            self.layer_dict[f"hidden_layer_{i}"] = nn.Linear(
                in_features=self.num_hidden_filters,
                out_features=self.num_hidden_filters,
            )
            self.layer_dict[f"hidden_layer_norm_{i}"] = nn.InstanceNorm1d(
                num_features=self.num_hidden_filters
            )

            self.layer_dict[f"hidden_layer_act_{i}"] = nn.LeakyReLU()

            out = self.layer_dict[f"hidden_layer_{i}"](out)
            out = self.layer_dict[f"hidden_layer_norm_{i}"](out)
            out = self.layer_dict[f"hidden_layer_act_{i}"](out)

        self.layer_dict["output_layer"] = nn.Linear(
            in_features=out.shape[1], out_features=self.num_hidden_filters
        )
        self.layer_dict["output_layer_norm"] = nn.InstanceNorm1d(
            num_features=self.num_hidden_filters
        )

        self.layer_dict["output_layer_act"] = nn.LeakyReLU()

        out = self.layer_dict["output_layer_norm"](self.layer_dict["output_layer"](out))
        out = self.layer_dict["output_layer_act"](out)

        log.info(
            f"Built HeadMLP with input_shape {dummy_x.shape} "
            f"output shape {out.shape} 👍"
        )

    def forward(self, input_dict: Dict[str, torch.Tensor]) -> Dict[str, Tensor]:

        out = input_dict["image"]
        # log.info(out.shape)
        out = F.adaptive_avg_pool2d(out, output_size=self.input_avg_pool_size)

        out = out.view(out.shape[0], -1)

        out = self.layer_dict["input_layer"](out)
        out = self.layer_dict["input_layer_norm"](out)
        out = self.layer_dict["input_layer_act"](out)

        for i in range(self.num_layers - 2):
            out = self.layer_dict[f"hidden_layer_{i}"](out)
            out = self.layer_dict[f"hidden_layer_norm_{i}"](out)
            out = self.layer_dict[f"hidden_layer_act_{i}"](out)

        out = self.layer_dict["output_layer_norm"](self.layer_dict["output_layer"](out))
        # log.info(self.output_activation_fn)
        if self.output_activation_fn is not None:
            out = self.output_activation_fn(out)

        log.debug(
            f"mean {out.mean()} std {out.std()} " f"min {out.min()} max {out.max()} "
        )

        return {"image": out}
