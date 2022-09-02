from __future__ import print_function

import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from clip.model import LayerNorm, Transformer
from einops import rearrange, repeat

from gate.model_blocks.auto_builder_modules.conv_blocks import (
    ClassificationModel,
    SqueezeExciteConv1dBNLeakyReLU,
)


class FCCNetwork(nn.Module):
    def __init__(
        self,
        num_hidden_features,
        num_output_features,
        num_hidden_layers,
        activation_fn=F.leaky_relu,
    ):
        super(FCCNetwork, self).__init__()
        self.layer_dict = nn.ModuleDict()
        self.is_layer_built = False
        self.num_hidden_features = num_hidden_features
        self.num_output_features = num_output_features
        self.num_hidden_layers = num_hidden_layers
        self.activation_fn = activation_fn

    def build(self, input_shape):
        out = torch.zeros(input_shape)

        for i in range(self.num_hidden_layers):
            self.layer_dict[f"fcc_layer_{i}"] = nn.Linear(
                in_features=out.shape[1],
                out_features=self.num_hidden_features,
                bias=True,
            )
            out = self.activation_fn(
                self.layer_dict[f"fcc_layer_{i}"].forward(out)
            )

        self.layer_dict["fcc_layer_output"] = nn.Linear(
            in_features=out.shape[1],
            out_features=self.num_output_features,
            bias=True,
        )

        out = self.layer_dict["fcc_layer_output"].forward(out)

        self.is_layer_built = True

        logging.debug(
            f"Build {self.__class__.__name__} with input shape {input_shape} with "
            f"output shape {out.shape}"
        )

    def forward(self, x):
        if not self.is_layer_built:
            self.build(input_shape=x.shape)
            self.to(x.device)

        out = x

        for i in range(self.num_hidden_layers):
            out = self.activation_fn(
                self.layer_dict[f"fcc_layer_{i}"].forward(out)
            )

        out = self.layer_dict["fcc_layer_output"].forward(out)

        return out


class ChooseSpecificTimeStepFromVector(nn.Module):
    def __init__(self, time_step_to_choose):
        super(ChooseSpecificTimeStepFromVector, self).__init__()
        self.time_step_to_choose = time_step_to_choose

    def forward(self, x):
        return x, x[:, self.time_step_to_choose, :]


class Conv2DTransformer(nn.Module):
    def __init__(
        self,
        grid_patch_size: int,
        transformer_num_filters: int,
        transformer_num_layers: int,
        transformer_num_heads: int,
        transformer_dim_feedforward: int,
        stem_conv_bias: False,
    ):
        super(Conv2DTransformer, self).__init__()
        self.layer_dict = nn.ModuleDict()
        self.grid_patch_size = grid_patch_size
        self.transformer_num_filters = transformer_num_filters
        self.transformer_num_layers = transformer_num_layers
        self.transformer_num_heads = transformer_num_heads
        self.transformer_dim_feedforward = transformer_dim_feedforward
        self.stem_conv_bias = stem_conv_bias

        self.is_built = False

    def build(self, input_shape):
        dummy_x = torch.zeros(input_shape)

        ratio = (dummy_x.shape[2]) / self.grid_patch_size

        if not ratio.is_integer():
            ceiling = int(np.ceil(ratio))
            new_h = ceiling * self.grid_patch_size
            new_w = ceiling * self.grid_patch_size
            dummy_x = F.interpolate(
                dummy_x,
                size=(new_h, new_w),
            )

        out = dummy_x

        b, c, h, w = out.shape

        out = rearrange(
            out,
            "b f (h h1) (w w1) -> (b h w) (h1 w1 f)",
            h1=self.grid_patch_size,
            w1=self.grid_patch_size,
        )
        logging.debug(f"{out.shape}")
        num_patches = out.shape[0] / dummy_x.shape[0]

        self.layer_dict["stem_linear"] = nn.Linear(
            in_features=out.shape[1],
            out_features=int(self.transformer_num_filters / 2),
            bias=False,
        )

        out = self.layer_dict["stem_linear"].forward(out)
        # b, c, h, w
        logging.debug(f"{out.shape}")

        self.layer_dict["stem_layer_normalization"] = nn.LayerNorm(out.shape[1])

        out = self.layer_dict["stem_layer_normalization"].forward(out)
        logging.debug(f"{out.shape}")

        out = rearrange(out, "(b s) (f) -> b s f", s=int(num_patches))
        logging.debug(f"{out.shape}")

        self.enumerate_patches_idx = (
            torch.arange(start=0, end=num_patches) / num_patches
        )

        position_inputs = repeat(
            self.enumerate_patches_idx, "p -> b p", b=dummy_x.shape[0]
        )
        logging.debug(f"{position_inputs.shape}")

        position_inputs = rearrange(position_inputs, "b (p d) -> (b p) d", d=1)
        logging.debug(f"{position_inputs.shape}")

        self.layer_dict["positional_embedding_generator_network"] = FCCNetwork(
            num_hidden_features=64,
            num_hidden_layers=2,
            num_output_features=out.shape[2],
        )

        positional_embeddings = self.layer_dict[
            "positional_embedding_generator_network"
        ].forward(position_inputs)
        logging.debug(f"{positional_embeddings.shape}")

        positional_embeddings = rearrange(
            positional_embeddings,
            "(b p) d -> b p d",
            b=dummy_x.shape[0],
            d=out.shape[2],
        )
        logging.debug(f"{positional_embeddings.shape} {out.shape}")
        out = torch.cat([out, positional_embeddings], dim=2)

        self.layer_dict[
            "transformer_encoder_layer"
        ] = nn.TransformerEncoderLayer(
            d_model=self.transformer_num_filters,
            dim_feedforward=self.transformer_dim_feedforward,
            nhead=self.transformer_num_heads,
            activation=F.gelu,
            dropout=0.0,
            batch_first=True,
            norm_first=True,
        )
        self.layer_dict["transformer_encoder"] = nn.TransformerEncoder(
            encoder_layer=self.layer_dict["transformer_encoder_layer"],
            num_layers=self.transformer_num_layers,
        )

        out = self.layer_dict["transformer_encoder"].forward(out)

        self.is_built = True
        logging.debug(
            f"Build {self.__class__.__name__} with input shape {input_shape} with "
            f"output shape {out.shape} {positional_embeddings.shape}"
        )

    def forward(self, x):
        if not self.is_built:
            self.build(input_shape=x.shape)

        ratio = (x.shape[2]) / self.grid_patch_size

        if not ratio.is_integer():
            ceiling = int(np.ceil(ratio))
            new_h = ceiling * self.grid_patch_size
            new_w = ceiling * self.grid_patch_size
            x = F.interpolate(
                x,
                size=(new_h, new_w),
            )

        out = x

        b, c, h, w = out.shape

        out = rearrange(
            out,
            "b f (h h1) (w w1) -> (b h w) (h1 w1 f)",
            h1=self.grid_patch_size,
            w1=self.grid_patch_size,
        )

        num_patches = out.shape[0] / x.shape[0]

        out = self.layer_dict["stem_linear"].forward(out)
        # b, c, h, w

        out = self.layer_dict["stem_layer_normalization"].forward(out)

        out = rearrange(out, "(b s) (f) -> b s f", s=int(num_patches))

        position_inputs = repeat(
            self.enumerate_patches_idx, "p -> b p", b=x.shape[0]
        ).to(x.device)

        position_inputs = rearrange(position_inputs, "b (p d) -> (b p) d", d=1)

        positional_embeddings = self.layer_dict[
            "positional_embedding_generator_network"
        ].forward(position_inputs)

        positional_embeddings = rearrange(
            positional_embeddings,
            "(b p) d -> b p d",
            b=x.shape[0],
            d=out.shape[2],
        )
        out = torch.cat([out, positional_embeddings], dim=2)

        out = self.layer_dict["transformer_encoder"].forward(out)

        return out


class Conv1DTransformer(nn.Module):
    def __init__(
        self,
        grid_patch_size: int,
        transformer_num_filters: int,
        transformer_num_layers: int,
        transformer_num_heads: int,
        transformer_dim_feedforward: int,
        stem_conv_bias: False,
    ):
        super(Conv1DTransformer, self).__init__()
        self.grid_patch_size = grid_patch_size
        self.transformer_num_filters = transformer_num_filters
        self.transformer_num_layers = transformer_num_layers
        self.transformer_num_heads = transformer_num_heads
        self.transformer_dim_feedforward = transformer_dim_feedforward
        self.stem_conv_bias = stem_conv_bias

        self.is_built = False

    def build(self, input_shape):
        dummy_x = torch.zeros(input_shape)

        ratio = (dummy_x.shape[2]) / self.grid_patch_size

        if not ratio.is_integer():
            ceiling = int(np.ceil(ratio))
            new_h = ceiling * self.grid_patch_size
            dummy_x = F.interpolate(
                dummy_x,
                size=(new_h),
            )

        out = dummy_x

        self.layer_dict = nn.ModuleDict()

        out = rearrange(
            out, "b f (h h1) -> (b h) (h1 f)", h1=self.grid_patch_size
        )

        num_patches = out.shape[0] / dummy_x.shape[0]

        self.layer_dict["stem_linear"] = nn.Linear(
            in_features=out.shape[1],
            out_features=int(self.transformer_num_filters / 2),
            bias=True,
        )

        out = self.layer_dict["stem_linear"].forward(out)
        # b, c, h, w

        self.layer_dict["stem_layer_normalization"] = nn.LayerNorm(out.shape[1])

        out = self.layer_dict["stem_layer_normalization"].forward(out)

        out = rearrange(out, "(b s) (f) -> b s f", s=int(num_patches))

        self.enumerate_patches_idx = (
            torch.arange(start=0, end=num_patches) / num_patches
        )

        position_inputs = repeat(
            self.enumerate_patches_idx, "p -> b p", b=dummy_x.shape[0]
        )

        position_inputs = rearrange(position_inputs, "b (p d) -> (b p) d", d=1)

        self.layer_dict["positional_embedding_generator_network"] = FCCNetwork(
            num_hidden_features=64,
            num_hidden_layers=2,
            num_output_features=out.shape[2],
        )

        positional_embeddings = self.layer_dict[
            "positional_embedding_generator_network"
        ].forward(position_inputs)

        positional_embeddings = rearrange(
            positional_embeddings,
            "(b p) d -> b p d",
            b=dummy_x.shape[0],
            d=out.shape[2],
        )
        out = torch.cat([out, positional_embeddings], dim=2)

        self.layer_dict[
            "transformer_encoder_layer"
        ] = nn.TransformerEncoderLayer(
            d_model=self.transformer_num_filters,
            dim_feedforward=self.transformer_dim_feedforward,
            nhead=self.transformer_num_heads,
            activation=F.gelu,
            dropout=0.0,
            batch_first=True,
            norm_first=True,
        )
        self.layer_dict["transformer_encoder"] = nn.TransformerEncoder(
            encoder_layer=self.layer_dict["transformer_encoder_layer"],
            num_layers=self.transformer_num_layers,
        )

        out = self.layer_dict["transformer_encoder"].forward(out)

        self.is_built = True
        logging.debug(
            f"Build {self.__class__.__name__} with input shape {input_shape} with "
            f"output shape {out.shape}"
        )

    def forward(self, x):
        if not self.is_built:
            self.build(input_shape=x.shape)

        ratio = (x.shape[2]) / self.grid_patch_size

        if not ratio.is_integer():
            ceiling = int(np.ceil(ratio))
            new_h = ceiling * self.grid_patch_size
            x = F.interpolate(
                x,
                size=(new_h),
            )

        out = x

        out = rearrange(
            out, "b f (h h1) -> (b h) (h1 f)", h1=self.grid_patch_size
        )

        num_patches = out.shape[0] / x.shape[0]

        out = self.layer_dict["stem_linear"].forward(out)
        # b, c, h, w

        out = self.layer_dict["stem_layer_normalization"].forward(out)

        out = rearrange(out, "(b s) (f) -> b s f", s=int(num_patches))

        position_inputs = repeat(
            self.enumerate_patches_idx, "p -> b p", b=x.shape[0]
        ).to(x.device)

        position_inputs = rearrange(position_inputs, "b (p d) -> (b p) d", d=1)

        positional_embeddings = F.leaky_relu(
            self.layer_dict["positional_embedding_generator_network"].forward(
                position_inputs
            )
        )

        positional_embeddings = rearrange(
            positional_embeddings,
            "(b p) d -> b p d",
            b=x.shape[0],
            d=out.shape[2],
        )
        out = torch.cat([out, positional_embeddings], dim=2)

        out = self.layer_dict["transformer_encoder"].forward(out)

        return out


class TexTransformer(nn.Module):
    def __init__(
        self,
        transformer_num_filters: int,
        transformer_num_layers: int,
        transformer_num_heads: int,
        transformer_dim_feedforward: int,
        vocab_size: int,
        context_length: int,
    ):
        super(TexTransformer, self).__init__()
        self.transformer_num_filters = transformer_num_filters
        self.transformer_num_layers = transformer_num_layers
        self.transformer_num_heads = transformer_num_heads
        self.transformer_dim_feedforward = transformer_dim_feedforward
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.layer_dict = nn.ModuleDict()
        self.layer_params = nn.ParameterDict()
        self.is_built = False

    def build(self, input_shape):
        dummy_x = torch.zeros(input_shape).long()

        out = dummy_x

        self.layer_dict["token_embedding"] = nn.Embedding(
            self.vocab_size, self.transformer_num_filters
        )

        out = self.layer_dict["token_embedding"].forward(out)

        # b, l, c

        self.enumerate_patches_idx = (
            torch.arange(start=0, end=out.shape[1]) / out.shape[1]
        )

        position_inputs = repeat(
            self.enumerate_patches_idx, "p -> b p", b=dummy_x.shape[0]
        )

        position_inputs = rearrange(position_inputs, "b (p d) -> (b p) d", d=1)

        self.layer_dict["positional_embedding_generator_network"] = FCCNetwork(
            num_hidden_features=64,
            num_hidden_layers=2,
            num_output_features=out.shape[2],
        )

        positional_embeddings = self.layer_dict[
            "positional_embedding_generator_network"
        ].forward(position_inputs)

        positional_embeddings = rearrange(
            positional_embeddings,
            "(b p) d -> b p d",
            b=dummy_x.shape[0],
            d=out.shape[2],
        )
        out = torch.cat([out, positional_embeddings], dim=2)

        self.layer_dict[
            "transformer_encoder_layer"
        ] = nn.TransformerEncoderLayer(
            d_model=self.transformer_num_filters * 2,
            dim_feedforward=self.transformer_dim_feedforward,
            nhead=self.transformer_num_heads,
            batch_first=True,
        )
        self.layer_dict["transformer_encoder"] = nn.TransformerEncoder(
            encoder_layer=self.layer_dict["transformer_encoder_layer"],
            num_layers=self.transformer_num_layers,
        )

        out = self.layer_dict["transformer_encoder"].forward(out)

        self.is_built = True

        logging.debug(
            f"Build {self.__class__.__name__} with input shape {input_shape} with "
            f"output shape {out.shape}"
        )

    def forward(self, x):
        if not self.is_built:
            self.build(input_shape=x.shape)

        out = x.long()

        out = self.layer_dict["token_embedding"].forward(out)

        position_inputs = repeat(
            self.enumerate_patches_idx, "p -> b p", b=x.shape[0]
        ).to(x.device)

        position_inputs = rearrange(position_inputs, "b (p d) -> (b p) d", d=1)

        positional_embeddings = F.leaky_relu(
            self.layer_dict["positional_embedding_generator_network"].forward(
                position_inputs
            )
        )

        positional_embeddings = rearrange(
            positional_embeddings,
            "(b p) d -> b p d",
            b=x.shape[0],
            d=out.shape[2],
        )
        out = torch.cat([out, positional_embeddings], dim=2)

        out = self.layer_dict["transformer_encoder"].forward(out)

        return out


class VideoTransformer(nn.Module):
    def __init__(
        self,
        transformer_num_filters: int,
        transformer_num_layers: int,
        transformer_num_heads: int,
        transformer_dim_feedforward: int,
        image_embedding: nn.Module,
    ):
        super(VideoTransformer, self).__init__()
        self.transformer_num_filters = transformer_num_filters
        self.transformer_num_layers = transformer_num_layers
        self.transformer_num_heads = transformer_num_heads
        self.transformer_dim_feedforward = transformer_dim_feedforward

        self.image_embedding = image_embedding
        self.layer_dict = nn.ModuleDict()
        self.layer_params = nn.ParameterDict()
        self.is_built = False

    def build(self, input_shape):
        dummy_x = torch.zeros(input_shape)

        out = dummy_x.view(
            -1, dummy_x.shape[-3], dummy_x.shape[-2], dummy_x.shape[-1]
        )

        out, _ = self.image_embedding(out)

        out = out.view(dummy_x.shape[0], dummy_x.shape[1], -1)

        self.enumerate_patches_idx = (
            torch.arange(start=0, end=out.shape[1]) / out.shape[1]
        )

        position_inputs = repeat(
            self.enumerate_patches_idx, "p -> b p", b=dummy_x.shape[0]
        )

        position_inputs = rearrange(position_inputs, "b (p d) -> (b p) d", d=1)

        self.layer_dict["positional_embedding_generator_network"] = FCCNetwork(
            num_hidden_features=64,
            num_hidden_layers=2,
            num_output_features=out.shape[2],
        )

        positional_embeddings = self.layer_dict[
            "positional_embedding_generator_network"
        ].forward(position_inputs)

        positional_embeddings = rearrange(
            positional_embeddings,
            "(b p) d -> b p d",
            b=dummy_x.shape[0],
            d=out.shape[2],
        )
        out = out + positional_embeddings

        self.layer_dict[
            "transformer_encoder_layer"
        ] = nn.TransformerEncoderLayer(
            d_model=self.transformer_num_filters,
            dim_feedforward=self.transformer_dim_feedforward,
            nhead=self.transformer_num_heads,
            batch_first=True,
        )
        self.layer_dict["transformer_encoder"] = nn.TransformerEncoder(
            encoder_layer=self.layer_dict["transformer_encoder_layer"],
            num_layers=self.transformer_num_layers,
        )

        out = self.layer_dict["transformer_encoder"].forward(out)

        self.is_built = True

        logging.debug(
            f"Build {self.__class__.__name__} with input shape {input_shape} with "
            f"output shape {out.shape}"
        )

    def forward(self, x):
        if not self.is_built:
            self.build(input_shape=x.shape)

        out = x.view(-1, x.shape[-3], x.shape[-2], x.shape[-1])

        out, _ = self.image_embedding(out)

        out = out.view(x.shape[0], x.shape[1], -1)

        position_inputs = repeat(
            self.enumerate_patches_idx, "p -> b p", b=x.shape[0]
        ).to(x.device)

        position_inputs = rearrange(position_inputs, "b (p d) -> (b p) d", d=1)

        positional_embeddings = F.leaky_relu(
            self.layer_dict["positional_embedding_generator_network"].forward(
                position_inputs
            )
        )

        positional_embeddings = rearrange(
            positional_embeddings,
            "(b p) d -> b p d",
            b=x.shape[0],
            d=out.shape[2],
        )
        out = out + positional_embeddings

        out = self.layer_dict["transformer_encoder"].forward(out)
        return out


class VisionTransformer(nn.Module):
    def __init__(
        self,
        patch_size: int,
        width: int,
        layers: int,
        heads: int,
        output_dim: int,
    ):
        super().__init__()
        self.output_dim = output_dim
        self.width = width
        self.patch_size = patch_size
        self.layers = layers
        self.heads = heads
        self.is_built = False

    def build(self, input_shape):
        self.input_resolution = input_shape[-1]
        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=self.width,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            bias=False,
        )

        scale = self.width**-0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(self.width))
        self.positional_embedding = nn.Parameter(
            scale
            * torch.randn(
                (self.input_resolution // self.patch_size) ** 2 + 1, self.width
            )
        )
        self.ln_pre = LayerNorm(self.width)

        self.transformer = Transformer(self.width, self.layers, self.heads)

        self.is_built = True

    def forward(self, x: torch.Tensor):
        if not self.is_built:
            self.build(x.shape)

        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(
            x.shape[0], x.shape[1], -1
        )  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat(
            [
                self.class_embedding.to(x.dtype)
                + torch.zeros(
                    x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device
                ),
                x,
            ],
            dim=1,
        )  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        return x


class AutoConv2DTransformersFlatten(ClassificationModel):
    def __init__(
        self,
        num_classes,
        grid_patch_size: int,
        transformer_num_filters: int,
        transformer_num_layers: int,
        transformer_num_heads: int,
        transformer_dim_feedforward: int,
        stem_conv_bias=False,
        **kwargs,
    ):
        feature_embedding_modules = [
            nn.InstanceNorm2d,
            Conv2DTransformer,
            SqueezeExciteConv1dBNLeakyReLU,
            nn.Flatten,
        ]

        feature_embeddings_args = [
            dict(num_features=3, affine=True, track_running_stats=True),
            dict(
                grid_patch_size=grid_patch_size,
                transformer_num_filters=transformer_num_filters,
                transformer_num_layers=transformer_num_layers,
                transformer_num_heads=transformer_num_heads,
                transformer_dim_feedforward=transformer_dim_feedforward,
                stem_conv_bias=stem_conv_bias,
                # patch_size=grid_patch_size,
                # width=transformer_num_filters,
                # layers=transformer_num_layers,
                # heads=transformer_num_heads,
                # output_dim=num_classes,
            ),
            dict(
                out_channels=1,
                kernel_size=1,
                stride=1,
                padding=0,
                dilation=1,
                bias=False,
            ),
            dict(start_dim=1, end_dim=-1),
        ]
        super(AutoConv2DTransformersFlatten, self).__init__(
            num_classes=num_classes,
            feature_embedding_module_list=feature_embedding_modules,
            feature_embedding_args=feature_embeddings_args,
        )


class AutoConv1DTransformersFlatten(ClassificationModel):
    def __init__(
        self,
        num_classes,
        grid_patch_size: int,
        transformer_num_filters: int,
        transformer_num_layers: int,
        transformer_num_heads: int,
        transformer_dim_feedforward: int,
        stem_conv_bias=False,
        **kwargs,
    ):
        feature_embedding_modules = [
            nn.InstanceNorm1d,
            Conv1DTransformer,
            nn.Flatten,
        ]
        feature_embeddings_args = [
            dict(num_features=2, affine=True, track_running_stats=True),
            dict(
                grid_patch_size=grid_patch_size,
                transformer_num_filters=transformer_num_filters,
                transformer_num_layers=transformer_num_layers,
                transformer_num_heads=transformer_num_heads,
                transformer_dim_feedforward=transformer_dim_feedforward,
                stem_conv_bias=stem_conv_bias,
            ),
            dict(),
        ]
        super(AutoConv1DTransformersFlatten, self).__init__(
            num_classes=num_classes,
            feature_embedding_module_list=feature_embedding_modules,
            feature_embedding_args=feature_embeddings_args,
        )


class AutoTextTransformersFlatten(ClassificationModel):
    def __init__(
        self,
        transformer_num_filters: int,
        transformer_num_layers: int,
        transformer_num_heads: int,
        transformer_dim_feedforward: int,
        vocab_size: int,
        context_length: int,
        num_classes: int,
        **kwargs,
    ):
        feature_embedding_modules = [
            TexTransformer,
            SqueezeExciteConv1dBNLeakyReLU,
            nn.Flatten,
        ]

        feature_embeddings_args = [
            dict(
                transformer_num_filters=transformer_num_filters,
                transformer_num_layers=transformer_num_layers,
                transformer_num_heads=transformer_num_heads,
                transformer_dim_feedforward=transformer_dim_feedforward,
                vocab_size=vocab_size,
                context_length=context_length,
            ),
            dict(
                out_channels=1,
                kernel_size=1,
                stride=1,
                padding=0,
                dilation=1,
                bias=False,
            ),
            dict(start_dim=1, end_dim=-1),
        ]
        super(AutoTextTransformersFlatten, self).__init__(
            num_classes=num_classes,
            feature_embedding_module_list=feature_embedding_modules,
            feature_embedding_args=feature_embeddings_args,
        )


class AutoVideoTransformersFlatten(ClassificationModel):
    def __init__(
        self,
        transformer_num_filters: int,
        transformer_num_layers: int,
        transformer_num_heads: int,
        transformer_dim_feedforward: int,
        num_classes: int,
        image_embedding: nn.Module,
        **kwargs,
    ):
        feature_embedding_modules = [
            VideoTransformer,
            nn.Flatten,
        ]

        feature_embeddings_args = [
            dict(
                transformer_num_filters=transformer_num_filters,
                transformer_num_layers=transformer_num_layers,
                transformer_num_heads=transformer_num_heads,
                transformer_dim_feedforward=transformer_dim_feedforward,
                image_embedding=image_embedding,
            ),
            dict(),
        ]
        super(AutoVideoTransformersFlatten, self).__init__(
            num_classes=num_classes,
            feature_embedding_module_list=feature_embedding_modules,
            feature_embedding_args=feature_embeddings_args,
        )
