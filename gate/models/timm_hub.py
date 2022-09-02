from typing import Callable, List

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from dotted_dict import DottedDict
from omegaconf import DictConfig

from gate.base.utils.loggers import get_logger
from gate.configs.datamodule.base import ShapeConfig
from gate.model_blocks.auto_builder_modules.gcm_blocks import HeadResNetBlock
from gate.models.base import ModelModule

log = get_logger()


def pop_layers_with_with_prefix_terms(model: nn.Module, term_list: List[str]):
    for name, module in model.named_children():
        for term in term_list:
            if name.startswith(term):
                print(f"Removing layer {name}")
                model.add_module(name, nn.Identity())
                break

    return model


class TimmImageModel(ModelModule):
    def __init__(
        self,
        input_shape_dict: DictConfig = None,
        model_name_to_download: str = "resnet18",
        pretrained: bool = True,
        global_pool: bool = True,
    ):
        """ResNet model for image classification.

        Parameters
        ----------
        model_name_to_download : str
            Name of the model to download. List of possible names:
            a
        pretrained : bool
            Whether to load the pretrained weights.
        audio_kernel_size : int
            Kernel size of the audio convolution.
        input_shape_dict : dict
            Shape configuration of the input modality.

        """
        self.image_shape = None
        self.resnet_image_embedding = None
        log.info(
            f"Init {self.__class__.__name__} with {model_name_to_download}, "
            f"{pretrained}"
        )

        super(TimmImageModel, self).__init__(input_shape_dict)
        self.is_built = False
        self.model_name_to_download = model_name_to_download
        self.pretrained = pretrained
        self.global_pool = global_pool

    def build(self, batch_dict):

        if isinstance(self.input_shape_dict, ShapeConfig):
            self.input_shape_dict = self.input_shape_dict.__dict__

        if "image" in self.input_shape_dict:
            self.build_image(self.input_shape_dict)

        self.is_built = True

        log.info(f"{self.__class__.__name__} built")

    def build_image(self, input_shape):

        self.image_shape = list(
            self.input_shape_dict["image"]["shape"].values()
        )

        image_input_dummy = torch.zeros([2] + self.image_shape)

        self.resnet_image_embedding = timm.create_model(
            self.model_name_to_download, pretrained=self.pretrained
        )
        log.info(self.resnet_image_embedding)
        self.resnet_image_embedding.fc = nn.Identity()  # remove logit layer

        self.resnet_image_embedding.global_pool = (
            nn.Identity()
        )  # remove global pool

        if image_input_dummy.shape[1:] != self.image_shape:
            image_input_dummy = F.interpolate(
                image_input_dummy,
                size=self.image_shape[1:],
            )

        out = self.resnet_image_embedding(image_input_dummy)

        if self.global_pool:
            out = F.adaptive_avg_pool2d(out, 1).squeeze(-1).squeeze(-1)

        log.info(
            f"Built image processing network of {self.__class__.__name__} "
            f"image model with output shape {out.shape}"
        )

    def forward_image(self, x_image):
        # expects b, c, w, h input_shape
        if x_image.shape[1:] != self.image_shape:
            x_image = F.interpolate(x_image, size=self.image_shape[1:])

        if len(x_image.shape) != 4:
            raise ValueError(
                f"Input shape for class {self.__class__.__name__} in "
                f"method forward_image must be 4, instead it is "
                f"{len(x_image.shape)}, for shape {x_image.shape}"
            )

        out = self.resnet_image_embedding(x_image)
        if self.global_pool:
            out = F.adaptive_avg_pool2d(out, 1).squeeze(-1).squeeze(-1)

        return out

    def forward(self, x):
        if not self.is_built:
            self.build(x)

        output_dict = {}

        if "image" in x:
            output_dict["image"] = self.forward_image(x["image"])

        return output_dict


class TimmImageModelConfigurableDepth(TimmImageModel):
    def __init__(
        self,
        input_shape_dict: DictConfig = None,
        model_name_to_download: str = "resnet18",
        pretrained: bool = True,
        global_pool: bool = True,
        list_of_layer_prefix_to_remove: List[str] = None,
    ):
        """ResNet model for image classification.

        Parameters
        ----------
        model_name_to_download : str
            Name of the model to download. List of possible names:
            a
        pretrained : bool
            Whether to load the pretrained weights.
        audio_kernel_size : int
            Kernel size of the audio convolution.
        input_shape_dict : dict
            Shape configuration of the input modality.

        """
        super(TimmImageModelConfigurableDepth, self).__init__(
            input_shape_dict=input_shape_dict,
            model_name_to_download=model_name_to_download,
            pretrained=pretrained,
            global_pool=global_pool,
        )
        self.list_of_layer_prefix_to_remove = list_of_layer_prefix_to_remove

    def build(self, batch_dict):
        if "image" in batch_dict:
            self.build_image(batch_dict["image"].shape)

        self.is_built = True

        log.info(f"{self.__class__.__name__} built")

    def build_image(self, input_shape):
        image_input_dummy = torch.zeros(input_shape)
        if isinstance(self.input_shape_dict, ShapeConfig):
            self.input_shape_dict = self.input_shape_dict.__dict__

        self.image_shape = list(
            self.input_shape_dict["image"]["shape"].values()
        )
        self.resnet_image_embedding = timm.create_model(
            self.model_name_to_download, pretrained=self.pretrained
        )
        self.resnet_image_embedding = pop_layers_with_with_prefix_terms(
            self.resnet_image_embedding, self.list_of_layer_prefix_to_remove
        )

        log.info(self.resnet_image_embedding)

        if image_input_dummy.shape[1:] != self.image_shape:
            image_input_dummy = F.interpolate(
                image_input_dummy,
                size=self.image_shape[1:],
            )

        out = self.resnet_image_embedding(image_input_dummy)

        if self.global_pool:
            out = F.adaptive_avg_pool2d(out, 1).squeeze(-1).squeeze(-1)

        log.info(
            f"Built image processing network of {self.__class__.__name__} "
            f"image model with output shape {out.shape}"
        )


class TimmImageModelConfigurableDepthAndHead(TimmImageModelConfigurableDepth):
    def __init__(
        self,
        head_num_output_filters: int,
        head_num_hidden_filters: int,
        input_shape_dict: DictConfig = None,
        model_name_to_download: str = "resnet18",
        pretrained: bool = True,
        list_of_layer_prefix_to_remove: List[str] = None,
        global_pool: bool = True,
        head_output_activation_fn: Callable = None,
        use_twin_head: bool = False,
    ):
        """ResNet model for image classification.

        Parameters
        ----------
        model_name_to_download : str
            Name of the model to download. List of possible names:
            a
        pretrained : bool
            Whether to load the pretrained weights.
        audio_kernel_size : int
            Kernel size of the audio convolution.
        input_shape_dict : dict
            Shape configuration of the input modality.

        """
        super(TimmImageModelConfigurableDepthAndHead, self).__init__(
            input_shape_dict=input_shape_dict,
            model_name_to_download=model_name_to_download,
            pretrained=pretrained,
            list_of_layer_prefix_to_remove=list_of_layer_prefix_to_remove,
            global_pool=global_pool,
        )
        self.head_num_output_filters = head_num_output_filters
        self.head_num_hidden_filters = head_num_hidden_filters
        self.head_output_activation_fn = head_output_activation_fn
        self.use_twin_head = use_twin_head

    def build_image(self, input_shape):
        image_input_dummy = torch.zeros(input_shape)
        if isinstance(self.input_shape_dict, ShapeConfig):
            self.input_shape_dict = self.input_shape_dict.__dict__

        self.image_shape = list(
            self.input_shape_dict["image"]["shape"].values()
        )
        self.resnet_image_embedding = timm.create_model(
            self.model_name_to_download, pretrained=self.pretrained
        )
        self.resnet_image_embedding = pop_layers_with_with_prefix_terms(
            self.resnet_image_embedding, self.list_of_layer_prefix_to_remove
        )

        log.info(self.resnet_image_embedding)

        if image_input_dummy.shape[1:] != self.image_shape:
            image_input_dummy = F.interpolate(
                image_input_dummy,
                size=self.image_shape[1:],
            )

        out = self.resnet_image_embedding(image_input_dummy)

        self.resnet_head_embedding = HeadResNetBlock(
            num_output_filters=self.head_num_output_filters,
            num_hidden_filters=self.head_num_hidden_filters,
            output_activation_fn=self.head_output_activation_fn,
        )
        self.resnet_head_embedding.build(out.shape)

        if self.use_twin_head:
            self.resnet_head_embedding_auxiliary = HeadResNetBlock(
                num_output_filters=self.head_num_output_filters,
                num_hidden_filters=self.head_num_hidden_filters,
                output_activation_fn=self.head_output_activation_fn,
            )
            self.resnet_head_embedding_auxiliary.build(out.shape)

            out = (
                self.resnet_head_embedding({"image": out})["image"]
                * self.resnet_head_embedding_auxiliary({"image": out})["image"]
            )

        else:
            out = self.resnet_head_embedding({"image": out})["image"]

        if self.global_pool:
            out = F.adaptive_avg_pool2d(out, 1).squeeze(-1).squeeze(-1)

        log.info(
            f"Built image processing network of {self.__class__.__name__} "
            f"image model with output shape {out.shape}"
        )

    def forward_image(self, x_image):
        # expects b, c, w, h input_shape
        if x_image.shape[1:] != self.image_shape:
            x_image = F.interpolate(x_image, size=self.image_shape[1:])

        if len(x_image.shape) != 4:
            raise ValueError(
                f"Input shape for class {self.__class__.__name__} in "
                f"method forward_image must be 4, instead it is "
                f"{len(x_image.shape)}, for shape {x_image.shape}"
            )

        out = self.resnet_image_embedding(x_image)

        if self.use_twin_head:
            out = (
                self.resnet_head_embedding({"image": out})["image"]
                * self.resnet_head_embedding_auxiliary({"image": out})["image"]
            )
        else:
            out = self.resnet_head_embedding({"image": out})["image"]

        if self.global_pool:
            out = F.adaptive_avg_pool2d(out, 1).squeeze(-1).squeeze(-1)

        return out

    def forward(self, x):
        if not self.is_built:
            self.build(x)

        output_dict = {}

        if "image" in x:
            output_dict["image"] = self.forward_image(x["image"])

        return output_dict
