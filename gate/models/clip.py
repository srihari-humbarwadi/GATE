import os
from typing import Union

import clip
import torch
from dotted_dict import DottedDict
from torchvision.transforms.functional import normalize

from gate.base.utils.loggers import get_logger
from gate.base.utils.model_utils import resize_custom
from gate.configs.datamodule.base import ShapeConfig
from gate.models.base import ModelModule

log = get_logger(set_default_handler=False)


class CLIP(ModelModule):
    def __init__(
        self,
        input_shape_dict: Union[ShapeConfig, DottedDict],
        model_root_dir: str = None,
        model_name_to_download: str = "resnet18",
        pretrained: bool = True,
        device: str = "cpu",
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
        self.text_shape = None
        self.input_transforms = None
        self.model = None
        self.image_shape = None
        self.resnet_image_embedding = None
        self.model_name_to_download = model_name_to_download

        log.info(
            f"Init {self.__class__.__name__} with {model_name_to_download}, "
            f"{pretrained}"
        )

        self.model_input_description_dict = {
            "RN50": dict(
                input_shape=(3, 224, 224), input_type=torch.FloatTensor
            ),
            "RN101": dict(
                input_shape=(3, 224, 224), input_type=torch.FloatTensor
            ),
            "RN50x4": dict(
                input_shape=(3, 288, 288), input_type=torch.FloatTensor
            ),
            "RN50x16": dict(
                input_shape=(3, 384, 384), input_type=torch.FloatTensor
            ),
            "RN50x64": dict(
                input_shape=(3, 448, 448), input_type=torch.FloatTensor
            ),
            "ViT-B/32": dict(
                input_shape=(3, 224, 224), input_type=torch.HalfTensor
            ),
            "ViT-B/16": dict(
                input_shape=(3, 224, 224), input_type=torch.HalfTensor
            ),
            "ViT-L/14": dict(
                input_shape=(3, 224, 224), input_type=torch.HalfTensor
            ),
        }[self.model_name_to_download]

        self.image_shape = self.model_input_description_dict["input_shape"]

        model, input_transforms = clip.load(
            name=model_name_to_download,
            download_root=os.environ["MODEL_DIR"]
            if model_root_dir is None
            else model_root_dir,
            device=device,
        )

        super(CLIP, self).__init__(
            input_shape_dict=input_shape_dict,
        )

        if not pretrained:
            model.model = model.initialize_parameters()

        self.model = model
        self.input_transforms = input_transforms
        self.device = device

        self.input_type = self.model_input_description_dict["input_type"]
        self.is_built = False
        self.model_name_to_download = model_name_to_download
        self.model_root_dir = model_root_dir
        self.pretrained = pretrained
        self.name = self.__class__.__name__

    def build(self, batch_dict):

        if "image" in batch_dict:
            self.build_image(batch_dict["image"].shape)

        if "text" in batch_dict:
            self.build_text(batch_dict["text"].shape)

        self.is_built = True

        log.info(f"{self.__class__.__name__} built")

    def build_image(self, input_shape):
        image_input_dummy = torch.zeros(input_shape)

        if isinstance(self.input_shape_dict, ShapeConfig):
            self.input_shape_dict = self.input_shape_dict.__dict__

        if image_input_dummy.shape[1:] != self.image_shape:
            image_input_dummy = resize_custom(
                image_input_dummy, target_image_shape=self.image_shape
            )

        out = self.model.encode_image(image_input_dummy.to(self.device))
        self.mean = self.input_transforms.transforms[-1].mean
        self.std = self.input_transforms.transforms[-1].mean
        log.info(self.input_transforms)

        output_shape = out.shape

        log.info(
            f"Built image processing network of {self.__class__.__name__} "
            f"image model with output shape {output_shape}"
        )

    def build_text(self, input_shape):
        text_input_dummy = torch.zeros(input_shape).long()

        if isinstance(self.input_shape_dict, ShapeConfig):
            self.input_shape_dict = self.input_shape_dict.__dict__

        self.text_shape = list(self.input_shape_dict["text"]["shape"])

        out = self.model.encode_text(text_input_dummy.to(self.device))

        output_shape = out.shape

        log.info(
            f"Built text processing network of {self.__class__.__name__} "
            f"text model with output shape {output_shape}, {self.input_transforms}"
        )

    def forward_image(self, x_image):
        # expects b, c, w, h input_shape
        # print("Pre", x_image.shape)
        if x_image.shape[1:] != self.image_shape:
            x_image = resize_custom(
                x_image, target_image_shape=self.image_shape
            )
        # print("Post", x_image.shape)
        x_image = normalize(x_image, mean=self.mean, std=self.std)

        if len(x_image.shape) != 4:
            raise ValueError(
                f"Input shape for class {self.__class__.__name__} in "
                f"method forward_image must be 4, instead it is "
                f"{len(x_image.shape)}, for shape {x_image.shape}"
            )

        return self.model.encode_image(x_image)

    def forward_text(self, x_text):

        if len(x_text.shape) != 2:
            raise ValueError(
                f"Input shape for class {self.__class__.__name__} in "
                f"method forward_image must be 2, instead it is "
                f"{len(x_text.shape)}, for shape {x_text.shape}"
            )
        log.info(
            f"Forward text with shape {x_text.shape} {self.model.positional_embedding.shape}"
        )
        return self.model.encode_text(x_text)

    def forward(self, x):
        if not self.is_built:
            self.build(x)

        output_dict = {}

        if "image" in x:
            output_dict["image"] = self.forward_image(x["image"])

        if "text" in x:
            output_dict["text"] = self.forward_text(x["text"])

        return output_dict
