import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

from gate.base.utils.loggers import get_logger
from gate.class_configs.base import ShapeConfig

from gate.models.base import ModelModule

log = get_logger()


class ImageResNet(ModelModule):
    def __init__(
        self,
        input_shape_dict: ShapeConfig = None,
        model_name_to_download: str = "resnet18",
        pretrained: bool = True,
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
        super(ImageResNet, self).__init__(input_shape_dict)
        self.is_built = False
        self.model_name_to_download = model_name_to_download
        self.pretrained = pretrained
        self.name = self.__class__.__name__

    def build(self, batch_dict):
        if "image" in batch_dict:
            self.build_image(batch_dict["image"].shape)

        self.is_built = True

        log.info(f"{self.__class__.__name__} built")

    def build_image(self, input_shape):
        image_input_dummy = torch.zeros(input_shape)
        if isinstance(self.input_shape_dict, ShapeConfig):
            self.input_shape_dict = self.input_shape_dict.__dict__

        self.image_shape = list(self.input_shape_dict["image"]["shape"].values())
        self.resnet_image_embedding = timm.create_model(
            self.model_name_to_download, pretrained=self.pretrained
        )
        log.info(self.resnet_image_embedding)
        self.resnet_image_embedding.fc = nn.Identity()  # remove logit layer

        self.resnet_image_embedding.global_pool = nn.Identity()  # remove global pool

        if image_input_dummy.shape[1:] != self.image_shape:
            image_input_dummy = F.interpolate(
                image_input_dummy,
                size=self.image_shape[1:],
            )

        output_shape = self.resnet_image_embedding(image_input_dummy).shape

        log.info(
            f"Built image processing network of {self.__class__.__name__} "
            f"image model with output shape {output_shape}"
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

        return self.resnet_image_embedding(x_image)

    def forward(self, x):
        if not self.is_built:
            self.build(x)

        output_dict = {}

        if "image" in x:
            output_dict["image"] = self.forward_image(x["image"])

        return output_dict
