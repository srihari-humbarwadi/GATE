import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from gate.base import utils
from gate.config import ShapeConfig

log = utils.get_logger()


class MissingModalityForward(Exception):
    pass


def generic_missing_forward(module, modality_name):
    raise MissingModalityForward(
        f"forward_{modality_name} method not implemented in model: "
        f"{module.__class__.__name__} "
    )


class ModelModule(nn.Module):
    def __init__(self, input_modality_shape_config: ShapeConfig):
        super().__init__()
        self.input_modality_shape_config = input_modality_shape_config


class AudioImageResNet(ModelModule):
    def __init__(
        self,
        model_name_to_download,
        pretrained,
        audio_kernel_size,
        input_modality_shape_config,
    ):
        logging.info(
            f"Init {self.__class__.__name__} with {model_name_to_download}, "
            f"{pretrained}"
        )
        super(AudioImageResNet, self).__init__(input_modality_shape_config)
        self.is_built = False
        self.model_name_to_download = model_name_to_download
        self.pretrained = pretrained
        self.audio_kernel_size = audio_kernel_size
        self.input_modality_shape_config = input_modality_shape_config

    def build(self, batch_dict):
        if "image" in batch_dict:
            self.build_image(batch_dict["image"].shape)

        if "audio" in batch_dict:
            self.build_audio(batch_dict["audio"].shape)

        self.is_built = True

        log.info(f"{self.__class__.__name__} built")

    def build_image(self, input_shape):
        image_input_dummy = torch.zeros(input_shape)
        self.resnet_image_embedding = torch.hub.load(
            "pytorch/vision",
            self.model_name_to_download,
            pretrained=self.pretrained,
        )

        self.resnet_image_embedding.fc = nn.Identity()  # remove logit layer

        self.resnet_image_embedding.avgpool = nn.Identity()  # remove global pool

        output_shape = self.resnet_image_embedding(image_input_dummy).shape

        log.info(
            f"Built image processing network of {self.__class__.__name__} "
            f"image model with output shape {output_shape}"
        )

    def build_audio(self, input_shape):
        audio_input_dummy = torch.zeros(input_shape)

        self.resnet_audio_to_image_conv = nn.Conv1d(
            in_channels=2,
            out_channels=3,
            bias=True,
            kernel_size=self.audio_kernel_size,
        )

        if not hasattr(self, "resnet_image_embedding"):
            self.resnet_image_embedding = torch.hub.load(
                "pytorch/vision",
                self.model_name_to_download,
                pretrained=self.pretrained,
            )

            self.resnet_image_embedding.fc = nn.Identity()  # remove logit layer

            self.resnet_image_embedding.avgpool = nn.Identity()  # remove global pool

        audio_embedding = self.resnet_audio_to_image_conv.forward(audio_input_dummy)

        output = self.resnet_image_embedding(audio_embedding)

        log.info(
            f"Built audio processing network of {self.__class__.__name__} "
            f"image model with output shape {output.shape}"
        )

    def forward_image(self, x_image):
        # expects b, c, w, h input_shape

        if len(x_image.shape) != 4:
            raise ValueError(
                f"Input shape for class {self.__class__.__name__} in "
                f"method forward_image must be 4, instead it is "
                f"{len(x_image.shape)}, for shape {x_image.shape}"
            )

        return self.resnet_image_embedding(x_image)

    def forward_audio(self, x_audio):  # sourcery skip: square-identity
        # expects b, c, sequence_length input_shape

        if len(x_audio.shape) != 3:
            return ValueError(
                f"Input shape for class {self.__class__.__name__} in "
                f"method forward_image must be 3, instead it is "
                f"{len(x_audio.shape)}, for shape {x_audio.shape}"
            )
        out = self.resnet_audio_to_image_conv.forward(x_audio)
        out = F.adaptive_avg_pool1d(out, output_size=224 * 224)
        out = out.view(x_audio.shape[0], 3, 224, 224)
        out = self.resnet_image_embedding.forward(out)

        return out

    def forward_audio_image(self, x_audio, x_image):

        image_embedding = self.forward_image(x_image).view(x_image.shape[0], -1)
        audio_embedding = self.forward_audio(x_audio).view(x_audio.shape[0], -1)

        image_audio_embedding = torch.cat((image_embedding, audio_embedding), 1)
        return self.fusion_layer(image_audio_embedding)

    def forward(self, x):
        if not self.is_built:
            self.build(x)

        output_dict = {}

        if "audio" in x and "image" in x:
            output_dict["image_audio"] = self.forward_audio_image(
                x_image=x["image"], x_audio=x["audio"]
            )

        else:
            if "image" in x:
                output_dict["image"] = self.forward_image(x["image"])

            if "audio" in x:
                output_dict["audio"] = self.forward_audio(x["audio"])

        return output_dict
