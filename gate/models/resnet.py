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
        input_modality_shape_config: ShapeConfig = None,
        model_name_to_download: str = "resnet18",
        pretrained: bool = True,
    ):
        """ResNet model for image classification.

        Parameters
        ----------
        model_name_to_download : str
            Name of the model to download. List of possible names:
            alexnet
            convnext_tiny, convnext_small, convnext_base, convnext_large
            densenet121, densenet169, densenet201, densenet161

            efficientnet_b0,
            efficientnet_b1,
            efficientnet_b2,
            efficientnet_b3,
            efficientnet_b4,
            efficientnet_b5,
            efficientnet_b6,
            efficientnet_b7,
            efficientnet_v2_s,
            efficientnet_v2_m,
            efficientnet_v2_l,

            googlenet
            inception_v3
            mnasnet0_5, mnasnet0_75, mnasnet1_0, mnasnet1_3
            mobilenet_v2
            mobilenet_v3_large, mobilenet_v3_small
            raft_large, raft_small

            regnet_y_400mf,
            regnet_y_800mf,
            regnet_y_1_6gf,
            regnet_y_3_2gf,
            regnet_y_8gf,
            regnet_y_16gf,
            regnet_y_32gf,
            regnet_y_128gf,
            regnet_x_400mf,
            regnet_x_800mf,
            regnet_x_1_6gf,
            regnet_x_3_2gf,
            regnet_x_8gf,
            regnet_x_16gf,
            regnet_x_32gf,



            resnet18,
            resnet34,
            resnet50,
            resnet101,
            resnet152,
            resnext50_32x4d,
            resnext101_32x8d,
            wide_resnet50_2,
            wide_resnet101_2,

            fcn_resnet50,
            fcn_resnet101,
            deeplabv3_resnet50,
            deeplabv3_resnet101,
            deeplabv3_mobilenet_v3_large,
            lraspp_mobilenet_v3_large,

            shufflenet_v2_x0_5, shufflenet_v2_x1_0
            squeezenet1_0, squeezenet1_1
            vgg11, vgg13, vgg16, vgg19, vgg11_bn, vgg13_bn, vgg16_bn, vgg19_bn

            vit_b_16,
            vit_b_32,
            vit_l_16,
            vit_l_32,
        pretrained : bool
            Whether to load the pretrained weights.
        audio_kernel_size : int
            Kernel size of the audio convolution.
        input_modality_shape_config : dict
            Shape configuration of the input modality.

        """
        self.resnet_image_embedding = None
        log.info(
            f"Init {self.__class__.__name__} with {model_name_to_download}, "
            f"{pretrained}"
        )
        super(ImageResNet, self).__init__(input_modality_shape_config)
        self.is_built = False
        self.model_name_to_download = model_name_to_download
        self.pretrained = pretrained

    def build(self, batch_dict):
        if "image" in batch_dict:
            self.build_image(batch_dict["image"].shape)

        self.is_built = True

        log.info(f"{self.__class__.__name__} built")

    def build_image(self, input_shape):
        image_input_dummy = torch.zeros(input_shape)
        self.resnet_image_embedding = timm.create_model(
            self.model_name_to_download, pretrained=self.pretrained
        )
        log.info(self.resnet_image_embedding)
        self.resnet_image_embedding.fc = nn.Identity()  # remove logit layer

        self.resnet_image_embedding.global_pool = nn.Identity()  # remove global pool

        if image_input_dummy.shape[1:] != (3, 224, 224):
            image_input_dummy = F.interpolate(image_input_dummy, size=(224, 224))

        output_shape = self.resnet_image_embedding(image_input_dummy).shape

        log.info(
            f"Built image processing network of {self.__class__.__name__} "
            f"image model with output shape {output_shape}"
        )

    def forward_image(self, x_image):
        # expects b, c, w, h input_shape
        if x_image.shape[1:] != (3, 224, 224):
            x_image = F.interpolate(x_image, size=(224, 224))

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


class AudioImageResNet(ImageResNet):
    def __init__(
        self,
        input_modality_shape_config: ShapeConfig = None,
        model_name_to_download: str = "resnet18",
        pretrained: bool = True,
        audio_kernel_size: int = 3,
    ):
        """ResNet model for image classification.

        Parameters
        ----------
        model_name_to_download : str
            Name of the model to download. List of possible names:
            alexnet
            convnext_tiny, convnext_small, convnext_base, convnext_large
            densenet121, densenet169, densenet201, densenet161

            efficientnet_b0,
            efficientnet_b1,
            efficientnet_b2,
            efficientnet_b3,
            efficientnet_b4,
            efficientnet_b5,
            efficientnet_b6,
            efficientnet_b7,
            efficientnet_v2_s,
            efficientnet_v2_m,
            efficientnet_v2_l,

            googlenet
            inception_v3
            mnasnet0_5, mnasnet0_75, mnasnet1_0, mnasnet1_3
            mobilenet_v2
            mobilenet_v3_large, mobilenet_v3_small
            raft_large, raft_small

            regnet_y_400mf,
            regnet_y_800mf,
            regnet_y_1_6gf,
            regnet_y_3_2gf,
            regnet_y_8gf,
            regnet_y_16gf,
            regnet_y_32gf,
            regnet_y_128gf,
            regnet_x_400mf,
            regnet_x_800mf,
            regnet_x_1_6gf,
            regnet_x_3_2gf,
            regnet_x_8gf,
            regnet_x_16gf,
            regnet_x_32gf,



            resnet18,
            resnet34,
            resnet50,
            resnet101,
            resnet152,
            resnext50_32x4d,
            resnext101_32x8d,
            wide_resnet50_2,
            wide_resnet101_2,

            fcn_resnet50,
            fcn_resnet101,
            deeplabv3_resnet50,
            deeplabv3_resnet101,
            deeplabv3_mobilenet_v3_large,
            lraspp_mobilenet_v3_large,

            shufflenet_v2_x0_5, shufflenet_v2_x1_0
            squeezenet1_0, squeezenet1_1
            vgg11, vgg13, vgg16, vgg19, vgg11_bn, vgg13_bn, vgg16_bn, vgg19_bn

            vit_b_16,
            vit_b_32,
            vit_l_16,
            vit_l_32,
        pretrained : bool
            Whether to load the pretrained weights.
        audio_kernel_size : int
            Kernel size of the audio convolution.
        input_modality_shape_config : dict
            Shape configuration of the input modality.

        """
        log.info(
            f"Init {self.__class__.__name__} with {model_name_to_download}, "
            f"{pretrained}"
        )
        super(ImageResNet, self).__init__(input_modality_shape_config)
        self.is_built = False
        self.audio_kernel_size = audio_kernel_size

    def build(self, batch_dict):
        if "image" in batch_dict:
            self.build_image(batch_dict["image"].shape)

        if "audio" in batch_dict:
            self.build_audio(batch_dict["audio"].shape)

        self.is_built = True

        log.info(f"{self.__class__.__name__} built")

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
