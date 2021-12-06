import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
# Modalities:
# A learning system (e.g. pretrained imagenet model that can be fine tuned,
# or a meta-learned MAML etc).
# Then gate should be able to evaluate the learning process on previously unseen tasks
# some will allow some training, and some will not (zero-shot)
#
from pytorch_lightning import LightningModule

# assume N data domains, and T tasks, and M modalities
# flexible skeleton
# a forward pass for each modality
# fprop for each modality_embeddings
# tasks might involve one or multiple modalities
# their update might inmvolve multiple modality
# how the hell do we define an optimization scheme such that we can
# explicit optimizer for each modality?

# Task:
# modalities, output shape, task process (losses etc)

# Data domains:
# modalities, different distributions


def generic_missing_forward(object, modality_name):
    raise NotImplementedError(
        f"forward_{modality_name} method not implemented in model: "
        f"{object.__class__.__name__} "
    )


class DataTaskModalityAgnosticModel(LightningModule):
    def __init__(self, modalities_supported: set):
        super(DataTaskModalityAgnosticModel, self).__init__()
        logging.info(f"Init {self.__class__.__name__}")
        self.modalities_supported = modalities_supported

    def check_if_dict_contains_valid_modalities(self, target_dict):
        for modality in self.modalities_supported:
            if isinstance(modality, str) and modality in target_dict:
                return True
            elif isinstance(modality, tuple):
                if all(sub_modality in target_dict for sub_modality in modality):
                    return True
                else:
                    continue

        raise ValueError(
            f"Invalid modalities provided in provided "
            f"dictionary. Expected"
            f" modalities were {self.modalities_supported}, "
            f"but received {list(target_dict.keys())}"
        )

    def is_modality_supported(self, modality_name):
        return modality_name in self.modalities_supported

    @staticmethod
    def add_model_specific_args(parser):
        raise NotImplementedError(
            "model specific arguments method was not implemented"
            " in {self.__class__.__name__} "
        )


# perhaps in the future try a single forward pass outputing a dictionary with
# different modalities

# learning_system.reset_learning()
# learning_system.set_task_input_output_shapes(
#     input_shape=self.input_shape,
#     output_shape=self.output_shape,
# )
# iter_metrics = learning_system.train_step(
# inputs=x, targets=y, metrics=self.metrics
# )

# TODO modalities supported necessary and optional


class AudioImageResNet(DataTaskModalityAgnosticModel):
    def __init__(self, model_name_to_download, pretrained, audio_kernel_size, **kwargs):
        logging.info(
            f"Init {self.__class__.__name__} with {model_name_to_download}, "
            f"{pretrained}"
        )
        super(AudioImageResNet, self).__init__(modalities_supported={"image", "audio"})
        self.is_built = False
        self.model_name_to_download = model_name_to_download
        self.pretrained = pretrained
        self.audio_kernel_size = audio_kernel_size
        self._input_shape = {"image": (3, 224, 224), "audio": (2, 44000)}
        self.save_hyperparameters()
        # pooling layer

    @staticmethod
    def add_model_specific_args(parser):
        parser.add_argument(
            "--model.model_name_to_download", type=str, default="resnet18"
        )
        parser.add_argument("--model.pretrained", default=False, action="store_true")
        parser.add_argument("--model.audio_kernel_size", type=int, default=5)

        return parser

    def build(self, input_dict):
        self.resnet_image_embedding = torch.hub.load(
            "pytorch/vision",
            self.model_name_to_download,
            pretrained=self.pretrained,
        )

        self.resnet_image_embedding.fc = nn.Identity()  # remove logit layer

        self.resnet_image_embedding.avgpool = nn.Identity()  # remove global

        self.resnet_audio_to_image_conv = nn.Conv1d(
            in_channels=2,
            out_channels=3,
            bias=True,
            kernel_size=self.audio_kernel_size,
        )

        self.is_built = True

    @property
    def input_shape(self):
        return self._input_shape

    @input_shape.setter
    def input_shape(self, value):

        self._input_shape = value

    def forward(self, x):
        assert self.check_if_dict_contains_valid_modalities(target_dict=x)

        if not self.is_built:
            self.build(x)

        output_dict = {}

        if "image" in x:
            output_dict["image"] = self.forward_image(x["image"])

        if "audio" in x:
            output_dict["audio"] = self.forward_audio(x["audio"])

        return output_dict

    def forward_image(self, x):
        # expects b, c, w, h input_shape

        assert len(x.shape) == 4, ValueError(
            f"Input shape for class {self.__class__.__name__} in "
            f"method forward_image must be 4, instead it is "
            f"{len(x.shape)}, for shape {x.shape}"
        )

        return self.resnet_image_embedding(x)

    def forward_audio(self, x):  # sourcery skip: square-identity
        # expects b, c, sequence_length input_shape

        if len(x.shape) != 3:
            return ValueError(
                f"Input shape for class {self.__class__.__name__} in "
                f"method forward_image must be 3, instead it is "
                f"{len(x.shape)}, for shape {x.shape}"
            )
        out = self.resnet_audio_to_image_conv.forward(x)
        out = F.adaptive_avg_pool1d(out, output_size=224 * 224)
        out = out.view(x.shape[0], 3, 224, 224)
        out = self.resnet_image_embedding.forward(out)

        return out
