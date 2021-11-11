import logging

import torch
import torch.nn as nn
import torch.nn.functional as F


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

# Modalities:
# A learning system (e.g. pretrained imagenet model that can be fine tuned,
# or a meta-learned MAML etc).
# Then gate should be able to evaluate the learning process on previously unseen tasks
# some will allow some training, and some will not (zero-shot)
#


def generic_missing_forward(object, modality_name):
    return NotImplementedError(
        f"forward_{modality_name} method not implemented in model: "
        f"{object.__class__.__name__} "
    )


class DataTaskModalityAgnosticModel(nn.Module):
    def __init__(self, modalities_supported: set):
        super(DataTaskModalityAgnosticModel, self).__init__()
        logging.info(f"Init {self.__class__.__name__}")
        self.modalities_supported = modalities_supported

        for modality_name in modalities_supported:
            if not hasattr(self, f"forward_{modality_name}"):
                setattr(
                    self,
                    f"forward_{modality_name}",
                    generic_missing_forward(object=self, modality_name=modality_name),
                )
                raise NotImplementedError(
                    f"forward_{modality_name}() method not implemented in model: "
                    f"{self.__class__.__name__} "
                )

    def is_modality_supported(self, modality_name):
        return modality_name in self.modalities_supported

    @staticmethod
    def add_model_specific_args(parser):
        return NotImplementedError(
            "f model specific arguments method was not implemented in "
            "self.__class__.__name__} "
        )

    # def optimizer(self):
    #     return NotImplementedError(
    #         f"Optimizer not implemented in model: " f"{self.__class__.__name__}"
    #     )
    #
    # def train_step(self, x, y):
    #     return NotImplementedError(
    #         f"Train step not implemented in model: " f"{self.__class__.__name__}"
    #     )
    #
    # def eval_step(self, x, y):
    #     return NotImplementedError(
    #         f"Evaluation step not implemented in model: " f"{self.__class__.__name__}"
    #     )


class ResNet(DataTaskModalityAgnosticModel):
    def __init__(self, model_name_to_download, pretrained, audio_kernel_size):
        print(
            f"Init {self.__class__.__name__} with {model_name_to_download}, "
            f"{pretrained}"
        )
        super(ResNet, self).__init__(modalities_supported={"image", "audio"})
        self.model_name_to_download = model_name_to_download
        self.pretrained = pretrained
        self.resnet_image_embedding = torch.hub.load(
            "pytorch/vision",
            self.model_name_to_download,
            pretrained=self.pretrained,
        )

        self.resnet_image_embedding.fc = nn.Identity()  # remove logit laayer

        self.resnet_image_embedding.avgpool = nn.Identity()  # remove global

        self.resnet_audio_to_image_conv = nn.Conv1d(
            in_channels=2, out_channels=3, bias=True, kernel_size=audio_kernel_size
        )
        # pooling layer

    def forward_image(self, x):
        # expects b, c, w, h input_shape

        if len(x.shape) != 4:
            return ValueError(
                f"Input shape for class {self.__class__.__name__} in "
                f"method forward_image must be 4, instead it is "
                f"{len(x.shape)}, for shape {x.shape}"
            )

        return self.resnet_image_embedding(x)

    def forward_audio(self, x):
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
