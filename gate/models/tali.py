import os
import pathlib
import shutil
from typing import Optional, Union

import torch
import torch.nn as nn
import wandb
from dotted_dict import DottedDict
from tali.models.systems import ModusPrime

from gate.base.utils.loggers import get_logger
from gate.base.utils.model_utils import resize_custom
from gate.configs.datamodule.base import ShapeConfig
from gate.models.base import ModelModule

log = get_logger()


def download_return_checkpoint_path(
    model_root: Union[str, pathlib.Path] = ".gate/model_hub/",
    model_name: str = "tali-vasi-modality-v-1.0",
    model_version: str = "v0",
    project_name: str = "machinelearningbrewery/tali-model-repo",
):
    """
    Downloads the model from the wandb repo and returns the path to the checkpoint.

    Parameters
    ----------
    model_root: str or pathlib.Path
    model_name: str
    model_version: str
    project_name: str

    Returns
    -------
    checkpoint_path: str
    """
    target_dir = f"{model_root}/{project_name}/{model_name}/{model_version}"
    model_wandb_path = f"{project_name}/{model_name}:{model_version}"
    target_filepath = os.path.abspath(f"{target_dir}/model.ckpt")
    log.info(f"Downloading model from {model_wandb_path}")
    if not pathlib.Path(target_dir).exists():
        os.makedirs(target_dir, exist_ok=True)

    if not os.path.exists(target_filepath):
        run = wandb.init()
        artifact = run.use_artifact(
            model_wandb_path,
            type="model-checkpoints",
        )
        artifact_dir = artifact.download(root=target_dir, recursive=True)
        downloaded_filepath = os.path.join(artifact_dir, "model.ckpt")
        for subdir, dir, files in os.walk(artifact_dir):
            for file in files:
                if file.endswith(".ckpt"):
                    downloaded_filepath = os.path.join(subdir, file)

        shutil.move(src=downloaded_filepath, dst=target_filepath)

    return target_filepath


def build_modus_prime_from_checkpoint(checkpoint_path, pretrained=True):
    loaded_model = torch.load(checkpoint_path)
    model = ModusPrime(**loaded_model["hyper_parameters"])
    batch_dict = {
        "image": torch.zeros((2, 3, 288, 176)),
        "video": torch.zeros((2, 8, 3, 288, 176)),
        "text": torch.zeros((2, 77)),
        "audio": torch.zeros((2, 2, 220500)),
    }
    _ = model(batch_dict)

    if pretrained:
        model.load_state_dict(loaded_model["state_dict"])

    return model


class TALIModusPrime(ModelModule):
    def __init__(
        self,
        input_shape_dict: Union[DottedDict, ShapeConfig],
        model_root_dir: str = None,
        model_name_to_download: str = "tali-vasi-modality-v-1.0",
        model_version: str = "v0",
        project_name: str = "machinelearningbrewery/tali-model-repo",
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
        self.model = None
        self.image_shape = None
        self.resnet_image_embedding = None
        log.info(
            f"Init {self.__class__.__name__} with {model_name_to_download}, "
            f"{pretrained}"
        )
        super(TALIModusPrime, self).__init__(
            input_shape_dict=input_shape_dict,
        )

        self.is_built = False
        self.model_name_to_download = model_name_to_download
        self.pretrained = pretrained
        self.name = self.__class__.__name__
        self.model_path = download_return_checkpoint_path(
            model_root=os.environ["MODEL_DIR"]
            if model_root_dir is None
            else model_root_dir,
            model_name=model_name_to_download,
            model_version=model_version,
            project_name=project_name,
        )
        self.image_shape = list(
            self.input_shape_dict["image"]["shape"].values()
        )

    def build(self, batch_dict: Optional[DottedDict] = None):

        self.model = build_modus_prime_from_checkpoint(
            checkpoint_path=self.model_path, pretrained=self.pretrained
        )

        self.model.system.modality_embeddings[
            "image"
        ].output_layer = nn.Identity()
        self.model.system.modality_embeddings[
            "text"
        ].output_layer = nn.Identity()
        self.model.system.modality_embeddings[
            "audio"
        ].output_layer = nn.Identity()
        self.model.system.modality_embeddings[
            "audio"
        ].output_layer = nn.Identity()

        self.is_built = True

        log.info(f"{self.__class__.__name__} built")

    def forward_image(self, x_image):
        # expects b, c, w, h input_shape
        # print(f"Pre image shape: {x_image.shape}")
        if x_image.shape[1:] != self.image_shape:
            x_image = resize_custom(
                x_image, target_image_shape=self.image_shape
            )
        # print(f"Post image shape: {x_image.shape}")
        # print(f"Target image shape: {self.image_shape}")

        if len(x_image.shape) != 4:
            raise ValueError(
                f"Input shape for class {self.__class__.__name__} in "
                f"method forward_image must be 4, instead it is "
                f"{len(x_image.shape)}, for shape {x_image.shape}"
            )
        (
            embedding_feature_dict,
            logits_similarities_dict,
            targets_dict,
        ) = self.model.forward({"image": x_image})
        return embedding_feature_dict["image"]

    def forward_audio(self, x_audio):
        # expects b, c, w, h input_shape
        if len(x_audio.shape) != 3:
            raise ValueError(
                f"Input shape for class {self.__class__.__name__} in "
                f"method forward_image must be 4, instead it is "
                f"{len(x_audio.shape)}, for shape {x_audio.shape}"
            )
        (
            embedding_feature_dict,
            logits_similarities_dict,
            targets_dict,
        ) = self.model.forward({"audio": x_audio})
        return embedding_feature_dict["audio"]

    def forward_text(self, x_text):
        # expects b, c, w, h input_shape
        if len(x_text.shape) != 2:
            raise ValueError(
                f"Input shape for class {self.__class__.__name__} in "
                f"method forward_image must be 4, instead it is "
                f"{len(x_text.shape)}, for shape {x_text.shape}"
            )

        (
            embedding_feature_dict,
            logits_similarities_dict,
            targets_dict,
        ) = self.model.forward({"text": x_text})
        return embedding_feature_dict["text"]

    def forward_video(self, x_video):
        # expects b, c, w, h input_shape
        if len(x_video.shape) != 5:
            raise ValueError(
                f"Input shape for class {self.__class__.__name__} in "
                f"method forward_image must be 4, instead it is "
                f"{len(x_video.shape)}, for shape {x_video.shape}"
            )
        b, s, c, w, h = x_video.shape
        if x_video.shape[2:] != self.image_shape:
            x_video = resize_custom(
                x_video.view(b * s, c, w, h),
                target_image_shape=self.image_shape,
            )

            x_video = x_video.view(b, s, *self.image_shape)

        (
            embedding_feature_dict,
            logits_similarities_dict,
            targets_dict,
        ) = self.model.forward({"video": x_video})
        return embedding_feature_dict["video"]

    def forward(self, x):
        if not self.is_built:
            self.build(x)

        output_dict = {}

        if "image" in x:
            output_dict["image"] = self.forward_image(x["image"])

        if "audio" in x:
            output_dict["audio"] = self.forward_audio(x["audio"])

        if "text" in x:
            output_dict["text"] = self.forward_text(x["text"])

        if "video" in x:
            output_dict["video"] = self.forward_video(x["video"])

        return output_dict
