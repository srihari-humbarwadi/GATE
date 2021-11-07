import logging
from collections import OrderedDict
from typing import Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from rich.logging import RichHandler
from torch import nn

FORMAT = "%(message)s"
logging.basicConfig(
    level=logging.INFO, format=FORMAT, datefmt="[%X]", handlers=[RichHandler()]
)

log = logging.getLogger("rich")


class CrossModalMatchingNetwork(nn.Module):
    def __init__(
            self,
            embed_dim: int,
            modality_embeddings: dict,
            logit_scale: float = 1.0
            # = 1 / 0.07
    ):
        super(CrossModalMatchingNetwork, self).__init__()

        self.embed_dim = embed_dim
        self.modality_embeddings = nn.ModuleDict(modality_embeddings)
        self.logit_scale = nn.Parameter(
            torch.ones([]) * np.log(logit_scale), requires_grad=True
        )
        self.is_built = False

    def build(
            self,
            text_input_shape=None,
            image_shape=None,
            video_shape=None,
            audio_shape=None,
            **kwargs,
    ):
        log.info(f"{text_input_shape}, {image_shape}, {video_shape}, {audio_shape}")
        if image_shape is not None:
            self.modality_embeddings["image_embedding"].build(image_shape)
            self._check_modality_embedding_shape(image_shape, "image_embedding")

        if text_input_shape is not None:
            self.modality_embeddings["text_embedding"].build(text_input_shape)
            self._check_modality_embedding_shape(text_input_shape, "text_embedding")

        if video_shape is not None:
            self.modality_embeddings["video_embedding"].build(video_shape)
            self._check_modality_embedding_shape(video_shape, "video_embedding")

        if audio_shape is not None:
            self.modality_embeddings["audio_embedding"].build(audio_shape)
            self._check_modality_embedding_shape(audio_shape, "audio_embedding")

        log.info(
            f"built {self.__class__.__name__} with output shape {self.embed_dim}",
        )

        self.is_built = True

    def _check_modality_embedding_shape(self, input_shape, embedding_name):
        input_dummy = torch.zeros(size=input_shape)

        embeddings, _ = self.modality_embeddings[embedding_name].forward(input_dummy)

        assert embeddings.shape[1] == self.embed_dim

    def _get_normalized_features(self, inputs, embedding_name):

        if inputs is not None:
            inputs, _ = self.modality_embeddings[embedding_name](inputs)
            inputs = inputs / inputs.norm(dim=-1, keepdim=True)
            return inputs
        else:
            return None

    def _compute_cross_modal_cosine_similarities(
            self, image_embeddings, text_embeddings, video_embeddings, audio_embeddings
    ):
        logit_scale = self.logit_scale.exp()
        embedding_dict = {
            "image": image_embeddings,
            "text": text_embeddings,
            "video": video_embeddings,
            "audio": audio_embeddings,
        }
        logit_dict = {}

        for source_key, source_value in embedding_dict.items():
            for target_key, target_value in embedding_dict.items():
                if (
                        source_key != target_key
                        and source_value is not None
                        and target_value is not None
                ):
                    logit_dict[
                        f"{source_key}_to_{target_key}_similarity"
                    ] = torch.matmul(logit_scale * source_value, target_value.t())

        return logit_dict

    def forward(
            self, image_input=None, text_input=None, video_input=None, audio_input=None
    ):
        # log.debug(
        #     f"{image_input.shape}, {text_input.shape}, "
        #     f"{video_input.shape}, {audio_input.shape}"
        # )
        if not self.is_built:
            self.build(
                text_input_shape=text_input.shape
                if isinstance(text_input, torch.Tensor)
                else None,
                image_shape=image_input.shape
                if isinstance(image_input, torch.Tensor)
                else None,
                video_shape=video_input.shape
                if isinstance(video_input, torch.Tensor)
                else None,
                audio_shape=audio_input.shape
                if isinstance(audio_input, torch.Tensor)
                else None,
            )

        (image_embeddings, text_embeddings, video_embeddings, audio_embeddings) = (
            self._get_normalized_features(inputs, embedding_name)
            for inputs, embedding_name in zip(
            [image_input, text_input, video_input, audio_input],
            [
                "image_embedding",
                "text_embedding",
                "video_embedding",
                "audio_embedding",
            ],
        )
        )

        return (
            dict(
                image_embeddings=image_embeddings,
                text_embeddings=text_embeddings,
                video_embeddings=video_embeddings,
                audio_embeddings=audio_embeddings,
            ),
            self._compute_cross_modal_cosine_similarities(
                image_embeddings=image_embeddings,
                text_embeddings=text_embeddings,
                video_embeddings=video_embeddings,
                audio_embeddings=audio_embeddings,
            ),
        )


def contrastive_logits_labels(logits):
    labels = (
        torch.arange(logits.shape[1])
            .unsqueeze(0)
            .repeat([logits.shape[0], 1])
            .to(logits.device)
    )
    return logits, labels
