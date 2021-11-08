from pytorch_lightning import LightningModule

from .auto_builder_densenet import (AutoConv1DDenseNet,
                                    AutoConv1DDenseNetVideo,
                                    AutoConv2DDenseNet, AutoTextNet)
from .auto_builder_models import *
from .auto_builder_transformers import *
from .densenet import *
from .resnet import *
from .system_models import CrossModalMatchingNetwork
from .wresnet import *


class CMMTransformerNetwork(LightningModule):
    def __init__(
        self,
        embedding_output_size=64,
        image_embedding_grid_patch_size=16,
        image_embedding_transformer_num_filters=128,
        image_embedding_transformer_num_layers=2,
        image_embedding_transformer_num_heads=2,
        image_embedding_transformer_dim_feedforward=128,
        image_embedding_stem_conv_bias=True,
        audio_embedding_grid_patch_size=16,
        audio_embedding_transformer_num_filters=128,
        audio_embedding_transformer_num_layers=2,
        audio_embedding_transformer_num_heads=2,
        audio_embedding_transformer_dim_feedforward=128,
        audio_embedding_stem_conv_bias=True,
        text_embedding_transformer_num_filters=128,
        text_embedding_transformer_num_layers=2,
        text_embedding_transformer_num_heads=2,
        text_embedding_transformer_dim_feedforward=128,
        text_embedding_vocab_size=49408,
        text_embedding_context_length=77,
        video_embedding_transformer_num_filters=64,
        video_embedding_transformer_num_layers=2,
        video_embedding_transformer_num_heads=8,
        video_embedding_transformer_dim_feedforward=128,
        **kwargs
    ):
        super().__init__()

        image_embedding = AutoConv2DTransformersFlatten(
            num_classes=embedding_output_size,
            grid_patch_size=image_embedding_grid_patch_size,
            transformer_num_filters=image_embedding_transformer_num_filters,
            transformer_num_layers=image_embedding_transformer_num_layers,
            transformer_num_heads=image_embedding_transformer_num_heads,
            transformer_dim_feedforward=image_embedding_transformer_dim_feedforward,
            stem_conv_bias=image_embedding_stem_conv_bias,
        )

        audio_embedding = AutoConv1DDenseNet(
            num_classes=embedding_output_size,
            num_filters=audio_embedding_transformer_num_filters,
            num_stages=audio_embedding_transformer_num_layers,
            num_blocks=audio_embedding_transformer_num_heads,
            use_squeeze_excite_attention=True,
            dilated=False,
        )

        text_embedding = AutoTextNet(
            vocab_size=text_embedding_vocab_size,
            num_filters=text_embedding_transformer_num_filters,
            num_stages=text_embedding_transformer_num_layers,
            num_blocks=text_embedding_transformer_num_heads,
            use_squeeze_excite_attention=True,
            dilated=False,
            num_classes=embedding_output_size,
        )

        video_embedding = AutoVideoTransformersFlatten(
            transformer_num_filters=video_embedding_transformer_num_filters,
            transformer_num_layers=video_embedding_transformer_num_layers,
            transformer_num_heads=video_embedding_transformer_num_heads,
            transformer_dim_feedforward=video_embedding_transformer_dim_feedforward,
            num_classes=embedding_output_size,
            image_embedding=image_embedding,
        )

        self.save_hyperparameters()

        self.model = CrossModalMatchingNetwork(
            embed_dim=embedding_output_size,
            modality_embeddings={
                "image_embedding": image_embedding,
                "audio_embedding": audio_embedding,
                "text_embedding": text_embedding,
                "video_embedding": video_embedding,
            },
        )

    def forward(self, image_input, audio_input, text_input, video_input):
        return self.model.forward(
            image_input=image_input,
            audio_input=audio_input,
            text_input=text_input,
            video_input=video_input,
        )


class CMMConvNetwork(LightningModule):
    def __init__(
        self,
        embedding_output_size=64,
        image_embedding_grid_patch_size=16,
        image_embedding_transformer_num_filters=128,
        image_embedding_transformer_num_layers=2,
        image_embedding_transformer_num_heads=2,
        image_embedding_transformer_dim_feedforward=128,
        image_embedding_stem_conv_bias=True,
        audio_embedding_grid_patch_size=16,
        audio_embedding_transformer_num_filters=128,
        audio_embedding_transformer_num_layers=2,
        audio_embedding_transformer_num_heads=2,
        audio_embedding_transformer_dim_feedforward=128,
        audio_embedding_stem_conv_bias=True,
        text_embedding_transformer_num_filters=128,
        text_embedding_transformer_num_layers=2,
        text_embedding_transformer_num_heads=2,
        text_embedding_transformer_dim_feedforward=128,
        text_embedding_vocab_size=49408,
        text_embedding_context_length=77,
        video_embedding_transformer_num_filters=64,
        video_embedding_transformer_num_layers=2,
        video_embedding_transformer_num_heads=8,
        video_embedding_transformer_dim_feedforward=128,
        **kwargs
    ):
        super().__init__()

        image_embedding = AutoConv2DDenseNet(
            num_classes=embedding_output_size,
            num_filters=image_embedding_transformer_num_filters,
            num_stages=image_embedding_transformer_num_layers,
            num_blocks=image_embedding_transformer_num_heads,
            use_squeeze_excite_attention=True,
            dilated=False,
        )

        audio_embedding = AutoConv1DDenseNet(
            num_classes=embedding_output_size,
            num_filters=audio_embedding_transformer_num_filters,
            num_stages=audio_embedding_transformer_num_layers,
            num_blocks=audio_embedding_transformer_num_heads,
            use_squeeze_excite_attention=True,
            dilated=False,
        )

        text_embedding = AutoTextNet(
            vocab_size=text_embedding_vocab_size,
            num_filters=text_embedding_transformer_num_filters,
            num_stages=text_embedding_transformer_num_layers,
            num_blocks=text_embedding_transformer_num_heads,
            use_squeeze_excite_attention=True,
            dilated=False,
            num_classes=embedding_output_size,
        )

        video_embedding = AutoVideoTransformersFlatten(
            transformer_num_filters=video_embedding_transformer_num_filters,
            transformer_num_layers=video_embedding_transformer_num_layers,
            transformer_num_heads=video_embedding_transformer_num_heads,
            transformer_dim_feedforward=video_embedding_transformer_dim_feedforward,
            num_classes=embedding_output_size,
            image_embedding=image_embedding,
        )

        self.save_hyperparameters()

        self.model = CrossModalMatchingNetwork(
            embed_dim=embedding_output_size,
            modality_embeddings={
                "image_embedding": image_embedding,
                "audio_embedding": audio_embedding,
                "text_embedding": text_embedding,
                "video_embedding": video_embedding,
            },
        )

    def forward(self, image_input, audio_input, text_input, video_input):
        return self.model.forward(
            image_input=image_input,
            audio_input=audio_input,
            text_input=text_input,
            video_input=video_input,
        )


model_zoo = {
    "CMMTransformerNetwork": CMMTransformerNetwork,
    "CMMConvNetwork": CMMConvNetwork,
}
