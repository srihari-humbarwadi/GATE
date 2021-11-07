import torch

from models.auto_builder_densenet import AutoDenseNet
from models.auto_builder_models import (EasyPeasyConvNet,
                                        EasyPeasyConvRelationalNet,
                                        EasyPeasyResNet)
from models.auto_builder_transformers import (AutoConv1DTransformersFlatten,
                                              AutoConv2DTransformersFlatten,
                                              AutoTextTransformersFlatten,
                                              AutoViTFlatten,
                                              AutoViTLastTimeStep,
                                              Conv1DTransformer,
                                              Conv2DTransformer,
                                              TexTransformer)

RUN_CUDA_test = False


def apply_to_test_device(model, input_tensor):
    if torch.cuda.is_available() and RUN_CUDA_test:
        model = model.to(torch.cuda.current_device())

        input_tensor = input_tensor.to(torch.cuda.current_device())

    else:

        model = model.to(torch.device("cpu"))

        input_tensor = input_tensor.to(torch.device("cpu"))

    return model, input_tensor


def test_EasyPeasyViTFlatten_layer_output_shape():
    model = AutoViTFlatten(
        grid_patch_size=32,
        transformer_num_filters=512,
        transformer_num_layers=8,
        transformer_num_heads=8,
        model_name_to_download=None,
        pretrained=False,
        num_classes=10,
    )
    dummy_x = torch.zeros((8, 3, 128, 128))
    model, dummy_x = apply_to_test_device(model, dummy_x)
    out, features = model.forward(dummy_x)

    assert len(out.shape) == 2
    assert out.shape[1] == 10


def test_EasyPeasyViTLastTimeStep_layer_output_shape():
    model = AutoViTLastTimeStep(
        grid_patch_size=32,
        transformer_num_filters=512,
        transformer_num_layers=8,
        transformer_num_heads=8,
        model_name_to_download=None,
        pretrained=False,
        num_classes=10,
    )
    dummy_x = torch.zeros((8, 3, 128, 128))
    model, dummy_x = apply_to_test_device(model, dummy_x)
    out, features = model.forward(dummy_x)

    assert len(out.shape) == 2
    assert out.shape[1] == 10


def test_EasyPeasyViTFlattenPretrained_layer_output_shape():
    model = AutoViTFlatten(
        grid_patch_size=32,
        transformer_num_filters=768,
        transformer_num_layers=12,
        transformer_num_heads=12,
        model_name_to_download="ViT-B-32",
        pretrained=True,
        num_classes=10,
    )
    dummy_x = torch.zeros((8, 3, 224, 224))
    model, dummy_x = apply_to_test_device(model, dummy_x)
    out, features = model.forward(dummy_x)

    assert len(out.shape) == 2
    assert out.shape[1] == 10


def test_EasyPeasyViTLastTimeStepPretrained_layer_output_shape():
    model = AutoViTLastTimeStep(
        grid_patch_size=32,
        transformer_num_filters=768,
        transformer_num_layers=12,
        transformer_num_heads=12,
        model_name_to_download="ViT-B-32",
        pretrained=True,
        num_classes=10,
    )
    dummy_x = torch.zeros((8, 3, 224, 224))
    model, dummy_x = apply_to_test_device(model, dummy_x)
    out, features = model.forward(dummy_x)

    assert len(out.shape) == 2
    assert out.shape[1] == 10


def test_AutoDenseNet_layer_output_shape():
    model = AutoDenseNet(
        num_classes=10,
        num_filters=16,
        num_stages=3,
        num_blocks=3,
        dilated=False,
        use_squeeze_excite_attention=False,
    )
    dummy_x = torch.zeros((8, 3, 224, 224))
    model, dummy_x = apply_to_test_device(model, dummy_x)
    out, features = model.forward(dummy_x)

    assert len(out.shape) == 2
    assert out.shape[1] == 10


def test_Conv2DTransformer_layer_output_shape():
    model = Conv2DTransformer(
        grid_patch_size=16,
        transformer_num_filters=128,
        transformer_num_layers=4,
        transformer_num_heads=4,
        transformer_dim_feedforward=128,
        stem_conv_bias=True,
    )
    dummy_x = torch.zeros((8, 3, 224, 224))
    model, dummy_x = apply_to_test_device(model, dummy_x)
    out = model.forward(dummy_x)

    assert len(out.shape) == 3
    # assert out.shape[1] == 10


def test_Conv1DTransformer_layer_output_shape():
    model = Conv1DTransformer(
        grid_patch_size=16,
        transformer_num_filters=128,
        transformer_num_layers=4,
        transformer_num_heads=4,
        transformer_dim_feedforward=128,
        stem_conv_bias=True,
    )
    dummy_x = torch.zeros((8, 3, 224))
    model, dummy_x = apply_to_test_device(model, dummy_x)
    out = model.forward(dummy_x)

    assert len(out.shape) == 3
    # assert out.shape[1] == 10


def test_TexTransformer_layer_output_shape():
    model = TexTransformer(
        transformer_num_filters=32,
        transformer_num_layers=8,
        transformer_num_heads=8,
        transformer_dim_feedforward=128,
        transformer_embedding_num_output_filters=12,
        vocab_size=32,
        context_length=64,
    )
    dummy_x = torch.randint(low=0, high=32, size=(64, 64)).long()
    model, dummy_x = apply_to_test_device(model, dummy_x)
    out = model.forward(dummy_x)

    assert len(out.shape) == 3
    # assert out.shape[1] == 10


def test_AutoConv1DTransformer_layer_output_shape():
    model = AutoConv1DTransformersFlatten(
        grid_patch_size=16,
        transformer_num_filters=128,
        transformer_num_layers=4,
        transformer_num_heads=4,
        transformer_dim_feedforward=128,
        stem_conv_bias=True,
        num_classes=10,
    )
    dummy_x = torch.zeros((8, 3, 224))
    model, dummy_x = apply_to_test_device(model, dummy_x)
    out = model.forward(dummy_x)

    assert len(out.shape) == 2
    assert out.shape[1] == 10


def test_AutoConv2DTransformer_layer_output_shape():
    model = AutoConv2DTransformersFlatten(
        grid_patch_size=16,
        transformer_num_filters=128,
        transformer_num_layers=4,
        transformer_num_heads=4,
        transformer_dim_feedforward=128,
        stem_conv_bias=True,
        num_classes=10,
    )
    dummy_x = torch.zeros((8, 3, 224, 224))
    model, dummy_x = apply_to_test_device(model, dummy_x)
    out = model.forward(dummy_x)

    assert len(out.shape) == 2
    assert out.shape[1] == 10


def test_AutoTexTransformerLastTimeStep_layer_output_shape():
    model = AutoTextTransformersFlatten(
        transformer_num_filters=32,
        transformer_num_layers=8,
        transformer_num_heads=8,
        transformer_dim_feedforward=128,
        transformer_embedding_num_output_filters=12,
        vocab_size=32,
        context_length=64,
        num_classes=10,
    )
    dummy_x = torch.randint(low=0, high=32, size=(64, 64)).long()
    model, dummy_x = apply_to_test_device(model, dummy_x)
    out, features = model.forward(dummy_x)

    assert len(out.shape) == 2
    assert out.shape[1] == 10
