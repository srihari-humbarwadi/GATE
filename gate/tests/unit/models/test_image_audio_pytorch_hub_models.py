# import pytest
# import torch
#
# from gate.base.utils.loggers import get_logger
# from gate.class_configs.base import (
#     ShapeConfig,
# )
# from gate.models.resnet import ImageResNet, AudioImageResNet
#
# log = get_logger(__name__, set_default_handler=True)
#
#
# @pytest.mark.parametrize(
#     "model_name",
#     [
#         "resnet18",
#         "resnet34",
#         "resnet50",
#         "resnet101",
#         "resnet152",
#     ],
# )
# @pytest.mark.parametrize("pretrained", [True, False])
# @pytest.mark.parametrize("image_shape", [[3, 32, 32], [3, 224, 224]])
# @pytest.mark.parametrize("audio_kernel_size", [1, 3, 5])
# def test_pytorch_hub_models(model_name, pretrained, image_shape, audio_kernel_size):
#     model = AudioImageResNet(
#         input_shape_dict=ShapeConfig(image=image_shape),
#         model_name_to_download=model_name,
#         pretrained=pretrained,
#         audio_kernel_size=audio_kernel_size,
#     )
#     dummy_x = {
#         "image": torch.randn(size=[2] + model.input_shape_dict.image),
#     }
#
#     log.info(f"dummy_x.shape: {dummy_x['image'].shape}")
#
#     out = model.forward(dummy_x)
#     log.debug(model)
#
#     assert out is not None
