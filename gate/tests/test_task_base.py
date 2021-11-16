# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from rich import print
#
# from gate.adaptation_schemes.base import ImageOnlyLinearLayerFineTuningScheme
# from gate.models import AudioImageResNet
# from gate.tasks.base import ImageClassificationTask
# from gate.utils.general_utils import compute_accuracy
# from gate.utils.logging_helpers import get_logging
#
# logging = get_logging('NOTSET')
#
# def test_AudioImageResNetBase():
#     model = AudioImageResNet(
#         model_name_to_download="resnet18", pretrained=True, audio_kernel_size=5
#     )
#
#     learning_system = ImageOnlyLinearLayerFineTuningScheme(
#         model=model,
#         input_shape_dict={"image": (3, 224, 224)},
#         output_shape_dict={"image": (100,)},
#         output_layer_activation=nn.Identity(),
#         num_epochs=100
#     )
#     args =
#     task = ImageClassificationTask()
#     x_dummy = torch.randn(32, 3, 224, 224)
#     y_dummy = torch.randint(high=100, size=(32,))
#     input_dict = {'image': x_dummy}
#     output_dict = {'image': y_dummy}
#     batch = (input_dict, output_dict)
#     metrics = {
#         "cross_entropy": lambda x, y: F.cross_entropy(input=x, target=y).detach(),
#         "accuracy": lambda x, y: compute_accuracy(x, y),
#     }
#     out = learning_system.training_step(batch=batch, metrics=metrics)
#     logging.info(out)
#     out = learning_system.evaluation_step(batch=batch, metrics=metrics)
#     logging.info(out)
#     out, features = learning_system.predict_step(batch=batch)
#     logging.info(f'{out.shape}, {features.shape}')
#     assert out.shape == torch.Size([32, 100])
#
#
#
#
#
