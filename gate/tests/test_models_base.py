import torch
import torch.nn as nn
import torch.nn.functional as F
from rich import print

from gate.models import AudioImageResNet


def test_AudioImageResNetBase():
    model = AudioImageResNet(
        model_name_to_download="resnet18", pretrained=True, audio_kernel_size=5
    )
    x_dummy = torch.randn(32, 3, 224, 224)
    out = model.forward_image(x_dummy)

    assert out.shape == torch.Size([32, 25088])
