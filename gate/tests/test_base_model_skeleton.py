import torch

from gate.models import ResNet
from gate.utils.logging_helpers import get_logging

logging = get_logging(level="NOTSET")

RUN_CUDA_test = False


def apply_to_test_device(model, input_tensor):
    if torch.cuda.is_available() and RUN_CUDA_test:
        model = model.to(torch.cuda.current_device())
        input_tensor = input_tensor.to(torch.cuda.current_device())
    else:
        model = model.to(torch.device("cpu"))
        input_tensor = input_tensor.to(torch.device("cpu"))

    return model, input_tensor


def test_resnet_example():
    input_image_tensor = torch.randn((8, 3, 224, 224))
    input_audio_tensor = torch.randn((8, 2, 44000))
    logging.info(f"{input_image_tensor.shape} {input_audio_tensor.shape}")
    model = ResNet(
        model_name_to_download="resnet18", pretrained=True, audio_kernel_size=7
    )
    logging.info(f"Model building done {model}")
    model, input_image_tensor = apply_to_test_device(
        model=model, input_tensor=input_image_tensor
    )

    out_image = model.forward_image(input_image_tensor)

    logging.info(f"Output image shape {out_image.shape}")

    model, input_audio_tensor = apply_to_test_device(
        model=model, input_tensor=input_audio_tensor
    )

    out_audio = model.forward_audio(input_audio_tensor)

    logging.info(f"Output audio shape {out_audio.shape}")


if __name__ == "__main__":
    test_resnet_example()
