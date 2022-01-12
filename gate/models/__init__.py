from pytorch_lightning import LightningModule

from .base import AudioImageResNet

model_library_dict = {AudioImageResNet.__name__: AudioImageResNet}
