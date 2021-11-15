from pytorch_lightning import LightningModule

from .base import ResNet

model_library_dict = {ResNet.__class__.__name__: ResNet}
