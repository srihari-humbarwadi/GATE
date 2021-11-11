from pytorch_lightning import LightningModule

from .base import ResNet

learning_systems_dict = {ResNet.__class__.__name__: ResNet}
