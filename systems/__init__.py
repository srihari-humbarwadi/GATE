from pytorch_lightning import LightningModule

from .base import DataTaskModalityAgnosticSystem

learning_systems_dict = {
    DataTaskModalityAgnosticSystem.__class__.__name__: DataTaskModalityAgnosticSystem
}
