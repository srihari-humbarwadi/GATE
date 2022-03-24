from typing import Optional

from pytorch_lightning import LightningDataModule
from pytorch_lightning.utilities.types import EVAL_DATALOADERS

from gate.class_configs.base import (
    DataLoaderConfig,
    DatasetConfig,
)


class DataModule(LightningDataModule):
    def __init__(
        self,
        dataset_config: DatasetConfig,
        data_loader_config: DataLoaderConfig,
    ):
        super(DataModule, self).__init__()

    def prepare_data(self, **kwargs):
        raise NotImplementedError

    def setup(self, stage: Optional[str] = None):
        raise NotImplementedError

    def dummy_batch(self):
        raise NotImplementedError

    def train_dataloader(self):
        raise NotImplementedError

    def val_dataloader(self):
        raise NotImplementedError

    def test_dataloader(self):
        raise NotImplementedError

    def predict_dataloader(self) -> EVAL_DATALOADERS:
        raise NotImplementedError
