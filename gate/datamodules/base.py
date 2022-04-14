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
        self.train_set = None
        self.val_set = None
        self.test_set = None
        self.dataset_name = dataset_config.dataset_name
        self.dataset_root = dataset_config.dataset_root
        self.seed = data_loader_config.seed
        self.dataset_config = dataset_config
        self.data_loader_config = data_loader_config

        self.input_shape_dict = dataset_config.input_shape_dict

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
