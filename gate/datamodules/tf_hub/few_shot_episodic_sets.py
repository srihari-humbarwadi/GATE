from typing import Optional

import hydra.utils
import torch.utils.data
from torch.utils.data import DataLoader

from gate.configs.datamodule.base import DataLoaderConfig
from gate.configs.datamodule.few_shot_classification import (
    FewShotDatasetConfig,
    FewShotTransformConfig,
)
from gate.datamodules.base import DataModule


class FewShotDataModule(DataModule):
    def __init__(
        self,
        dataset_config: FewShotDatasetConfig,
        data_loader_config: DataLoaderConfig,
        transform_train: FewShotTransformConfig,
        transform_eval: FewShotTransformConfig,
        train_num_episodes: int,
        eval_num_episodes: int,
    ):

        super(FewShotDataModule, self).__init__(
            dataset_config, data_loader_config
        )

        self.data_loader_config = data_loader_config

        self.transform_train = transform_train
        self.transform_eval = transform_eval
        self.rescan_cache = self.dataset_config.rescan_cache
        self.train_num_episodes = train_num_episodes
        self.eval_num_episodes = eval_num_episodes

    def setup(self, stage: Optional[str] = None):

        if stage == "fit":
            self.train_set = hydra.utils.instantiate(
                config=self.dataset_config,
                split_name="train",
                support_set_input_transform=(
                    self.transform_train.support_set_input_transform
                ),
                query_set_input_transform=(
                    self.transform_train.query_set_input_transform
                ),
                support_set_target_transform=(
                    self.transform_train.support_set_target_transform
                ),
                query_set_target_transform=(
                    self.transform_train.query_set_target_transform
                ),
                _recursive_=False,
                rescan_cache=self.rescan_cache,
                num_episodes=self.train_num_episodes,
            )

            self.val_set = hydra.utils.instantiate(
                config=self.dataset_config,
                split_name="val",
                support_set_input_transform=(
                    self.transform_eval.support_set_input_transform
                ),
                query_set_input_transform=(
                    self.transform_eval.query_set_input_transform
                ),
                support_set_target_transform=(
                    self.transform_eval.support_set_target_transform
                ),
                query_set_target_transform=(
                    self.transform_eval.query_set_target_transform
                ),
                _recursive_=False,
                rescan_cache=False,
                num_episodes=self.eval_num_episodes,
            )
            if self.rescan_cache is True:
                self.rescan_cache = False

            self.input_shape_dict = self.train_set.input_shape_dict
        elif stage == "validate":
            self.val_set = hydra.utils.instantiate(
                config=self.dataset_config,
                split_name="val",
                support_set_input_transform=(
                    self.transform_eval.support_set_input_transform
                ),
                query_set_input_transform=(
                    self.transform_eval.query_set_input_transform
                ),
                support_set_target_transform=(
                    self.transform_eval.support_set_target_transform
                ),
                query_set_target_transform=(
                    self.transform_eval.query_set_target_transform
                ),
                _recursive_=self.rescan_cache,
                rescan_cache=False,
                num_episodes=self.eval_num_episodes,
            )
            if self.rescan_cache is True:
                self.rescan_cache = False

            self.input_shape_dict = self.train_set.input_shape_dict
            # Assign test dataset for use in dataloader(s)
        elif stage == "test" or stage is None:
            self.test_set = hydra.utils.instantiate(
                config=self.dataset_config,
                split_name="test",
                support_set_input_transform=(
                    self.transform_eval.support_set_input_transform
                ),
                query_set_input_transform=(
                    self.transform_eval.query_set_input_transform
                ),
                support_set_target_transform=(
                    self.transform_eval.support_set_target_transform
                ),
                query_set_target_transform=(
                    self.transform_eval.query_set_target_transform
                ),
                _recursive_=False,
                rescan_cache=self.rescan_cache,
                num_episodes=self.eval_num_episodes,
            )

            if self.rescan_cache is True:
                self.rescan_cache = False

            self.input_shape_dict = self.test_set.input_shape_dict

        else:
            raise ValueError(
                f"Stage {stage} is not supported."
                f" Supported stages are: fit, validate, test."
            )

    def dummy_batch(self):
        temp_dataloader = DataLoader(
            self.val_set,
            batch_size=self.data_loader_config.val_batch_size,
            shuffle=self.data_loader_config.eval_shuffle,
            num_workers=self.data_loader_config.num_workers,
            pin_memory=self.data_loader_config.pin_memory,
            prefetch_factor=self.data_loader_config.prefetch_factor,
            persistent_workers=self.data_loader_config.persistent_workers,
            drop_last=self.data_loader_config.eval_drop_last,
        )

        for batch in temp_dataloader:
            input_dict, target_dict = batch
            return input_dict, target_dict

    def train_dataloader(self):

        return DataLoader(
            self.train_set,
            batch_size=self.data_loader_config.train_batch_size,
            shuffle=self.data_loader_config.train_shuffle,
            num_workers=self.data_loader_config.num_workers,
            pin_memory=self.data_loader_config.pin_memory,
            prefetch_factor=self.data_loader_config.prefetch_factor,
            persistent_workers=self.data_loader_config.persistent_workers,
            drop_last=self.data_loader_config.train_drop_last,
        )

    def val_dataloader(self):

        return DataLoader(
            self.val_set,
            batch_size=self.data_loader_config.val_batch_size,
            shuffle=self.data_loader_config.eval_shuffle,
            num_workers=self.data_loader_config.num_workers,
            pin_memory=self.data_loader_config.pin_memory,
            prefetch_factor=self.data_loader_config.prefetch_factor,
            persistent_workers=self.data_loader_config.persistent_workers,
            drop_last=self.data_loader_config.eval_drop_last,
        )

    def test_dataloader(self):

        return DataLoader(
            self.test_set,
            batch_size=self.data_loader_config.test_batch_size,
            shuffle=self.data_loader_config.eval_shuffle,
            num_workers=self.data_loader_config.num_workers,
            pin_memory=self.data_loader_config.pin_memory,
            prefetch_factor=self.data_loader_config.prefetch_factor,
            persistent_workers=self.data_loader_config.persistent_workers,
            drop_last=self.data_loader_config.eval_drop_last,
        )

    def predict_dataloader(self):
        return self.test_dataloader()
