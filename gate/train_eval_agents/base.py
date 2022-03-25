from typing import List, Any

import torch
from hydra.utils import instantiate
from pytorch_lightning import LightningModule

from gate.base.utils.loggers import get_logger
from gate.class_configs.base import (
    ModelConfig,
    LearnerConfig,
    TaskConfig,
    ModalitiesSupportedConfig,
)
from gate.datamodules.base import DataModule
from gate.learners.base import LearnerModule
from gate.models.base import ModelModule
from gate.tasks.base import TaskModule

log = get_logger(__name__, set_default_handler=True)


class TrainingEvaluationAgent(LightningModule):
    def __init__(
        self,
        model_config: ModelConfig,
        learner_config: LearnerConfig,
        task_config: TaskConfig,
        modality_config: ModalitiesSupportedConfig,
        datamodule: DataModule,
    ):
        super().__init__()
        self.base_model: ModelModule = instantiate(
            model_config,
            _recursive_=False,
            _convert_="partial",
        )
        self.learner: LearnerModule = instantiate(
            learner_config,
            _recursive_=False,
            _convert_="partial",
        )
        self.task: TaskModule = instantiate(
            task_config,
            _recursive_=False,
            _convert_="partial",
        )
        self.task_config = task_config
        self.modality_config = modality_config
        self.input_shape_dict = self.base_model.input_modality_shape_config
        self.output_shape_dict = self.task.output_shape_dict
        self.build(datamodule)

    def build(self, datamodule):
        input_dummy_dict, target_dummy_dict = datamodule.dummy_batch()

        dummy_batch_dict = {
            key: value
            for key, value in input_dummy_dict.items()
            if isinstance(value, torch.Tensor)
        }
        self.base_model.build(dummy_batch_dict)

        self.learner.build(
            model=self.base_model,
            task_config=self.task_config,
            modality_config=self.modality_config,
            input_shape_dict=self.input_shape_dict,
            output_shape_dict=self.output_shape_dict,
        )

        output_dict = self.learner.forward(dummy_batch_dict)
        output_shape_dict = {name: value.shape for name, value in output_dict.items()}
        log.info(
            f"Built {self.__class__.__name__} with {self.base_model.__class__.__name__} and {self.learner.__class__.__name__} and {self.task.__class__.__name__} and output shape {output_shape_dict}"
        )

    def forward(self, batch):
        self.learner.forward(batch)

    def training_step(self, batch, batch_idx):
        self.learner.training_step(
            batch, batch_idx, task_metrics_dict=self.task.task_metrics
        )

    def training_epoch_end(self, outputs: List[Any]):
        log.info(f"\nTraining epoch {self.current_epoch} ended.\n")
        self.collect_metrics_epoch(phase_name="training")
        self.reset_metric_caches(phase_name="training")

    def validation_step(self, batch, batch_idx):
        self.learner.validation_step(
            batch, batch_idx, task_metrics_dict=self.task.task_metrics
        )

    def validation_epoch_end(self, outputs: List[Any]):
        log.info(f"\nValidation epoch {self.current_epoch} ended.\n")
        self.collect_metrics_epoch(phase_name="validation")
        self.reset_metric_caches(phase_name="validation")

    def test_step(self, batch, batch_idx):
        self.learner.testing_step(
            batch, batch_idx, task_metrics_dict=self.task.task_metrics
        )

    def test_epoch_end(self, outputs: List[Any]):
        log.info(f"\nTest epoch {self.current_epoch} ended.\n")
        self.collect_metrics_epoch(phase_name="test")
        self.reset_metric_caches(phase_name="test")

    def configure_optimizers(self):
        return self.learner.configure_optimizers()

    def reset_metric_caches(self, phase_name):
        for key in self.per_modality_metrics_computed_dict[
            f"{phase_name}-metrics"
        ].keys():
            self.per_modality_metrics_computed_dict[f"{phase_name}-metrics"][
                key
            ].reset()
