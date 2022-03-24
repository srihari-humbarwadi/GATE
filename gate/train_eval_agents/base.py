from typing import List, Any

import torch
from hydra.utils import instantiate
from pytorch_lightning import LightningModule

from gate.base.utils.loggers import get_logger
from gate.class_configs.base import ModelConfig, LearnerConfig, TaskConfig
from gate.learners.base import LearnerModule
from gate.models.base import ModelModule
from gate.tasks.base import TaskModule

log = get_logger(__name__)


class TrainingEvaluationAgent(LightningModule):
    def __init__(
        self,
        model_config: ModelConfig,
        learner_config: LearnerConfig,
        task_config: TaskConfig,
    ):
        super().__init__()
        self.model: ModelModule = instantiate(model_config)
        self.learner: LearnerModule = instantiate(learner_config)
        self.task: TaskModule = instantiate(task_config)
        self.input_shape_dict = self.model.input_modality_shape_config
        self.output_shape_dict = self.task.output_shape_dict

    def build(self):
        dummy_batch_dict = self.datamodule.dummy_batch()
        dummy_batch_shape_dict = {
            key: value.shape
            for key, value in dummy_batch_dict.items()
            if isinstance(value, torch.Tensor)
        }
        self.model.build(dummy_batch_shape_dict)

        self.learner.build(
            model=self.model,
            input_shape_dict=self.input_shape_dict,
            output_shape_dict=self.output_shape_dict,
        )

        self.task.build(dummy_batch_shape_dict)

        output = self.learner.forward(dummy_batch_dict)

        log.info(
            f"Built {self.__class__.__name__} with {self.model.__class__.__name__} and {self.learner.__class__.__name__} and {self.task.__class__.__name__} and output shape {output.shape}"
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
