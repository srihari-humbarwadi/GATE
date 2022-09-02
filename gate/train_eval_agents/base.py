from typing import Any, Dict

import torch
from dotted_dict import DottedDict
from hydra.utils import instantiate
from omegaconf import DictConfig
from pytorch_lightning import LightningModule

from gate.base.utils.loggers import get_logger
from gate.configs.learner.linear_layer_fine_tuning import LearnerConfig
from gate.configs.task.image_classification import TaskConfig
from gate.datamodules.base import DataModule
from gate.learners.base import LearnerModule
from gate.models.base import ModelModule
from gate.tasks.base import TaskModule

log = get_logger(__name__, set_default_handler=False)


class TrainingEvaluationAgent(LightningModule):
    def __init__(
        self,
        model_config: Any,
        learner_config: LearnerConfig,
        task_config: TaskConfig,
        modality_config: DictConfig,
        datamodule: DataModule,
    ):
        super().__init__()
        log.info(
            f"Initializing {self.__class__.__name__}, model_config: {model_config}, "
            f"learner_config: {learner_config}, task_config: {task_config}, "
            f"modality_config: {modality_config} "
        )
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
        self.input_shape_dict = model_config.input_shape_dict
        self.output_shape_dict = task_config.output_shape_dict
        self.build(datamodule)

    def build(self, datamodule):
        input_dummy_dict, target_dummy_dict = datamodule.dummy_batch()

        self.base_model.build(input_dummy_dict)

        self.learner.build(
            model=self.base_model,
            task_config=self.task_config,
            modality_config=self.modality_config,
            input_shape_dict=self.input_shape_dict,
            output_shape_dict=self.output_shape_dict,
        )

        output_dict, _, _ = self.learner.step(
            batch=(input_dummy_dict, target_dummy_dict), batch_idx=0
        )
        get_dict_shapes = lambda x: {key: value.shape for key, value in x.items()}

        output_shape_dict = {
            name: get_dict_shapes(value) if isinstance(value, Dict) else value.shape
            for name, value in output_dict.items()
        }

        log.info(
            f"Built {self.__class__.__name__} "
            f"with {self.base_model.__class__.__name__} "
            f"and {self.learner.__class__.__name__} "
            f"and {self.task.__class__.__name__} "
            f"and output shape {output_shape_dict} "
        )

    def forward(self, batch):
        return self.learner.step(batch, batch_idx=0)

    def training_step(self, batch, batch_idx):
        task_batch = self.task.data_flow(batch_dict=batch, batch_idx=batch_idx)
        opt_loss, computed_task_metrics_dict = self.learner.training_step(
            task_batch,
            batch_idx,
            task_metrics_dict=self.task.task_metrics_dict,
        )
        self.collect_metrics_step(computed_task_metrics_dict)
        return opt_loss

    def validation_step(self, batch, batch_idx):
        task_batch = self.task.data_flow(batch_dict=batch, batch_idx=batch_idx)
        opt_loss, computed_task_metrics_dict = self.learner.validation_step(
            task_batch,
            batch_idx,
            task_metrics_dict=self.task.task_metrics_dict,
        )
        self.collect_metrics_step(computed_task_metrics_dict)

    def test_step(self, batch, batch_idx):
        task_batch = self.task.data_flow(batch_dict=batch, batch_idx=batch_idx)
        opt_loss, computed_task_metrics_dict = self.learner.test_step(
            task_batch,
            batch_idx,
            task_metrics_dict=self.task.task_metrics_dict,
        )
        self.collect_metrics_step(computed_task_metrics_dict)

    def configure_optimizers(self):
        return self.learner.configure_optimizers()

    def collect_metrics_step(self, computed_task_metrics_dict):
        # sourcery skip: boolean-if-exp-identity

        for metric_key, computed_value in computed_task_metrics_dict.items():

            if computed_value is not None:
                self.log(
                    name=metric_key,
                    value=computed_value.detach(),
                    prog_bar=True if "opt_loss" in metric_key else False,
                    logger=True,
                    on_step=True,
                    on_epoch=True,
                    sync_dist=True,
                )
