from typing import Any, Dict, Union

import hydra
import torch
import torch.nn as nn
from dotted_dict import DottedDict
from omegaconf import DictConfig

from gate.base.utils.loggers import get_logger
from gate.configs.datamodule.base import ShapeConfig
from gate.configs.task.image_classification import TaskConfig
from gate.learners.utils import learning_scheduler_smart_autofill

log = get_logger(__name__)


class LearnerModule(nn.Module):
    def __init__(
        self,
    ):
        """
        Initialize the learner.
        Parameters
        ----------
        modality_config: ModelModalityConfig - the modality configuration
        """
        super(LearnerModule, self).__init__()
        self.modality_config = None
        self.task_config = None
        self.input_shape_dict = None
        self.output_shape_dict = None

        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.lr_scheduler_step_must_be_called_manually = False

    def lr_scheduler_step_manual_mode(self):
        self.lr_scheduler_step_must_be_called_manually = True

        return self.lr_scheduler_step_must_be_called_manually

    def lr_scheduler_step_auto_mode(self):
        self.lr_scheduler_step_must_be_called_manually = False

        return self.lr_scheduler_step_must_be_called_manually

    def build(
        self,
        model: torch.nn.Module,
        task_config: TaskConfig,
        modality_config: DictConfig,
        input_shape_dict: Union[ShapeConfig, Dict],
        output_shape_dict: Union[ShapeConfig, Dict],
    ):
        """
        Build the learner.
        Parameters
        ----------
        modality_config
        task_config
        model
        input_shape_dict
        output_shape_dict

        Returns
        -------

        """
        self.input_shape_dict = input_shape_dict
        self.output_shape_dict = output_shape_dict
        self.modality_config = modality_config
        self.model = model
        self.task_config = task_config
        raise NotImplementedError

    def reset_parameters(self):
        raise NotImplementedError

    def configure_optimizers(self, params=None, named_params=None):

        if params is None:
            params = self.parameters()

        self.optimizer = hydra.utils.instantiate(
            config=self.optimizer_config, params=params
        )

        if named_params is not None:
            log.info("Printing optimizer learnable weights 🏋")

            log.info("------------------------------------------------------")
            for name, param in named_params:
                if param.requires_grad:
                    log.info(f"{name} {param.shape}")

        self.optimizer_dict = {"optimizer": self.optimizer}

        self.lr_scheduler_config = learning_scheduler_smart_autofill(
            lr_scheduler_config=self.lr_scheduler_config,
            batch_size=self.lr_scheduler_config.batch_size,
            num_train_samples=self.lr_scheduler_config.num_train_samples,
        )

        del self.lr_scheduler_config.batch_size
        del self.lr_scheduler_config.num_train_samples
        learning_scheduler_update_interval = (
            self.lr_scheduler_config.update_interval
        )
        del self.lr_scheduler_config.update_interval

        lr_scheduler = hydra.utils.instantiate(
            config=self.lr_scheduler_config, optimizer=self.optimizer
        )

        if isinstance(lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            self.lr_scheduler_step_manual_mode()
            self.lr_scheduler = lr_scheduler

        else:
            self.lr_scheduler_step_auto_mode()
            self.optimizer_dict["lr_scheduler"] = {
                "scheduler": lr_scheduler,
                "interval": learning_scheduler_update_interval,
            }

        log.info(
            f"\noptimizer: {self.optimizer} \n" f"lr_scheduler: {lr_scheduler}"
        )

        return self.optimizer_dict

    def forward(self, batch):
        raise NotImplementedError

    def step(
        self,
        batch,
        batch_idx,
        task_metrics_dict=None,
        learner_metrics_dict=None,
        phase_name="undefined",
    ):
        raise NotImplementedError

    def training_step(
        self,
        batch,
        batch_idx,
        task_metrics_dict,
    ):
        raise NotImplementedError

    def validation_step(self, batch, batch_idx, task_metrics_dict):
        raise NotImplementedError

    def test_step(self, batch, batch_idx, task_metrics_dict):
        raise NotImplementedError

    def predict_step(self, batch: Any, batch_idx: int):
        raise NotImplementedError
