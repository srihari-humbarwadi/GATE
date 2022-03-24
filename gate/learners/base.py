from typing import Any, Dict

import torch.nn as nn

from gate.base.utils.loggers import get_logger
from gate.class_configs.base import TaskConfig, LearnerModalityConfig

log = get_logger(__name__)


class LearnerModule(nn.Module):
    def __init__(
        self,
        task_config: TaskConfig,
        modality_config: LearnerModalityConfig,
    ):
        """
        Initialize the learner.
        Parameters
        ----------
        model: nn.Module - the model to adapt
        task_config: TaskConfig - the task configuration
        modality_config: ModelModalityConfig - the modality configuration
        """
        super(LearnerModule, self).__init__()
        self.model = Any[nn.Module] = None
        self.input_shape_dict: Any[Dict] = None
        self.output_shape_dict = Any[Dict] = None
        self.optimizer = None
        self.scheduler = None

        self.task_config = task_config
        self.modality_config = modality_config

    def build(
        self,
        model: nn.Module,
        input_shape_dict: Any[Dict],
        output_shape_dict: Any[Dict],
    ):
        """
        Build the learner.
        Parameters
        ----------
        model
        input_shape_dict
        output_shape_dict

        Returns
        -------

        """
        self.input_shape_dict = input_shape_dict
        self.output_shape_dict = output_shape_dict
        self.model = model
        raise NotImplementedError

    def reset_parameters(self):
        raise NotImplementedError

    def optimizers(self):
        raise NotImplementedError

    def forward(self, batch):
        raise NotImplementedError

    def step(
        self, batch, batch_idx, task_metrics_dict, learner_metrics_dict, phase_name
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
