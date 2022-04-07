from typing import Any, Dict, Union

import torch
import torch.nn as nn

from gate.base.utils.loggers import get_logger
from gate.class_configs.base import TaskConfig, ModalitiesSupportedConfig, ShapeConfig

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

    def build(
        self,
        model: torch.nn.Module,
        task_config: TaskConfig,
        modality_config: ModalitiesSupportedConfig,
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

    def configure_optimizers(self):
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
