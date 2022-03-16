from typing import Any, Tuple

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

from gate.base.utils import get_logger
from gate.config import LearnerlModalityConfig, TaskConfig

log = get_logger(__name__)


class LearnerModule(nn.Module):
    def __init__(
        self,
        model,
        task_config: TaskConfig,
        modality_config: LearnerlModalityConfig,
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
        self.optimizer = None
        self.scheduler = None
        self.model = model
        self.task_config = task_config
        self.modality_config = modality_config

        self.input_shape_dict = model.input_shape_dict
        self.output_shape_dict = task_config.output_shape_dict

        self.build(
            input_shape_dict=self.input_shape_dict,
            output_shape_dict=self.output_shape_dict,
        )

    def build(self, input_shape_dict, output_shape_dict):
        """
        Build the learner.
        Parameters
        ----------
        input_shape_dict
        output_shape_dict

        Returns
        -------

        """

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


class LinearLayerFineTuningScheme(LearnerModule):
    def __init__(
        self,
        model: nn.Module,
        task_config: TaskConfig,
        modality_config: LearnerlModalityConfig,
        fine_tune_all_layers=False,
        max_epochs: int = 100,
        min_learning_rate: float = 1e-6,
        lr: float = 1e-3,
        betas: Tuple[float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        amsgrad: bool = False,
    ):
        super(LinearLayerFineTuningScheme, self).__init__(
            model=model, task_config=task_config, modality_config=modality_config
        )
        self.output_layer_dict = nn.ModuleDict()
        self.lr = lr
        self.min_learning_rate = min_learning_rate
        self.max_epochs = max_epochs
        self.betas = betas
        self.eps = eps
        self.fine_tune_all_layers = fine_tune_all_layers
        self.weight_decay = weight_decay
        self.amsgrad = amsgrad

        self.input_shape_dict = model.input_shape_dict
        self.output_shape_dict = task_config.output_shape_dict
        self.learner_metrics_dict = nn.ModuleDict({"loss": nn.CrossEntropyLoss()})

        self.build(
            input_shape_dict=self.input_shape_dict,
            output_shape_dict=self.output_shape_dict,
        )

    def build(self, input_shape_dict, output_shape_dict):

        output_dict = {}

        for modality_name, is_supported in self.modality_config.__dict__.items():
            if is_supported:

                image_dummy_x = torch.randn((2,) + input_shape_dict[modality_name])
                model_features = self.model.forward({modality_name: image_dummy_x})[
                    f"{modality_name}_predictions"
                ]
                model_features_flatten = model_features.view(
                    (model_features.shape[0], -1)
                )
                log.info(
                    f"Output shape of model features {model_features_flatten.shape} "
                    f"{output_shape_dict}"
                )

                self.output_layer_dict[modality_name] = nn.Linear(
                    model_features_flatten.shape[1],
                    output_shape_dict[modality_name][0],
                )

                logits = self.output_layer_dict[modality_name](model_features_flatten)

                output_dict[modality_name] = logits

        log.info(
            f"Built {self.__class__.__name__} "
            f"with input_shape {input_shape_dict}"
            f"with output_shape {output_shape_dict} "
            f"{[item.shape for name, item in output_dict.items()]}"
        )

    def reset_parameters(self):
        self.linear_output_layer.reset_parameters()

    def optimizers(self):
        if self.fine_tune_all_layers:
            params = self.parameters()
        else:
            params = self.output_layer_dict.parameters()

        self.optimizer = Adam(
            params=params,
            lr=self.lr,
            betas=self.betas,
            eps=self.eps,
            weight_decay=self.weight_decay,
            amsgrad=self.amsgrad,
        )

        self.scheduler = CosineAnnealingLR(
            optimizer=self.optimizer,
            T_max=self.max_epochs,
            eta_min=self.min_learning_rate,
        )

        return {"optimizer": self.optimizer, "lr_scheduler": self.scheduler}

    def forward(self, batch):
        output_dict = {}

        for modality_name, is_supported in self.modality_config.__dict__.items():
            if is_supported:
                model_features = self.model.forward(
                    {modality_name: batch[modality_name]}
                )[modality_name]
                model_features_flatten = model_features.view(
                    (model_features.shape[0], -1)
                )

                output_dict[modality_name] = self.output_layer_dict[modality_name](
                    model_features_flatten
                )

        return output_dict

    def step(
        self, batch, batch_idx, task_metrics_dict, learner_metrics_dict, phase_name
    ):
        input_dict, target_dict = batch

        output_dict = self.forward(input_dict)

        computed_task_metrics_dict = {}

        for metric_key, metric_function in task_metrics_dict.items():
            for output_name, output_value in output_dict.items():
                computed_task_metrics_dict[
                    f"{phase_name}/{metric_key}"
                ] = metric_function(
                    output_dict[output_name],
                    target_dict[output_name],
                )
        opt_loss = torch.tensor(0.0)
        for metric_key, metric_function in learner_metrics_dict.items():
            for output_name, output_value in output_dict.items():
                computed_task_metrics_dict[
                    f"{phase_name}/{metric_key}"
                ] = metric_function(
                    output_dict[output_name],
                    target_dict[output_name],
                )
                opt_loss += computed_task_metrics_dict[metric_key]

        return output_dict, computed_task_metrics_dict, opt_loss

    def training_step(self, batch, batch_idx, task_metrics_dict):
        output_dict, computed_task_metrics_dict, opt_loss = self.step(
            batch,
            batch_idx,
            task_metrics_dict,
            self.learner_metrics_dict,
            phase_name="training",
        )

        output_dict["loss"] = opt_loss

        return output_dict

    def validation_step(self, batch, batch_idx, task_metrics_dict):
        output_dict, computed_task_metrics_dict, opt_loss = self.step(
            batch,
            batch_idx,
            task_metrics_dict,
            self.learner_metrics_dict,
            phase_name="validation",
        )

        output_dict["loss"] = opt_loss

        return output_dict

    def test_step(self, batch, batch_idx, task_metrics_dict):
        output_dict, computed_task_metrics_dict, opt_loss = self.step(
            batch,
            batch_idx,
            task_metrics_dict,
            self.learner_metrics_dict,
            phase_name="test",
        )

        output_dict["loss"] = opt_loss

        return output_dict

    def predict_step(self, batch: Any, batch_idx: int, **kwargs):
        input_dict = batch
        return self.forward(input_dict)
