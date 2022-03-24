from typing import Tuple, Any

import hydra
import torch

import gate.base.utils.loggers as loggers
from gate.class_configs.base import TaskConfig, LearnerModalityConfig, ShapeConfig
from gate.learners.base import LearnerModule

log = loggers.get_logger(__name__)


class LinearLayerFineTuningScheme(LearnerModule):
    def __init__(
        self,
        task_config: TaskConfig,
        modality_config: LearnerModalityConfig,
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
            task_config=task_config, modality_config=modality_config
        )
        self.output_layer_dict = torch.nn.ModuleDict()
        self.lr = lr
        self.min_learning_rate = min_learning_rate
        self.max_epochs = max_epochs
        self.betas = betas
        self.eps = eps
        self.fine_tune_all_layers = fine_tune_all_layers
        self.weight_decay = weight_decay
        self.amsgrad = amsgrad

        self.learner_metrics_dict = torch.nn.ModuleDict(
            {"loss": torch.nn.CrossEntropyLoss()}
        )

    def build(
        self,
        model: torch.nn.Module,
        input_shape_dict: ShapeConfig,
        output_shape_dict: ShapeConfig,
    ):
        self.input_shape_dict = input_shape_dict
        self.output_shape_dict = output_shape_dict
        self.model = model

        output_dict = {}

        for modality_name, is_supported in self.modality_config.__dict__.items():
            if is_supported:

                image_dummy_x = torch.randn((2,) + input_shape_dict[modality_name])
                model_features = self.model.forward({modality_name: image_dummy_x})[
                    modality_name
                ]
                model_features_flatten = model_features.view(
                    (model_features.shape[0], -1)
                )
                log.info(
                    f"Output shape of model features {model_features_flatten.shape} "
                    f"{output_shape_dict}"
                )

                self.output_layer_dict[modality_name] = torch.nn.Linear(
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

        optimizer = hydra.utils.instantiate(config=self.optimizer_config, params=params)

        optimizer_dict = {"optimizer": optimizer}
        if self.lr_scheduler_config._target_.split(".")[-1] == "CosineAnnealingLR":
            if "T_max" not in self.lr_scheduler_config:
                self.lr_scheduler_config["T_max"] = (
                    self.num_train_samples / self.batch_size
                )
        elif (
            self.lr_scheduler_config._target_.split(".")[-1]
            == "CosineAnnealingWarmRestarts"
        ):
            if "T_0" not in self.lr_scheduler_config:
                self.lr_scheduler_config["T_0"] = (
                    self.num_train_samples / self.batch_size // 2
                )

        elif self.lr_scheduler_config._target_.split(".")[-1] == "ReduceLROnPlateau":
            self.lr_scheduler_config["patience"] = (
                self.lr_scheduler_config["patience"] * torch.cuda.device_count()
                if torch.cuda.is_available()
                else 1
            )

        lr_scheduler = hydra.utils.instantiate(
            config=self.lr_scheduler_config, optimizer=optimizer
        )
        log.info(
            f"\noptimizer: {optimizer} \n" f"lr_scheduler: {self.lr_scheduler_config}"
        )
        if isinstance(lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            self.lr_scheduler = lr_scheduler
            self.lr_scheduler_step_must_be_called_manually = True
        else:
            self.lr_scheduler_step_must_be_called_manually = False
            optimizer_dict["lr_scheduler"] = {
                "scheduler": lr_scheduler,
                "interval": "step",
            }

        return optimizer_dict

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
