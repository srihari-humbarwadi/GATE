from typing import Any, Dict, Tuple, Union

import torch
from dotted_dict import DottedDict

import gate.base.utils.loggers as loggers
from gate.configs.datamodule.base import ShapeConfig
from gate.configs.task.image_classification import TaskConfig
from gate.learners.base import LearnerModule

log = loggers.get_logger(
    __name__,
)


class LinearLayerFineTuningScheme(LearnerModule):
    def __init__(
        self,
        optimizer_config: Dict[str, Any],
        lr_scheduler_config: Dict[str, Any],
        fine_tune_all_layers: bool = False,
        use_input_instance_norm: bool = False,
    ):
        super(LinearLayerFineTuningScheme, self).__init__()
        self.output_layer_dict = torch.nn.ModuleDict()
        self.input_layer_dict = torch.nn.ModuleDict()
        self.optimizer_config = optimizer_config
        self.lr_scheduler_config = lr_scheduler_config
        self.fine_tune_all_layers = fine_tune_all_layers
        self.use_input_instance_norm = use_input_instance_norm

        self.learner_metrics_dict = torch.nn.ModuleDict(
            {"loss": torch.nn.CrossEntropyLoss()}
        )

    def build(
        self,
        model: torch.nn.Module,
        task_config: TaskConfig,
        modality_config: Union[DottedDict, Dict],
        input_shape_dict: Union[ShapeConfig, Dict, DottedDict],
        output_shape_dict: Union[ShapeConfig, Dict, DottedDict],
    ):
        self.input_shape_dict = (
            input_shape_dict.__dict__
            if isinstance(input_shape_dict, ShapeConfig)
            else input_shape_dict
        )

        self.output_shape_dict = (
            output_shape_dict.__dict__
            if isinstance(output_shape_dict, ShapeConfig)
            else output_shape_dict
        )

        self.modality_config = (
            modality_config.__dict__
            if isinstance(modality_config, DottedDict)
            else modality_config
        )

        self.model = model
        self.task_config = task_config

        output_dict = {}
        for modality_name, is_supported in self.modality_config.items():
            if is_supported:
                input_dummy_x = torch.randn(
                    [2]
                    + list(
                        self.input_shape_dict[modality_name]["shape"].values()
                    )
                )

                if modality_name == "image":
                    self.input_layer_dict[
                        f"{modality_name}_input_adaptor"
                    ] = torch.nn.InstanceNorm2d(
                        num_features=input_dummy_x.shape[1],
                        affine=True,
                        track_running_stats=True,
                    )

                    input_dummy_x = self.input_layer_dict[
                        f"{modality_name}_input_adaptor"
                    ](input_dummy_x)
                elif modality_name == "audio":
                    self.input_layer_dict[
                        f"{modality_name}_input_adaptor"
                    ] = torch.nn.InstanceNorm1d(
                        num_features=input_dummy_x.shape[1],
                        affine=True,
                        track_running_stats=True,
                    )

                    input_dummy_x = self.input_layer_dict[
                        f"{modality_name}_input_adaptor"
                    ](input_dummy_x)
                elif modality_name == "video":
                    b, s, c, w, h = input_dummy_x.shape
                    input_dummy_x = input_dummy_x.view(b * s, c, w, h)
                    input_dummy_x = self.input_layer_dict[
                        f"{modality_name}_input_adaptor"
                    ](input_dummy_x)
                    input_dummy_x = input_dummy_x.view(b, s, c, w, h)

                model_features = self.model.forward(
                    {modality_name: input_dummy_x}
                )[modality_name]

                model_features_flatten = model_features.view(
                    (model_features.shape[0], -1)
                )
                # log.info(
                #     f"Output shape of model features {model_features_flatten.shape} "
                #     f"{self.output_shape_dict}"
                # )

                self.output_layer_dict[modality_name] = torch.nn.Linear(
                    model_features_flatten.shape[1],
                    self.output_shape_dict[modality_name]["num_classes"],
                )

                logits = self.output_layer_dict[modality_name](
                    model_features_flatten
                )

                output_dict[modality_name] = logits

        log.info(
            f"Built {self.__class__.__name__} "
            f"with input_shape {input_shape_dict}"
            f"with output_shape {output_shape_dict} "
            f"{[item.shape for name, item in output_dict.items()]}"
        )

    def reset_parameters(self):
        self.output_layer_dict.reset_parameters()
        self.input_layer_dict.reset_parameters()

    def get_learner_only_params(self):

        yield from list(self.input_layer_dict.parameters()) + list(
            self.output_layer_dict.parameters()
        )

    def configure_optimizers(self):
        if self.fine_tune_all_layers:
            params = self.parameters()
        else:
            params = self.get_learner_only_params()

        return super().configure_optimizers(params=params)

    def get_feature_embeddings(self, batch):

        output_dict = {}

        for modality_name, is_supported in self.modality_config.items():
            if is_supported:
                current_input = batch[modality_name]
                if self.use_input_instance_norm:
                    current_input = self.input_layer_dict[
                        f"{modality_name}_input_adaptor"
                    ](current_input)

                model_features = self.model.forward(
                    {modality_name: current_input}
                )[modality_name]

                model_features_flatten = model_features.view(
                    (model_features.shape[0], -1)
                )
                output_dict[modality_name] = model_features_flatten

        return output_dict

    def forward(self, batch):
        feature_embedding_dict = self.get_feature_embeddings(batch)
        output_dict = {}

        for modality_name, features in feature_embedding_dict.items():
            output_dict[modality_name] = self.output_layer_dict[modality_name](
                features
            )

        return output_dict

    def step(
        self,
        batch,
        batch_idx,
        task_metrics_dict,
        learner_metrics_dict,
        phase_name,
    ):
        computed_task_metrics_dict = {}
        opt_loss_list = []
        input_dict, target_dict = batch

        target_dict = {
            key: value.view(-1) for key, value in target_dict.items()
        }

        output_dict = self.forward(input_dict)

        for metric_key, metric_function in task_metrics_dict.items():
            for output_name, output_value in output_dict.items():
                computed_task_metrics_dict[
                    f"{phase_name}/{metric_key}"
                ] = metric_function(
                    output_dict[output_name],
                    target_dict[output_name],
                )

        for metric_key, metric_function in learner_metrics_dict.items():
            for output_name, output_value in output_dict.items():
                computed_task_metrics_dict[
                    f"{phase_name}/{metric_key}"
                ] = metric_function(
                    output_dict[output_name],
                    target_dict[output_name],
                )

                opt_loss_list.append(
                    computed_task_metrics_dict[f"{phase_name}/{metric_key}"]
                )

        return (
            output_dict,
            computed_task_metrics_dict,
            torch.mean(torch.stack(opt_loss_list)),
        )

    def training_step(self, batch, batch_idx, task_metrics_dict):
        output_dict, computed_task_metrics_dict, opt_loss = self.step(
            batch,
            batch_idx,
            task_metrics_dict,
            self.learner_metrics_dict,
            phase_name="training",
        )

        computed_task_metrics_dict["training/opt_loss"] = opt_loss
        output_dict["loss"] = opt_loss

        return opt_loss, computed_task_metrics_dict

    def validation_step(self, batch, batch_idx, task_metrics_dict):

        # for key, value in batch[0].items():
        #     if isinstance(value, torch.Tensor):
        #         log.info(f"{key} {value.shape}")

        output_dict, computed_task_metrics_dict, opt_loss = self.step(
            batch,
            batch_idx,
            task_metrics_dict,
            self.learner_metrics_dict,
            phase_name="validation",
        )

        computed_task_metrics_dict["validation/opt_loss"] = opt_loss

        return opt_loss, computed_task_metrics_dict

    def test_step(self, batch, batch_idx, task_metrics_dict):
        output_dict, computed_task_metrics_dict, opt_loss = self.step(
            batch,
            batch_idx,
            task_metrics_dict,
            self.learner_metrics_dict,
            phase_name="test",
        )

        computed_task_metrics_dict["test/opt_loss"] = opt_loss

        return opt_loss, computed_task_metrics_dict

    def predict_step(self, batch: Any, batch_idx: int, **kwargs):
        input_dict = batch
        return self.forward(input_dict)
