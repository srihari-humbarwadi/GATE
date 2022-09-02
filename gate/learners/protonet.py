from typing import Any, Dict, Union

import torch
import torch.nn.functional as F
from dotted_dict import DottedDict

import gate.base.utils.loggers as loggers
from gate.configs.datamodule.base import ShapeConfig
from gate.configs.task.image_classification import TaskConfig
from gate.learners.base import LearnerModule
from gate.learners.utils import get_accuracy, get_prototypes, prototypical_loss

log = loggers.get_logger(__name__)


class PrototypicalNetworkEpisodicTuningScheme(LearnerModule):
    def __init__(
        self,
        optimizer_config: Dict[str, Any],
        lr_scheduler_config: Dict[str, Any],
        fine_tune_all_layers: bool = False,
        use_input_instance_norm: bool = False,
    ):
        super(PrototypicalNetworkEpisodicTuningScheme, self).__init__()
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
                    [2] + list(self.input_shape_dict[modality_name]["shape"].values())
                )
                if self.use_input_instance_norm:
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

                model_features = self.model.forward({modality_name: input_dummy_x})[
                    modality_name
                ]
                self.backbone_output_shape = {"image": model_features.shape}
                model_features_flatten = model_features.view(
                    (model_features.shape[0], -1)
                )

                output_dict[modality_name] = model_features_flatten

        log.info(
            f"Built {self.__class__.__name__} "
            f"with input_shape {input_shape_dict}"
            f"with output_shape {output_shape_dict} "
            f"{[item.shape for name, item in output_dict.items()]}"
        )

    def reset_parameters(self):
        self.input_layer_dict.reset_parameters()

    def get_learner_only_params(self):

        yield from list(self.input_layer_dict.parameters())

    def get_learner_only_named_params(self):

        yield from list(self.input_layer_dict.named_parameters())

    def configure_optimizers(self):
        if self.fine_tune_all_layers:
            params = self.parameters()
            named_params = self.named_parameters()
        else:
            params = self.get_learner_only_params()
            named_params = self.get_learner_only_named_params()

        return super().configure_optimizers(params=params, named_params=named_params)

    def get_feature_embeddings(self, batch):

        output_dict = {}

        for modality_name, is_supported in self.modality_config.items():
            if is_supported:
                current_input = batch[modality_name]
                if self.use_input_instance_norm:
                    current_input = self.input_layer_dict[
                        f"{modality_name}_input_adaptor"
                    ](current_input)

                model_features = self.model.forward({modality_name: current_input})[
                    modality_name
                ]

                # Keep features non-flattened for now for downstream use in GCM
                output_dict[modality_name] = model_features  # _flatten

        return output_dict

    def forward(self, batch):

        return self.get_feature_embeddings(batch)

    def step(
        self,
        batch,
        batch_idx,
        task_metrics_dict=None,
        learner_metrics_dict=None,
        phase_name="debug",
    ):
        output_dict = {}

        input_dict, target_dict = batch

        support_set_inputs = input_dict["image"]["support_set"]
        support_set_targets = target_dict["image"]["support_set"]
        query_set_inputs = input_dict["image"]["query_set"]
        query_set_targets = target_dict["image"]["query_set"]

        num_tasks, num_examples = support_set_inputs.shape[:2]
        support_set_embedding = self.forward(
            {"image": support_set_inputs.view(-1, *support_set_inputs.shape[2:])}
        )["image"]
        support_set_embedding = F.adaptive_avg_pool2d(support_set_embedding, 1)
        support_set_embedding = support_set_embedding.view(num_tasks, num_examples, -1)

        num_tasks, num_examples = query_set_inputs.shape[:2]
        query_set_embedding = self.forward(
            {"image": query_set_inputs.view(-1, *query_set_inputs.shape[2:])}
        )["image"]
        query_set_embedding = F.adaptive_avg_pool2d(query_set_embedding, 1)
        query_set_embedding = query_set_embedding.view(num_tasks, num_examples, -1)

        prototypes = get_prototypes(
            embeddings=support_set_embedding,
            targets=support_set_targets,
            num_classes=int(torch.max(support_set_targets)) + 1,
        )

        computed_task_metrics_dict = {
            f"{phase_name}/loss": prototypical_loss(
                prototypes, query_set_embedding, query_set_targets
            )
        }

        opt_loss_list = [computed_task_metrics_dict[f"{phase_name}/loss"]]

        with torch.no_grad():
            computed_task_metrics_dict[f"{phase_name}/accuracy"] = get_accuracy(
                prototypes, query_set_embedding, query_set_targets
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
        # TODO create a predict step that can return embeddings, prototypes and distances between a support and query set depending on what keys are in the batch dictionary i.e. if only 'image' with one tensor as input, do just fprop, and if it includes support and query sets then do prototypes and distances, if only support set, generate predictions and prototypes

        input_dict = batch
        return self.forward(input_dict)
