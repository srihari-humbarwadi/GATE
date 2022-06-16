import torch
import torch.nn as nn
import torch.nn.functional as F
from dotted_dict import DottedDict
from typing import Any, Dict, Tuple, Union, Callable
import copy

from torch import Tensor

import gate.base.utils.loggers as loggers
from gate.configs.datamodule.base import ShapeConfig
from gate.configs.task.image_classification import TaskConfig
from gate.learners.protonet import PrototypicalNetworkEpisodicTuningScheme
from gate.learners.utils import inner_gaussian_product, outer_gaussian_product

log = loggers.get_logger(__name__)


# torch.autograd.set_detect_anomaly(True)


class HeadMLP(nn.Module):
    def __init__(
        self,
        num_output_filters: int,
        num_layers: int,
        num_hidden_filters: int,
        output_activation_fn: Callable = None,
    ):
        super(HeadMLP, self).__init__()
        self.layer_dict = nn.ModuleDict()
        self.num_output_filters = num_output_filters
        self.num_layers = num_layers
        self.num_hidden_filters = num_hidden_filters
        self.output_activation_fn = output_activation_fn

    def build(self, input_shape: Tuple[int, ...]):

        dummy_x = torch.zeros(
            input_shape
        )  # this should be b, f; or b, c, h, w

        out = dummy_x.view(dummy_x.shape[0], -1)
        self.embedding_size = out.shape[-1]

        self.layer_dict["input_layer"] = nn.Linear(
            in_features=out.shape[1], out_features=self.num_hidden_filters
        )
        self.layer_dict["input_layer_norm"] = nn.InstanceNorm1d(
            num_features=self.num_hidden_filters
        )

        self.layer_dict["input_layer_act"] = nn.LeakyReLU()

        out = self.layer_dict["input_layer_norm"](
            self.layer_dict["input_layer"](out)
        )
        out = self.layer_dict["input_layer_act"](out)

        for i in range(self.num_layers - 2):
            self.layer_dict[f"hidden_layer_{i}"] = nn.Linear(
                in_features=self.num_hidden_filters,
                out_features=self.num_hidden_filters,
            )
            self.layer_dict[f"hidden_layer_norm_{i}"] = nn.InstanceNorm1d(
                num_features=self.num_hidden_filters
            )

            self.layer_dict[f"hidden_layer_act_{i}"] = nn.LeakyReLU()

            out = self.layer_dict[f"hidden_layer_{i}"](out)
            out = self.layer_dict[f"hidden_layer_norm_{i}"](out)
            out = self.layer_dict[f"hidden_layer_act_{i}"](out)

        self.layer_dict["output_layer"] = nn.Linear(
            in_features=out.shape[1], out_features=self.embedding_size
        )
        self.layer_dict["output_layer_norm"] = nn.InstanceNorm1d(
            num_features=self.embedding_size
        )

        self.layer_dict["output_layer_act"] = nn.LeakyReLU()

        out = self.layer_dict["output_layer_norm"](
            self.layer_dict["output_layer"](out)
        )
        out = self.layer_dict["output_layer_act"](out)

        log.info(
            f"Built HeadMLP with input_shape {dummy_x.shape} "
            f"output shape {out.shape} 👍"
        )

    def forward(
        self, input_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, Tensor]:

        out = input_dict["image"]
        # log.info(out.shape)
        out = self.layer_dict["input_layer_norm"](
            self.layer_dict["input_layer"](out)
        )
        out = self.layer_dict["input_layer_act"](out)

        for i in range(self.num_layers - 2):
            out = self.layer_dict[f"hidden_layer_{i}"](out)
            out = self.layer_dict[f"hidden_layer_norm_{i}"](out)
            out = self.layer_dict[f"hidden_layer_act_{i}"](out)

        out = self.layer_dict["output_layer_norm"](
            self.layer_dict["output_layer"](out)
        )
        # log.info(self.output_activation_fn)
        if self.output_activation_fn is not None:
            out = self.output_activation_fn(out)

        log.debug(
            f"mean {out.mean()} std {out.std()} "
            f"min {out.min()} max {out.max()} "
        )

        return {"image": out}


class ConditionalGenerativeContrastiveModelling(
    PrototypicalNetworkEpisodicTuningScheme
):
    def __init__(
        self,
        optimizer_config: Dict[str, Any],
        lr_scheduler_config: Dict[str, Any],
        fine_tune_all_layers: bool = False,
        use_input_instance_norm: bool = False,
        head_num_layers: int = 3,
        head_num_hidden_filters: int = 64,
        head_num_output_filters: int = 64,
    ):
        super(ConditionalGenerativeContrastiveModelling, self).__init__(
            optimizer_config,
            lr_scheduler_config,
            fine_tune_all_layers,
            use_input_instance_norm,
        )
        self.head_num_layers = head_num_layers
        self.head_num_hidden_filters = head_num_hidden_filters
        self.head_num_output_filters = head_num_output_filters

    def build(
        self,
        model: torch.nn.Module,
        task_config: TaskConfig,
        modality_config: Union[DottedDict, Dict],
        input_shape_dict: Union[ShapeConfig, Dict, DottedDict],
        output_shape_dict: Union[ShapeConfig, Dict, DottedDict],
    ):
        super(ConditionalGenerativeContrastiveModelling, self).build(
            model,
            task_config,
            modality_config,
            input_shape_dict,
            output_shape_dict,
        )

        dummy_x = {
            "image": torch.randn(
                [2] + list(self.input_shape_dict["image"]["shape"].values())
            )
        }  # this should be b, f; or b, c, h, w

        dummy_out = super(
            ConditionalGenerativeContrastiveModelling, self
        ).forward(dummy_x)
        dummy_image_out = dummy_out["image"]
        dummy_image_out = dummy_image_out.view(dummy_image_out.shape[0], -1)

        self.precision_head = HeadMLP(
            num_output_filters=dummy_image_out.shape[1],
            num_layers=self.head_num_layers,
            num_hidden_filters=self.head_num_hidden_filters,
            output_activation_fn=torch.exp,
        )
        self.precision_head.build(input_shape=dummy_image_out.shape)

        self.mean_head = HeadMLP(
            num_output_filters=dummy_image_out.shape[1],
            num_layers=self.head_num_layers,
            num_hidden_filters=self.head_num_hidden_filters,
            output_activation_fn=None,
        )

        self.mean_head.build(input_shape=dummy_image_out.shape)

        out_mean = self.mean_head({"image": dummy_image_out})["image"]
        out_precision = self.precision_head({"image": dummy_image_out})[
            "image"
        ]

        log.info(
            f"Built GCM learner with input_shape {self.input_shape_dict} and "
            f"mean output shape {out_mean.shape} "
            f"and precision output shape {out_precision.shape} 👍"
        )

    def forward(
        self, input_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        out = super(ConditionalGenerativeContrastiveModelling, self).forward(
            input_dict
        )
        # log.info(out["image"].shape)
        learner_output_dict = {
            "mean": self.mean_head(out)["image"],
            "precision": self.precision_head(out)["image"],
        }
        out["image"] = learner_output_dict
        return out

    def step(
        self,
        batch,
        batch_idx,
        task_metrics_dict=None,
        learner_metrics_dict=None,
        phase_name="debug",
    ):
        computed_task_metrics_dict = {}
        output_dict = {}

        input_dict, target_dict = batch

        support_set_inputs = input_dict["image"]["support_set"]
        support_set_targets = target_dict["image"]["support_set"]
        query_set_inputs = input_dict["image"]["query_set"]
        query_set_targets = target_dict["image"]["query_set"]

        num_tasks, num_support_examples = support_set_inputs.shape[:2]
        num_classes = int(torch.max(support_set_targets)) + 1

        support_set_embedding = self.forward(
            {
                "image": support_set_inputs.view(
                    -1, *support_set_inputs.shape[2:]
                )
            }
        )["image"]

        support_set_embedding_mean = support_set_embedding["mean"].view(
            num_tasks, num_support_examples, -1
        )
        support_set_embedding_precision = support_set_embedding[
            "precision"
        ].view(num_tasks, num_support_examples, -1)

        num_tasks, num_query_examples = query_set_inputs.shape[:2]

        query_set_embedding = self.forward(
            {"image": query_set_inputs.view(-1, *query_set_inputs.shape[2:])}
        )["image"]

        query_set_embedding_mean = query_set_embedding["mean"].view(
            num_tasks, num_query_examples, -1
        )

        query_set_embedding_precision = query_set_embedding["precision"].view(
            num_tasks, num_query_examples, -1
        )

        (
            proto_mean,
            proto_precision,
            log_proto_normalisation,
        ) = inner_gaussian_product(
            support_set_embedding_mean,
            support_set_embedding_precision,
            support_set_targets,
            num_classes,
        )

        (
            proto_query_product_means,
            proto_query_product_precisions,
            log_proto_query_product_normalisation,
        ) = outer_gaussian_product(
            query_set_embedding_mean,
            query_set_embedding_precision,
            proto_mean,
            proto_precision,
        )

        # NOTE: Could try exponentiating log normalisation here
        computed_task_metrics_dict[f"{phase_name}/loss"] = F.cross_entropy(
            log_proto_query_product_normalisation, query_set_targets
        )

        opt_loss_list = [computed_task_metrics_dict[f"{phase_name}/loss"]]

        _, predictions = log_proto_query_product_normalisation.max(1)

        output_dict["predictions"] = predictions

        with torch.no_grad():
            computed_task_metrics_dict[f"{phase_name}/accuracy"] = torch.mean(
                predictions.eq(query_set_targets).float()
            )

        return (
            output_dict,
            computed_task_metrics_dict,
            torch.mean(torch.stack(opt_loss_list)),
        )


# TODO: Add predict step
