from typing import Any, Callable, Dict, Tuple, Union

import torch
import torch.nn.functional as F
from dotted_dict import DottedDict
from hydra.utils import instantiate

import gate.base.utils.loggers as loggers
from gate.configs.datamodule.base import ShapeConfig
from gate.configs.task.image_classification import TaskConfig
from gate.learners.protonet import PrototypicalNetworkEpisodicTuningScheme
from gate.learners.utils import inner_gaussian_product, outer_gaussian_product

log = loggers.get_logger(__name__)


class ConditionalGenerativeContrastiveModelling(
    PrototypicalNetworkEpisodicTuningScheme
):
    def __init__(
        self,
        optimizer_config: Dict[str, Any],
        lr_scheduler_config: Dict[str, Any],
        fine_tune_all_layers: bool = False,
        use_input_instance_norm: bool = False,
        use_mean_head: bool = True,
        use_precision_head: bool = True,
        head_num_layers: int = 3,
        head_num_hidden_filters: int = 64,
        head_num_output_filters: int = 64,
        mean_head_config: Dict[str, Any] = None,
        precision_head_config: Dict[str, Any] = None,
    ):
        super(ConditionalGenerativeContrastiveModelling, self).__init__(
            optimizer_config,
            lr_scheduler_config,
            fine_tune_all_layers,
            use_input_instance_norm,
        )
        self.mean_head_config = mean_head_config
        self.precision_head_config = precision_head_config
        self.use_mean_head = use_mean_head
        self.use_precision_head = use_precision_head
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

        dummy_out = super(ConditionalGenerativeContrastiveModelling, self).forward(
            dummy_x
        )
        dummy_image_out = dummy_out["image"]
        dummy_features = {
            "image": dummy_image_out,
            "view_information": torch.randn(
                [2] + [self.precision_head_config.view_information_num_filters]
            )
            if self.precision_head_config.view_information_num_filters is not None
            else None,
        }
        # dummy_image_out = dummy_image_out.view(dummy_image_out.shape[0], -1)

        if self.use_precision_head:
            self.precision_head = instantiate(
                config=self.precision_head_config,
                num_output_filters=dummy_image_out.shape[1],
                num_hidden_filters=self.head_num_hidden_filters,
                output_activation_fn=torch.exp,
            )
            self.precision_head.build(
                input_shape={
                    key: value.shape if value is not None else None
                    for key, value in dummy_features.items()
                }
            )

            out_precision = self.precision_head(dummy_features)["image"]
        else:
            out_precision = F.adaptive_avg_pool2d(dummy_image_out, 1).view(
                dummy_image_out.shape[0], -1
            )

        if self.use_mean_head:
            self.mean_head = instantiate(
                config=self.mean_head_config,
                num_output_filters=dummy_image_out.shape[1],
                num_hidden_filters=self.head_num_hidden_filters,
                output_activation_fn=None,
            )

            self.mean_head.build(
                input_shape={
                    key: value.shape if value is not None else None
                    for key, value in dummy_features.items()
                }
            )

            out_mean = self.mean_head(dummy_features)["image"]
        else:
            out_mean = F.adaptive_avg_pool2d(dummy_image_out, 1).view(
                dummy_image_out.shape[0], -1
            )

        log.info(
            f"Built GCM learner with input_shape {self.input_shape_dict} and "
            f"mean output shape {out_mean.shape} "
            f"and precision output shape {out_precision.shape} 👍"
        )

    def forward(self, input_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:

        out = super(ConditionalGenerativeContrastiveModelling, self).forward(input_dict)

        out = {
            "image": out["image"],
            "view_information": input_dict["view_information"]
            if input_dict["view_information"] is not None
            else None,
        }

        if self.use_mean_head:
            out_mean = self.mean_head(out)["image"]
        else:
            out_mean = F.adaptive_avg_pool2d(out["image"], 1).view(
                out["image"].shape[0], -1
            )
        if self.use_precision_head:
            out_precision = self.precision_head(out)["image"]
        else:
            out_precision = F.adaptive_avg_pool2d(out["image"], 1).view(
                out["image"].shape[0], -1
            )
        learner_output_dict = {
            "mean": out_mean,
            "precision": out_precision,
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

        support_set_inputs = {"image": input_dict["image"]["support_set"]}
        support_set_targets = target_dict["image"]["support_set"]
        query_set_inputs = {"image": input_dict["image"]["query_set"]}
        query_set_targets = target_dict["image"]["query_set"]

        if "support_set_extras" in input_dict["image"]:
            support_set_view_information = input_dict["image"]["support_set_extras"][
                "crop_coordinates"
            ]
            support_set_inputs["view_information"] = support_set_view_information
        else:
            support_set_view_information = None
        print(support_set_view_information)

        if "query_set_extras" in input_dict["image"]:
            query_set_view_information = input_dict["image"]["query_set_extras"][
                "crop_coordinates"
            ]
            query_set_inputs["view_information"] = query_set_view_information
        else:
            query_set_inputs["view_information"] = None

        num_tasks, num_support_examples = support_set_inputs["image"].shape[:2]
        num_classes = int(torch.max(support_set_targets)) + 1

        support_set_inputs["image"] = support_set_inputs["image"].view(
            -1, *support_set_inputs["image"].shape[2:]
        )
        support_set_embedding = self.forward(support_set_inputs)["image"]

        support_set_embedding_mean = support_set_embedding["mean"].view(
            num_tasks, num_support_examples, -1
        )
        support_set_embedding_precision = support_set_embedding["precision"].view(
            num_tasks, num_support_examples, -1
        )

        num_tasks, num_query_examples = query_set_inputs["image"].shape[:2]

        query_set_inputs["image"] = query_set_inputs["image"].view(
            -1, *query_set_inputs["image"].shape[2:]
        )

        query_set_embedding = self.forward(query_set_inputs)["image"]

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
