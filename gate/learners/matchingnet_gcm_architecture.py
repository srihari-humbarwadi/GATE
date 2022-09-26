from typing import Any, Dict, Union

import torch
import torch.nn.functional as F
from dotted_dict import DottedDict

import gate.base.utils.loggers as loggers
from gate.configs.datamodule.base import ShapeConfig
from gate.configs.task.image_classification import TaskConfig
from gate.learners.protonet_gcm_architecture import PrototypicalNetworkGCMHead
from gate.learners.utils import (
    get_cosine_distances,
    matching_logits,
    matching_loss,
    get_matching_accuracy,
)

log = loggers.get_logger(__name__)


class MatchingNetworkGCMHead(PrototypicalNetworkGCMHead):
    def __init__(
            self,
            optimizer_config: Dict[str, Any],
            lr_scheduler_config: Dict[str, Any],
            fine_tune_all_layers: bool = False,
            use_input_instance_norm: bool = False,
            use_mean_head: bool = True,
            use_precision_head: bool = True,
            head_num_layers: int = 3,
            head_num_hidden_filters: int = 512,
            head_num_output_filters: int = 512,
            mean_head_config: Dict[str, Any] = None,
            precision_head_config: Dict[str, Any] = None,
        ):
        super(MatchingNetworkGCMHead, self).__init__(
            optimizer_config,
            lr_scheduler_config,
            fine_tune_all_layers,
            use_input_instance_norm,
            use_mean_head,
            use_precision_head,
            head_num_layers,
            head_num_hidden_filters,
            head_num_output_filters,
            mean_head_config,
            precision_head_config,
        )

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
            support_set_view_information = torch.cat(
                [
                    input_dict["image"]["support_set_extras"][key]
                    for key in input_dict["image"]["support_set_extras"].keys()
                ],
                dim=2,
            )
            support_set_inputs["view_information"] = support_set_view_information
        else:
            support_set_inputs["view_information"] = None

        if "query_set_extras" in input_dict["image"]:
            query_set_view_information = torch.cat(
                [
                    input_dict["image"]["query_set_extras"][key]
                    for key in input_dict["image"]["query_set_extras"].keys()
                ],
                dim=2,
            )
            query_set_inputs["view_information"] = query_set_view_information
        else:
            query_set_inputs["view_information"] = None

        num_tasks, num_support_examples = support_set_inputs["image"].shape[:2]
        num_classes = int(torch.max(support_set_targets)) + 1

        support_set_inputs["image"] = support_set_inputs["image"].view(
            -1, *support_set_inputs["image"].shape[2:]
        )
        if support_set_inputs["view_information"] is not None:
            support_set_inputs["view_information"]=support_set_inputs["view_information"].view(-1, support_set_inputs["view_information"].shape[2])

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
        if query_set_inputs["view_information"] is not None:
            query_set_inputs["view_information"]=query_set_inputs["view_information"].view(-1, query_set_inputs["view_information"].shape[2])


        query_set_embedding = self.forward(query_set_inputs)["image"]

        query_set_embedding_mean = query_set_embedding["mean"].view(
            num_tasks, num_query_examples, -1
        )

        query_set_embedding_precision = query_set_embedding["precision"].view(
            num_tasks, num_query_examples, -1
        )

        cosine_distances = get_cosine_distances(
            query_embeddings=query_set_embedding_mean,
            support_embeddings=support_set_embedding_mean,
        )

        logits = matching_logits(
            cosine_distances=cosine_distances,
            targets=support_set_targets,
            num_classes=int(torch.max(support_set_targets)) + 1,
        )

        computed_task_metrics_dict = {
            f"{phase_name}/loss": matching_loss(logits, query_set_targets)
        }

        opt_loss_list = [computed_task_metrics_dict[f"{phase_name}/loss"]]

        with torch.no_grad():
            computed_task_metrics_dict[
                f"{phase_name}/accuracy"
            ] = get_matching_accuracy(logits=logits, targets=query_set_targets)

        return (
            output_dict,
            computed_task_metrics_dict,
            torch.mean(torch.stack(opt_loss_list)),
        )


# TODO: Add predict step
