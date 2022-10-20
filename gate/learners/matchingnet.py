from typing import Any, Dict

import gate.base.utils.loggers as loggers
import torch
import torch.nn.functional as F
from gate.learners.protonet import PrototypicalNetworkEpisodicTuningScheme
from gate.learners.utils import (get_cosine_distances, get_matching_accuracy,
                                 matching_logits, matching_loss)

log = loggers.get_logger(__name__)


class MatchingNetworkEpisodicTuningScheme(PrototypicalNetworkEpisodicTuningScheme):
    def __init__(
        self,
        optimizer_config: Dict[str, Any],
        lr_scheduler_config: Dict[str, Any],
        fine_tune_all_layers: bool = False,
        use_input_instance_norm: bool = False,
    ):
        super(MatchingNetworkEpisodicTuningScheme, self).__init__(
            optimizer_config,
            lr_scheduler_config,
            fine_tune_all_layers,
            use_input_instance_norm,
        )

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

        cosine_distances = get_cosine_distances(
            query_embeddings=query_set_embedding,
            support_embeddings=support_set_embedding,
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
