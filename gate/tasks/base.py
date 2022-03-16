from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, Union, List

import torch
import torchmetrics

from gate.base import utils

log = utils.get_logger()


class TaskModule(torch.nn.Module):
    def __init__(self, output_shape_dict: Dict, metric_class_dict: Dict):
        super(TaskModule, self).__init__()
        self.output_shape_dict = output_shape_dict
        if len(metric_class_dict) == 0:
            raise ValueError(
                f"{self.__class__.__name__} requires a metric_class_dict with at "
                f"least one entry of metric_name: metric_class format, "
                f"where a metric_class can be a torchmetric or a nn.Module based metric"
            )
        self.task_metrics = torch.nn.ModuleDict(defaultdict(torch.nn.ModuleDict))

        for phase_name in ["training", "validation", "test"]:
            for metric_name, metric_class in metric_class_dict.items():
                self.task_metrics[phase_name][metric_name] = metric_class()

    def compute_task_metrics_from_output_dict(
        self, output_dict: Dict[str, Any]
    ) -> Dict[str, Any]:
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement this necessary method."
        )


class ImageClassificationTaskModule(TaskModule):
    def __init__(self, output_shape_dict: Dict):
        self.task_class_dict = {"accuracy": torchmetrics.Accuracy}
        super(ImageClassificationTaskModule, self).__init__(
            output_shape_dict=output_shape_dict, metric_class_dict=self.task_class_dict
        )

    def compute_task_metrics_from_output_dict(
        self, output_dict: Dict[str, Any]
    ) -> Dict[str, Any]:
        return {
            metric_name: metric(
                output_dict["image_predictions"], output_dict["image_targets"]
            )
            for metric_name, metric in self.task_metrics.items()
        }
