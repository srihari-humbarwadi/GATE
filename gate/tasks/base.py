from collections import defaultdict
from typing import Any, Dict

import torch

from gate.base.utils import loggers

log = loggers.get_logger()


# task normally involves a loss/objective, perhaps the output shape and the data flow


class TaskModule(torch.nn.Module):
    def __init__(
        self,
        output_shape_dict: Dict,
        task_metrics_dict: Dict,
    ):
        super(TaskModule, self).__init__()
        self.output_shape_dict = output_shape_dict
        if len(task_metrics_dict) == 0:
            raise ValueError(
                f"{self.__class__.__name__} requires a metric_class_dict with at "
                f"least one entry of metric_name: metric_class format, "
                f"where a metric_class can be a torchmetric or a nn.Module based metric"
            )
        self.task_metrics_dict = torch.nn.ModuleDict(
            defaultdict(torch.nn.ModuleDict)
        )

        for metric_name, metric_class in task_metrics_dict.items():
            self.task_metrics_dict[metric_name] = metric_class()

    def data_flow(
        self, batch_dict: Dict[str, Any], batch_idx: int
    ) -> Dict[str, Any]:
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement this necessary method."
        )
