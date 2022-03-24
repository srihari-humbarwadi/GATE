from typing import Any, Dict

import torchmetrics

from gate.tasks.base import TaskModule


class ImageClassificationTaskModule(TaskModule):
    def __init__(self, output_shape_dict: Dict):
        self.task_class_dict = {"accuracy": torchmetrics.Accuracy}
        super(ImageClassificationTaskModule, self).__init__(
            output_shape_dict=output_shape_dict, metric_class_dict=self.task_class_dict
        )

    def data_flow(self, data_dict: Dict[str, Any], batch_idx: int) -> Dict[str, Any]:
        return data_dict
