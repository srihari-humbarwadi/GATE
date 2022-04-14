from typing import Any, Dict, Generator

import torchmetrics

from gate.tasks.base import TaskModule


class ImageClassificationTaskModule(TaskModule):
    def __init__(
        self,
        output_shape_dict: Dict,
        name: str,
    ):
        self.task_metrics_dict = {"accuracy": torchmetrics.Accuracy}
        super(ImageClassificationTaskModule, self).__init__(
            name=self.__class__.__name__ if name is None else name,
            output_shape_dict=output_shape_dict,
            task_metrics_dict=self.task_metrics_dict,
        )

    # integrate in the main train_eval_agent
    def data_flow(self, batch_dict: Dict[str, Any], batch_idx: int) -> Generator:
        yield batch_dict
