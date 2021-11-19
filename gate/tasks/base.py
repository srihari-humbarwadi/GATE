import logging
from typing import Any, NamedTuple, Optional

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

from gate.adaptation_schemes import adaptation_scheme_library_dict
from gate.models import model_library_dict
from gate.utils.arg_parsing import DictWithDotNotation
from gate.utils.general_utils import compute_accuracy

# TODO instead remove full args


class TaskModule(pl.LightningModule):
    def __init__(self, task_args, model_args, adaptation_scheme_args, full_args):
        super(TaskModule, self).__init__()

        self.save_hyperparameters()
        self.task_args = task_args
        self.model_args = model_args
        self.adaptation_scheme_args = adaptation_scheme_args
        self.full_args = full_args
        self.args = self.hparams

    def build(self, input_shape_dict, output_shape_dict):
        raise NotImplementedError

    @staticmethod
    def add_task_specific_args(parser):
        return parser

    def forward(self, input_dict):
        raise NotImplementedError

    def collect_metrics(self, metrics_dict, phase_name):
        for metric_key, metric_value in metrics_dict.items():
            self.log(
                name=f"{phase_name}/overall_{metric_key}",
                value=metric_value,
                prog_bar=True,
                logger=True,
                on_step=True,
                on_epoch=True,
            )

    def training_step(self, batch, batch_idx):
        raise NotImplementedError

    def validation_step(self, batch, batch_idx):
        raise NotImplementedError

    def test_step(self, batch, batch_idx):
        raise NotImplementedError

    def predict_step(
            self, batch: Any, batch_idx: int, dataloader_idx: Optional[int] = None
    ) -> Any:
        raise NotImplementedError

    def configure_optimizers(self):
        raise NotImplementedError


class BaseTaskModule(TaskModule):
    def __init__(self, task_args, model_args, adaptation_scheme_args, full_args):
        super(BaseTaskModule, self).__init__(
            task_args, model_args, adaptation_scheme_args, full_args
        )

    @property
    def task_metrics(self):
        raise NotImplementedError

    @staticmethod
    def add_task_specific_args(parser):
        return parser

    def forward(self, input_dict):
        return self.learning_system.inference_step(input_dict)

    def collect_metrics(self, metrics_dict, phase_name):
        for metric_key, metric_value in metrics_dict.items():
            self.log(
                name=f"{phase_name}/overall_{metric_key}",
                value=metric_value,
                prog_bar=True,
                logger=True,
                on_step=True,
                on_epoch=True,
            )

    def training_step(self, batch, batch_idx):
        iter_metrics = self.learning_system.training_step(
            batch=batch, metrics=self.task_metrics
        )

        self.collect_metrics(metrics_dict=iter_metrics, phase_name="training")
        return iter_metrics["loss"]

    def validation_step(self, batch, batch_idx):
        iter_metrics = self.learning_system.evaluation_step(
            batch=batch, metrics=self.task_metrics
        )
        self.collect_metrics(metrics_dict=iter_metrics, phase_name="validation")

    def test_step(self, batch, batch_idx):
        iter_metrics = self.learning_system.evaluation_step(
            batch=batch, metrics=self.task_metrics
        )

        self.collect_metrics(metrics_dict=iter_metrics, phase_name="testing")

    def predict_step(
            self, batch: Any, batch_idx: int, dataloader_idx: Optional[int] = None
    ) -> Any:
        return self.learning_system.inference_step(batch)

    def configure_optimizers(self):
        logging.info(
            f"optimizer: {repr(self.learning_system.optimizer)},"
            f"max-epochs: {self.adaptation_scheme_args.max_epochs},"
            f"current-epoch: {self.current_epoch}",
        )
        return {
            "optimizer": self.learning_system.optimizer,
            "lr_scheduler": self.learning_system.scheduler,
        }


class ImageClassificationTask(BaseTaskModule):
    def __init__(self, task_args, model_args, adaptation_scheme_args, full_args):
        super(ImageClassificationTask, self).__init__(
            task_args, model_args, adaptation_scheme_args, full_args
        )

    @property
    def task_metrics(self):
        return {
            "cross_entropy": lambda x, y: F.cross_entropy(input=x, target=y),
            "accuracy": lambda x, y: compute_accuracy(x, y),
        }

    def build(self, input_shape_dict, output_shape_dict):
        logging.info(
            f"{input_shape_dict}, {output_shape_dict}"
        )
        self.model = model_library_dict[self.model_args.name](**self.model_args)

        self.learning_system = adaptation_scheme_library_dict[
            self.adaptation_scheme_args.name
        ](
            model=self.model,
            task_config=DictWithDotNotation(dict(
                input_shape_dict={"image": input_shape_dict["image"]},
                output_shape_dict={"image": output_shape_dict["image"]},
                type="image_classification",
                eval_metric_dict=self.task_metrics,
            )),
            **self.adaptation_scheme_args,
        )

        self.learning_system.reset_learning()

