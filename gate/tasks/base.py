import torch
import torch.nn as nn
import torch.nn.functional as F

# Task has data, loss and data stream
from gate.utils.general_utils import compute_accuracy


class Task(nn.Module):
    def __init__(self):
        super(Task, self).__init__()
        self.metrics = None
        self.data_provider = None

    def train_on_task(self, learning_system, data_provider):
        return NotImplementedError

    def eval_on_task(self, learning_system, data_provider):
        return NotImplementedError

    def inference_on_task(self, learning_system, x):
        return NotImplementedError


class ImageClassificationTask(Task):
    def __init__(self, num_epochs, input_shape, output_shape):
        super(ImageClassificationTask, self).__init__()
        self.metrics = {
            "cross_entropy": lambda x, y: F.cross_entropy(input=x, target=y),
            "accuracy": lambda x, y: compute_accuracy(x, y),
        }
        self.num_epochs = num_epochs
        self.input_shape = input_shape
        self.output_shape = output_shape

    def train_on_task(self, learning_system, data_batch):
        iter_metrics = learning_system.train_step(
            inputs=x, targets=y, metrics=self.metrics
        )

    def eval_on_task(self, learning_system, data_provider):
        for _ in range(self.num_epochs):
            for iter_idx, (x, y) in enumerate(data_provider):
                iter_metrics = learning_system.eval_step(
                    inputs=x, targets=y, metrics=self.metrics
                )
                yield iter_metrics

    def inference_on_task(self, learning_system, inputs):
        return learning_system.inference_step(inputs=inputs)


class ReconstructionTask(Task):
    def __init__(self, num_epochs, input_shape, output_shape):
        super(ReconstructionTask, self).__init__()
        self.metrics = {
            "mse": lambda x, y: F.mse_loss(input=x, target=y),
            "mae": lambda x, y: F.l1_loss(input=x, target=y),
        }
        self.num_epochs = num_epochs
        self.input_shape = input_shape
        self.output_shape = output_shape

    def train_on_task(self, learning_system, data_provider):
        for epoch_idx in range(self.num_epochs):
            for iter_idx, (x, y) in enumerate(data_provider):
                if iter_idx and epoch_idx == 0:
                    learning_system.reset_learning()
                    learning_system.set_task_input_output_shapes(
                        input_shape=self.input_shape,
                        output_shape=self.output_shape,
                    )
                iter_metrics = learning_system.train_step(
                    inputs=x, targets=y, metrics=self.metrics
                )
                yield iter_metrics

    def eval_on_task(self, learning_system, data_provider):
        for _ in range(self.num_epochs):
            for iter_idx, (x, y) in enumerate(data_provider):
                iter_metrics = learning_system.eval_step(
                    inputs=x, targets=y, metrics=self.metrics
                )
                yield iter_metrics

    def inference_on_task(self, learning_system, inputs):
        return learning_system.inference_step(inputs=inputs)
