from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np


class BaseAdaptationScheme(LightningModule):
    def __init__(self):
        super(BaseAdaptationScheme, self).__init__()

    def optimizers(self, **kwargs):
        raise NotImplementedError(
            f"Optimizer not implemented in model: " f"{self.__class__.__name__}"
        )

    def training_step(self, **kwargs):
        raise NotImplementedError(
            f"Train step not implemented in model: " f"{self.__class__.__name__}"
        )

    def evaluation_step(self, **kwargs):
        raise NotImplementedError(
            f"Evaluation step not implemented in model: " f"{self.__class__.__name__}"
        )

    @staticmethod
    def add_adaptation_scheme_specific_args(parser):
        raise NotImplementedError


# a. Embedding component
# b. Adaptation_scheme component: Updates weights, provide a dictionary/set of which
# layers to update
# c.

# learning system: <"model" (architecture, weights), way to process different combos
# of modalities,
# way to update the model, way to transforms inputs, way to map embeddings to outputs

# learning_system.reset_learning()
# learning_system.set_task_input_output_shapes(
#     input_shape=self.input_shape,
#     output_shape=self.output_shape,
# )
# iter_metrics = learning_system.train_step(
# inputs=x, targets=y, metrics=self.metrics
# )
import logging


class ImageOnlyLinearLayerFineTuningScheme(BaseAdaptationScheme):
    def __init__(
            self,
            model,
            task_config,
            max_epochs=100,
            min_learning_rate=1e-6,
            lr=1e-3,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0,
            amsgrad=False,
            **kwargs,
    ):
        super(ImageOnlyLinearLayerFineTuningScheme, self).__init__()
        self.model = model

        self.lr = lr
        self.min_learning_rate = min_learning_rate
        self.max_epochs = max_epochs
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.amsgrad = amsgrad
        self.input_shape_dict = task_config.input_shape_dict
        self.output_shape_dict = task_config.output_shape_dict
        self.task_type = task_config.type
        self.eval_metric_dict = task_config.eval_metric_dict

        self.build(
            input_shape_dict=task_config.input_shape_dict,
            output_shape_dict=task_config.output_shape_dict,
            task_type=task_config.type,
            eval_metric_dict=task_config.eval_metric_dict,
        )

        self.optimizers()

    @staticmethod
    def add_adaptation_scheme_specific_args(parser):
        # parser.add_argument("--data.download", default=False, action="store_true")
        # parser.add_argument("--data.val_set_percentage", type=float, default=0.1)
        return parser

    def build(self, input_shape_dict, output_shape_dict, task_type, eval_metric_dict):
        image_dummy_x = torch.randn((2,) + input_shape_dict["image"])
        model_features = self.model.forward({'image': image_dummy_x})['image']
        model_features_flatten = model_features.view((
            model_features.shape[0], -1))
        logging.info(f"Output shape of model features {model_features_flatten.shape} "
                     f"{output_shape_dict}")

        self.linear_output_layer = nn.Linear(
            in_features=model_features_flatten.shape[1],
            out_features=output_shape_dict["image"][0],
            bias=True,
        )

        logits = self.linear_output_layer(model_features_flatten)

        output_dict = {"image_features": model_features_flatten, "image": logits}

        logging.info(
            f"Built {self.__class__.__name__} "
            f"with input_shape {input_shape_dict} {image_dummy_x.shape}"
            f"with output_shape {output_shape_dict} "
            f"{[item.shape for name, item in output_dict.items()]}"
        )

    def reset_learning(self):
        self.linear_output_layer.reset_parameters()

    def optimizers(self):
        self.optimizer = Adam(
            params=self.parameters(),
            lr=self.lr,
            betas=self.betas,
            eps=self.eps,
            weight_decay=self.weight_decay,
            amsgrad=self.amsgrad,
        )

        self.scheduler = CosineAnnealingLR(
            optimizer=self.optimizer,
            T_max=self.max_epochs,
            eta_min=self.min_learning_rate,
        )

        return {"optimizer": self.optimizer, "lr_scheduler": self.scheduler}

    def training_step(self, batch, metrics):
        input_dict, target_dict = batch
        features = self.model.forward(input_dict)["image"]
        features_flatten = F.leaky_relu(features.view((features.shape[0], -1)))
        logits = self.linear_output_layer(features_flatten)
        loss = F.cross_entropy(input=logits, target=target_dict["image"][:, 0])

        output_dict = {
            f"{metric_key}": metric_function(logits, target_dict["image"][:, 0])
            for metric_key, metric_function in metrics.items()
        }

        output_dict["loss"] = loss

        return output_dict

    def evaluation_step(self, batch, metrics):
        input_dict, target_dict = batch
        features = self.model.forward(input_dict)["image"]
        features_flatten = F.leaky_relu(features.view((features.shape[0], -1)))
        logits = self.linear_output_layer(features_flatten)

        return {
            metric_key: metric_function(logits, target_dict["image"][:, 0])
            for metric_key, metric_function in metrics.items()
        }

    def predict_step(self, batch: Any, batch_idx: int, **kwargs):
        input_dict, target_dict = batch
        features = self.model.forward(input_dict)["image"]
        features_flatten = F.leaky_relu(features.view((features.shape[0], -1)))
        logits = self.linear_output_layer(features_flatten)

        return {"image_features": features, "image": logits}
