from collections import defaultdict
from copy import deepcopy
from typing import Any, Dict, Union, Optional

import higher
import hydra
import torch
import torch.nn.functional as F
import tqdm
from dotted_dict import DottedDict
from torch import nn

import gate.base.utils.loggers as loggers
from gate.configs.datamodule.base import ShapeConfig
from gate.configs.task.image_classification import TaskConfig
from gate.learners.base import LearnerModule

log = loggers.get_logger(
    __name__,
)


class DynamicWeightLinear(nn.Module):
    def __init__(
        self,
        weights: torch.Tensor,
        use_cosine_similarity: bool = False,
        bias: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.weight = weights
        self.bias = bias
        self.use_cosine_similarity = use_cosine_similarity

    def forward(self, x):
        x = x["image"]
        if len(x.shape) > 2:
            x = x.view(x.shape[0], -1)

        if self.use_cosine_similarity:
            x = F.normalize(x, dim=-1)

        return {"image": F.linear(x, self.weight, self.bias)}


class AdaptivePool2DFlatten(nn.Module):
    def __init__(self, pool_type: str = "avg", output_size: int = 1):
        super().__init__()
        self.pool_type = pool_type
        self.output_size = output_size

    def forward(self, x):
        x = x["image"]
        if self.pool_type == "avg":
            x = F.adaptive_avg_pool2d(x, self.output_size)
        elif self.pool_type == "max":
            x = F.adaptive_max_pool2d(x, self.output_size)
        else:
            raise ValueError(f"Unknown pool type {self.pool_type}")

        return {"image": x.view(x.shape[0], -1)}


class EpisodicMAML(LearnerModule):
    def __init__(
        self,
        optimizer_config: Dict[str, Any],
        lr_scheduler_config: Dict[str, Any],
        fine_tune_all_layers: bool = False,
        use_input_instance_norm: bool = False,
        use_cosine_similarity: bool = False,
        use_weight_norm: bool = False,
        temperature: float = 10.0,
        inner_loop_steps: int = 5,
        manual_optimization: bool = True,
    ):
        super(EpisodicMAML, self).__init__()
        self.output_layer_dict = torch.nn.ModuleDict()
        self.input_layer_dict = torch.nn.ModuleDict()
        self.optimizer_config = optimizer_config.outer_loop_optimizer_config
        self.lr_scheduler_config = lr_scheduler_config.outer_loop_lr_scheduler_config
        self.inner_loop_optimizer_config = optimizer_config.inner_loop_optimizer_config
        self.inner_loop_lr_scheduler_config = (
            lr_scheduler_config.inner_loop_lr_scheduler_config
        )
        self.fine_tune_all_layers = fine_tune_all_layers
        self.use_input_instance_norm = use_input_instance_norm
        self.inner_loop_steps = inner_loop_steps
        self.use_cosine_similarity = use_cosine_similarity
        self.use_weight_norm = use_weight_norm
        self.manual_optimization = manual_optimization
        self.temperature = nn.Parameter(torch.tensor(temperature), requires_grad=True)

        self.learner_metrics_dict = {"loss": F.cross_entropy}

    def build(
        self,
        model: torch.nn.Module,
        task_config: TaskConfig,
        modality_config: Union[DottedDict, Dict],
        input_shape_dict: Union[ShapeConfig, Dict, DottedDict],
        output_shape_dict: Union[ShapeConfig, Dict, DottedDict],
    ):
        self.input_shape_dict = (
            input_shape_dict.__dict__
            if isinstance(input_shape_dict, ShapeConfig)
            else input_shape_dict
        )

        self.output_shape_dict = (
            output_shape_dict.__dict__
            if isinstance(output_shape_dict, ShapeConfig)
            else output_shape_dict
        )

        self.modality_config = (
            modality_config.__dict__
            if isinstance(modality_config, DottedDict)
            else modality_config
        )

        self.model: torch.nn.Module = model
        self.task_config = task_config
        self.episode_idx = 0

        output_dict = {}
        for modality_name, is_supported in self.modality_config.items():
            if is_supported:
                input_dummy_x = torch.randn(
                    [2] + list(self.input_shape_dict[modality_name]["shape"].values())
                )

                model_features = self.model.forward({modality_name: input_dummy_x})[
                    modality_name
                ]

                self.pooling_layer = AdaptivePool2DFlatten(output_size=2)

                model_features_pooled_flatten = self.pooling_layer(model_features)

                self.feature_embedding_shape_dict = {
                    "image": model_features_pooled_flatten.shape[1]
                }

                for key, value in self.feature_embedding_shape_dict.items():
                    self.output_layer_dict[key] = nn.ModuleDict(
                        {
                            "pre_pred_layer": torch.nn.Linear(
                                in_features=value,
                                out_features=value,
                                bias=True,
                            ),
                            "pred_layer": torch.nn.Linear(
                                in_features=value,
                                out_features=1,
                                bias=True,
                            ),
                        }
                    )

                    if self.use_weight_norm:
                        self.output_layer_dict[key] = torch.nn.utils.weight_norm(
                            self.output_layer_dict[key]
                        )

        log.info(
            f"Built {self.__class__.__name__} "
            f"with input_shape {input_shape_dict}"
            # f"with output_shape {output_shape_dict} "
            f"{[item.shape for name, item in output_dict.items()]}"
        )

    def get_learner_only_params(self):

        yield from list(self.input_layer_dict.parameters()) + list(
            self.output_layer_dict.parameters()
        )

    def configure_optimizers(self):
        outer_loop_parameters = (
            self.parameters()
            if self.fine_tune_all_layers
            else list(self.output_layer_dict.parameters()) + [self.temperature]
        )

        return super().configure_optimizers(params=outer_loop_parameters)

    def predict(
        self,
        batch,
        model: torch.nn.Module,
    ):

        output_dict = {}

        for modality_name, is_supported in self.modality_config.items():
            if is_supported:
                current_input = batch[modality_name]

                model_logits = model.forward({modality_name: current_input})[
                    modality_name
                ]

                model_logits = self.temperature * model_logits

                output_dict[modality_name] = model_logits

        return output_dict

    def forward(
        self,
        batch,
        model: torch.nn.Module,
    ):

        return self.predict(batch, model=model)

    def step(
        self, batch, batch_idx, task_metrics_dict=None, phase_name="debug", train=True
    ):
        torch.set_grad_enabled(True)
        self.train()
        computed_task_metrics_dict = defaultdict(list)
        opt_loss_list = []
        input_dict, target_dict = batch
        support_set_inputs = input_dict["image"]["support_set"].to(
            torch.cuda.current_device()
        )
        support_set_targets = target_dict["image"]["support_set"].to(
            torch.cuda.current_device()
        )
        query_set_inputs = input_dict["image"]["query_set"].to(
            torch.cuda.current_device()
        )
        query_set_targets = target_dict["image"]["query_set"].to(
            torch.cuda.current_device()
        )
        self.to(torch.cuda.current_device())
        self.train()

        episodic_optimizer = None
        output_dict = defaultdict(list)
        for (
            support_set_input,
            support_set_target,
            query_set_input,
            query_set_target,
        ) in zip(
            support_set_inputs,
            support_set_targets,
            query_set_inputs,
            query_set_targets,
        ):

            support_set_input = dict(image=support_set_input)
            support_set_target = dict(image=support_set_target)
            query_set_input = dict(image=query_set_input)
            query_set_target = dict(image=query_set_target)

            classifier_weights = self.output_layer_dict["image"][
                "pred_layer"
            ].weight.repeat([max(support_set_target["image"]) + 1, 1])

            classifier_bias = self.output_layer_dict["image"]["pred_layer"].bias.repeat(
                [max(support_set_target["image"]) + 1]
            )

            classifier_weights = nn.Parameter(classifier_weights, requires_grad=True)
            classifier_bias = nn.Parameter(classifier_bias, requires_grad=True)

            if episodic_optimizer:
                del episodic_optimizer

            pre_classifier = DynamicWeightLinear(
                weights=self.output_layer_dict["image"]["pre_pred_layer"].weight,
                bias=self.output_layer_dict["image"]["pre_pred_layer"].bias,
                use_cosine_similarity=self.use_cosine_similarity,
            )

            classifer = DynamicWeightLinear(
                weights=classifier_weights,
                bias=classifier_bias,
                use_cosine_similarity=self.use_cosine_similarity,
            )

            model = (
                torch.nn.Sequential(
                    self.model,
                    self.pooling_layer,
                    pre_classifier,
                    classifer,
                )
                if self.fine_tune_all_layers
                else torch.nn.Sequential(self.pooling_layer, pre_classifier, classifer)
            )
            inner_loop_params = model.parameters()

            if batch_idx == 0:
            
                log.info(f"Inner loop params:")
                for name, param in model.named_parameters():
                    if param.requires_grad:
                        log.info(f"{name}, {param.data.shape}")



            episodic_optimizer = hydra.utils.instantiate(
                config=self.inner_loop_optimizer_config,
                params=inner_loop_params,
            )

            track_higher_grads = True if train else False

            if not self.fine_tune_all_layers:
                for modality_name, is_supported in self.modality_config.items():
                    if is_supported:
                        support_set_input[modality_name] = self.model.forward(
                            {modality_name: support_set_input[modality_name]}
                        )[modality_name].detach()

                        query_set_input[modality_name] = self.model.forward(
                            {modality_name: query_set_input[modality_name]}
                        )[modality_name].detach()


            with higher.innerloop_ctx(
                model,
                episodic_optimizer,
                copy_initial_weights=False,
                track_higher_grads=track_higher_grads,
            ) as (inner_loop_model, inner_loop_optimizer):

                for step_idx in range(self.inner_loop_steps):
                    current_output_dict = self.forward(
                        support_set_input, model=inner_loop_model
                    )

                    (
                        support_set_loss,
                        computed_task_metrics_dict,
                    ) = self.compute_metrics(
                        phase_name=phase_name,
                        set_name="support_set",
                        output_dict=current_output_dict,
                        target_dict=support_set_target,
                        task_metrics_dict=task_metrics_dict,
                        learner_metrics_dict=self.learner_metrics_dict,
                        episode_idx=self.episode_idx,
                        step_idx=step_idx,
                        computed_metrics_dict=computed_task_metrics_dict,
                    )

                    inner_loop_optimizer.step(support_set_loss)

                current_output_dict = self.forward(
                    query_set_input,
                    model=inner_loop_model,
                )

                query_set_loss, computed_task_metrics_dict = self.compute_metrics(
                    phase_name=phase_name,
                    set_name="query_set",
                    output_dict=current_output_dict,
                    target_dict=query_set_target,
                    task_metrics_dict=task_metrics_dict,
                    learner_metrics_dict=self.learner_metrics_dict,
                    episode_idx=self.episode_idx,
                    step_idx=step_idx,
                    computed_metrics_dict=computed_task_metrics_dict,
                )

                for key, value in current_output_dict.items():
                    output_dict[key].append(value)

                opt_loss_list.append(query_set_loss)

                self.episode_idx += 1

        for key, value in output_dict.items():
            output_dict[key] = torch.stack(value, dim=0)

        return dict(
            output_dict=output_dict,
            computed_task_metrics_dict=computed_task_metrics_dict,
            opt_loss=torch.mean(torch.stack(opt_loss_list)),
        )

    def compute_metrics(
        self,
        phase_name,
        set_name,
        output_dict,
        target_dict,
        task_metrics_dict,
        learner_metrics_dict,
        episode_idx,
        step_idx,
        computed_metrics_dict,
    ):
        """
        Compute metrics for the given phase and set.

        Args:
            phase_name (str): The phase name.
            set_name (str): The set name.
            output_dict (dict): The output dictionary.
            target_dict (dict): The target dictionary.
            task_metrics_dict (dict): The task metrics dictionary.
            learner_metrics_dict (dict): The learner metrics dictionary.

        Returns:
            dict: The computed metrics.
        """
        opt_loss_list = []
        if task_metrics_dict is not None:
            for metric_key, metric_function in task_metrics_dict.items():
                for output_name, output_value in output_dict.items():

                    metric_value = metric_function(
                        output_dict[output_name].clone().detach().cpu(),
                        target_dict[output_name].clone().detach().cpu(),
                    )

                    if phase_name != "training":
                        computed_metrics_dict[
                            f"{phase_name}/episode_{episode_idx}/{set_name}/{metric_key}"
                        ].append(metric_value.detach().cpu())

                    if step_idx == self.inner_loop_steps - 1:
                        computed_metrics_dict[
                            f"{phase_name}/{set_name}/{metric_key}"
                        ].append(metric_value.detach().cpu())

                    if set_name == "query_set":
                        computed_metrics_dict[f"{phase_name}/accuracy"].append(
                            metric_value.detach().cpu()
                        )

        for (
            metric_key,
            metric_function,
        ) in learner_metrics_dict.items():
            for output_name, output_value in output_dict.items():
                metric_value = metric_function(
                    output_dict[output_name],
                    target_dict[output_name],
                )

                if phase_name != "training":
                    computed_metrics_dict[
                        f"{phase_name}/episode_{episode_idx}/{set_name}/{metric_key}"
                    ].append(metric_value.detach().cpu())

                opt_loss_list.append(metric_value)

                if set_name == "query_set":
                    computed_metrics_dict[f"{phase_name}/loss"].append(
                        metric_value.detach().cpu()
                    )

        return torch.stack(opt_loss_list).mean(), computed_metrics_dict

    def training_step(self, batch, batch_idx, task_metrics_dict, top_level_pl_module):

        optimizers = top_level_pl_module.optimizers()

        step_dict = self.step(
            batch=batch,
            batch_idx=batch_idx,
            task_metrics_dict=task_metrics_dict,
            phase_name="training",
        )

        step_dict["computed_task_metrics_dict"]["training/opt_loss"] = step_dict[
            "opt_loss"
        ]
        step_dict["output_dict"]["loss"] = step_dict["opt_loss"]

        optimizers.zero_grad()
        top_level_pl_module.manual_backward(step_dict["opt_loss"])
        optimizers.step()

        return step_dict["opt_loss"], step_dict["computed_task_metrics_dict"]

    def validation_step(
        self, batch, batch_idx, task_metrics_dict, top_level_pl_module=None
    ):

        step_dict = self.step(
            batch=batch,
            batch_idx=batch_idx,
            task_metrics_dict=task_metrics_dict,
            phase_name="validation",
        )

        step_dict["computed_task_metrics_dict"]["validation/opt_loss"] = step_dict[
            "opt_loss"
        ]
        step_dict["output_dict"]["loss"] = step_dict["opt_loss"]

        return step_dict["opt_loss"], step_dict["computed_task_metrics_dict"]

    def test_step(self, batch, batch_idx, task_metrics_dict, top_level_pl_module=None):
        step_dict = self.step(
            batch=batch,
            batch_idx=batch_idx,
            task_metrics_dict=task_metrics_dict,
            phase_name="test",
        )

        step_dict["computed_task_metrics_dict"]["test/opt_loss"] = step_dict["opt_loss"]
        step_dict["output_dict"]["loss"] = step_dict["opt_loss"]

        return step_dict["opt_loss"], step_dict["computed_task_metrics_dict"]

    def predict_step(self, batch: Any, batch_idx: int, **kwargs):
        return self.step(
            batch=batch,
            batch_idx=batch_idx,
            task_metrics_dict=None,
            phase_name="predict",
        )
