from collections import defaultdict
from copy import deepcopy
from typing import Any, Dict, Union

import gate.base.utils.loggers as loggers
import hydra
import torch
import torch.nn.functional as F
import tqdm
from dotted_dict import DottedDict
from gate.configs.datamodule.base import ShapeConfig
from gate.configs.task.image_classification import TaskConfig
from gate.learners.base import LearnerModule

log = loggers.get_logger(
    __name__,
)


class EpisodicLinearLayerFineTuningScheme(LearnerModule):
    def __init__(
        self,
        optimizer_config: Dict[str, Any],
        lr_scheduler_config: Dict[str, Any],
        fine_tune_all_layers: bool = False,
        use_input_instance_norm: bool = False,
        use_cosine_similarity: bool = True,
        use_weight_norm: bool = True,
        temperature: float = 10.0,
        inner_loop_steps: int = 100,
    ):
        super(EpisodicLinearLayerFineTuningScheme, self).__init__()
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
        self.temperature = temperature

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
        self.inner_loop_model = deepcopy(self.model)
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

                model_features_flatten = model_features.view(
                    (model_features.shape[0], -1)
                )
                self.feature_embedding_shape_dict = {
                    "image": model_features_flatten.shape[1]
                }

        log.info(
            f"Built {self.__class__.__name__} "
            f"with input_shape {input_shape_dict}"
            # f"with output_shape {output_shape_dict} "
            f"{[item.shape for name, item in output_dict.items()]}"
        )

    def reset_parameters(self):
        self.output_layer_dict.reset_parameters()
        self.input_layer_dict.reset_parameters()

    def get_learner_only_params(self):

        yield from list(self.input_layer_dict.parameters()) + list(
            self.output_layer_dict.parameters()
        )

    def configure_optimizers(self):
        if self.fine_tune_all_layers:
            params = self.parameters()
        else:
            params = self.get_learner_only_params()

        return super().configure_optimizers(params=params)

    def predict(
        self,
        batch,
        backbone_module: torch.nn.Module = None,
        head_modules: Dict[str, torch.nn.Module] = None,
    ):

        output_dict = {}

        for modality_name, is_supported in self.modality_config.items():
            if is_supported:
                current_input = batch[modality_name]

                model_features = backbone_module.forward(
                    {modality_name: current_input}
                )[modality_name]

                model_features_flatten = model_features.view(
                    (model_features.shape[0], -1)
                )

                if self.use_cosine_similarity:
                    model_features_flatten = F.normalize(model_features_flatten, dim=-1)

                if "view_information" in batch:
                    model_features_flatten = torch.cat(
                        (model_features_flatten, batch["view_information"]), dim=1
                    )

                model_logits = head_modules[modality_name](model_features_flatten)

                model_logits = self.temperature * model_logits

                output_dict[modality_name] = model_logits

        return output_dict

    def forward(
        self,
        batch,
        backbone_module: torch.nn.Module = None,
        head_modules: Dict[str, torch.nn.Module] = None,
    ):
        if backbone_module is None:
            backbone_module = self.model

        if head_modules is None:
            head_modules = self.output_layer_dict

        return self.predict(
            batch, backbone_module=backbone_module, head_modules=head_modules
        )

    def step(
        self,
        batch,
        batch_idx,
        task_metrics_dict=None,
        phase_name="debug",
    ):
        torch.set_grad_enabled(True)
        self.train()
        computed_task_metrics_dict = defaultdict(list)
        opt_loss_list = []
        input_dict, target_dict = batch

        support_set_inputs = {
            "image": input_dict["image"]["support_set"].to(torch.cuda.current_device())
        }
        support_set_targets = target_dict["image"]["support_set"].to(
            torch.cuda.current_device()
        )
        query_set_inputs = {
            "image": input_dict["image"]["query_set"].to(torch.cuda.current_device())
        }
        query_set_targets = target_dict["image"]["query_set"].to(
            torch.cuda.current_device()
        )

        if "support_set_extras" in input_dict["image"]:
            support_set_view_information = torch.cat(
                [
                    input_dict["image"]["support_set_extras"][key]
                    for key in input_dict["image"]["support_set_extras"].keys()
                ],
                dim=2,
            )
            support_set_inputs["view_information"] = support_set_view_information.to(
                torch.cuda.current_device()
            )

        if "query_set_extras" in input_dict["image"]:
            query_set_view_information = torch.cat(
                [
                    input_dict["image"]["query_set_extras"][key]
                    for key in input_dict["image"]["query_set_extras"].keys()
                ],
                dim=2,
            )
            query_set_inputs["view_information"] = query_set_view_information.to(
                torch.cuda.current_device()
            )

        episodic_optimizer = None
        output_dict = defaultdict(list)
        for idx, (
            support_set_input,
            support_set_target,
            query_set_input,
            query_set_target,
        ) in enumerate(
            zip(
                support_set_inputs["image"],
                support_set_targets,
                query_set_inputs["image"],
                query_set_targets,
            )
        ):

            support_set_input = dict(image=support_set_input)
            support_set_target = dict(image=support_set_target)
            query_set_input = dict(image=query_set_input)
            query_set_target = dict(image=query_set_target)

            if "view_information" in support_set_inputs:
                support_set_input["view_information"] = support_set_inputs[
                    "view_information"
                ][idx]
            if "view_information" in query_set_inputs:
                query_set_input["view_information"] = query_set_inputs[
                    "view_information"
                ][idx]

            self.inner_loop_model.load_state_dict(self.model.state_dict())
            self.inner_loop_model.to(support_set_input["image"].device)
            self.inner_loop_model.train()

            for key, value in self.feature_embedding_shape_dict.items():
                self.output_layer_dict[key] = torch.nn.Linear(
                    in_features=value + support_set_input["view_information"].shape[1]
                    if "view_information" in support_set_input
                    else value,
                    out_features=int(torch.max(support_set_target[key]) + 1),
                    bias=False,
                )
                self.output_layer_dict[key].to(support_set_input[key].device)
                if self.use_weight_norm:
                    self.output_layer_dict[key] = torch.nn.utils.weight_norm(
                        self.output_layer_dict[key]
                    )

            self.output_layer_dict.train()
            non_feature_embedding_params = list(self.output_layer_dict.parameters())

            params = (
                (
                    list(self.inner_loop_model.parameters())
                    + non_feature_embedding_params
                )
                if self.fine_tune_all_layers
                else self.output_layer_dict.parameters()
            )

            if episodic_optimizer:
                del episodic_optimizer

            episodic_optimizer = hydra.utils.instantiate(
                config=self.inner_loop_optimizer_config,
                params=params,
            )

            with tqdm.tqdm(total=self.inner_loop_steps) as pbar:
                for step_idx in range(self.inner_loop_steps):
                    current_output_dict = self.forward(
                        support_set_input,
                        backbone_module=self.inner_loop_model,
                        head_modules=self.output_layer_dict,
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

                    episodic_optimizer.zero_grad()

                    support_set_loss.backward()

                    episodic_optimizer.step()

                    with torch.no_grad():
                        current_output_dict = self.forward(
                            query_set_input,
                            backbone_module=self.inner_loop_model,
                            head_modules=self.output_layer_dict,
                        )

                        (
                            query_set_loss,
                            computed_task_metrics_dict,
                        ) = self.compute_metrics(
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

                    pbar.update(1)
                    pbar.set_description(
                        f"Support Set Loss: {support_set_loss}, "
                        f"Query Set Loss: {query_set_loss}"
                    )

                for key, value in current_output_dict.items():
                    output_dict[key].append(value)

            opt_loss_list.append(query_set_loss)
        self.episode_idx += 1

        for key, value in output_dict.items():
            output_dict[key] = torch.stack(value, dim=0)

        return (
            output_dict,
            computed_task_metrics_dict,
            torch.mean(torch.stack(opt_loss_list)),
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
                    computed_metrics_dict[
                        f"{phase_name}/episode_{episode_idx}/{set_name}_{metric_key}"
                    ].append(metric_value.detach().cpu())

                    if step_idx == self.inner_loop_steps - 1:
                        computed_metrics_dict[
                            f"{phase_name}/{set_name}_{metric_key}"
                        ].append(metric_value.detach().cpu())

        for (
            metric_key,
            metric_function,
        ) in learner_metrics_dict.items():
            for output_name, output_value in output_dict.items():
                metric_value = metric_function(
                    output_dict[output_name],
                    target_dict[output_name],
                )

                computed_metrics_dict[
                    f"{phase_name}/episode_{episode_idx}/{set_name}_{metric_key}"
                ].append(metric_value.detach().cpu())

                opt_loss_list.append(metric_value)

                if step_idx == self.inner_loop_steps - 1:
                    computed_metrics_dict[
                        f"{phase_name}/{set_name}_{metric_key}"
                    ].append(metric_value.detach().cpu())

        return torch.stack(opt_loss_list).mean(), computed_metrics_dict

    def training_step(
        self, batch, batch_idx, task_metrics_dict, top_level_pl_module=None
    ):
        output_dict, computed_task_metrics_dict, opt_loss = self.step(
            batch=batch,
            batch_idx=batch_idx,
            task_metrics_dict=task_metrics_dict,
            phase_name="training",
        )

        computed_task_metrics_dict["training/opt_loss"] = opt_loss
        output_dict["loss"] = opt_loss

        return opt_loss, computed_task_metrics_dict

    def validation_step(
        self, batch, batch_idx, task_metrics_dict, top_level_pl_module=None
    ):

        output_dict, computed_task_metrics_dict, opt_loss = self.step(
            batch=batch,
            batch_idx=batch_idx,
            task_metrics_dict=task_metrics_dict,
            phase_name="validation",
        )

        computed_task_metrics_dict["validation/opt_loss"] = opt_loss

        return opt_loss, computed_task_metrics_dict

    def test_step(self, batch, batch_idx, task_metrics_dict, top_level_pl_module=None):
        output_dict, computed_task_metrics_dict, opt_loss = self.step(
            batch=batch,
            batch_idx=batch_idx,
            task_metrics_dict=task_metrics_dict,
            phase_name="test",
        )

        computed_task_metrics_dict["test/opt_loss"] = opt_loss

        return opt_loss, computed_task_metrics_dict

    def predict_step(self, batch: Any, batch_idx: int, **kwargs):
        input_dict = batch
        return self.forward(input_dict)
