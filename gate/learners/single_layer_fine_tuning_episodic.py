from collections import defaultdict
from copy import deepcopy
from typing import Any, Dict, Union

import hydra
import torch
import torch.nn.functional as F
import tqdm
from dotted_dict import DottedDict

import gate.base.utils.loggers as loggers
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
        inner_loop_steps: int = 100,
    ):
        super(EpisodicLinearLayerFineTuningScheme, self).__init__()
        self.output_layer_dict = torch.nn.ModuleDict()
        self.input_layer_dict = torch.nn.ModuleDict()
        self.optimizer_config = optimizer_config.outer_loop_optimizer_config
        self.lr_scheduler_config = (
            lr_scheduler_config.outer_loop_lr_scheduler_config
        )
        self.inner_loop_optimizer_config = (
            optimizer_config.inner_loop_optimizer_config
        )
        self.inner_loop_lr_scheduler_config = (
            lr_scheduler_config.inner_loop_lr_scheduler_config
        )
        self.fine_tune_all_layers = fine_tune_all_layers
        self.use_input_instance_norm = use_input_instance_norm
        self.inner_loop_steps = inner_loop_steps

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
                    [2]
                    + list(
                        self.input_shape_dict[modality_name]["shape"].values()
                    )
                )

                if modality_name == "image":
                    self.input_layer_dict[
                        f"{modality_name}_input_adaptor"
                    ] = torch.nn.InstanceNorm2d(
                        num_features=input_dummy_x.shape[1],
                        affine=True,
                        track_running_stats=True,
                    )

                    input_dummy_x = self.input_layer_dict[
                        f"{modality_name}_input_adaptor"
                    ](input_dummy_x)
                elif modality_name == "audio":
                    self.input_layer_dict[
                        f"{modality_name}_input_adaptor"
                    ] = torch.nn.InstanceNorm1d(
                        num_features=input_dummy_x.shape[1],
                        affine=True,
                        track_running_stats=True,
                    )

                    input_dummy_x = self.input_layer_dict[
                        f"{modality_name}_input_adaptor"
                    ](input_dummy_x)
                elif modality_name == "video":
                    b, s, c, w, h = input_dummy_x.shape
                    input_dummy_x = input_dummy_x.view(b * s, c, w, h)
                    input_dummy_x = self.input_layer_dict[
                        f"{modality_name}_input_adaptor"
                    ](input_dummy_x)
                    input_dummy_x = input_dummy_x.view(b, s, c, w, h)

                model_features = self.model.forward(
                    {modality_name: input_dummy_x}
                )[modality_name]

                model_features_flatten = model_features.view(
                    (model_features.shape[0], -1)
                )
                self.feature_embedding_shape_dict = {
                    "image": model_features_flatten.shape[1]
                }
                self.output_layer_dict[modality_name] = torch.nn.Linear(
                    model_features_flatten.shape[1],
                    self.output_shape_dict[modality_name]["num_classes"],
                )

                logits = self.output_layer_dict[modality_name](
                    model_features_flatten
                )

                output_dict[modality_name] = logits

        log.info(
            f"Built {self.__class__.__name__} "
            f"with input_shape {input_shape_dict}"
            f"with output_shape {output_shape_dict} "
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

    def get_feature_embeddings(
        self, batch, model_module: torch.nn.Module = None
    ):

        output_dict = {}

        for modality_name, is_supported in self.modality_config.items():
            if is_supported:
                current_input = batch[modality_name]
                if self.use_input_instance_norm:
                    current_input = self.input_layer_dict[
                        f"{modality_name}_input_adaptor"
                    ](current_input)

                model_features = model_module.forward(
                    {modality_name: current_input}
                )[modality_name]

                model_features_flatten = model_features.view(
                    (model_features.shape[0], -1)
                )
                output_dict[modality_name] = model_features_flatten

        return output_dict

    def forward(self, batch, model_module=None):
        if model_module is None:
            model_module = self.model

        return self.get_feature_embeddings(batch, model_module=model_module)

    def step(
        self,
        batch,
        batch_idx,
        task_metrics_dict,
        learner_metrics_dict,
        phase_name,
    ):
        computed_task_metrics_dict = {}
        opt_loss_list = []
        input_dict, target_dict = batch
        support_set_inputs = input_dict["image"]["support_set"]
        support_set_targets = target_dict["image"]["support_set"]
        query_set_inputs = input_dict["image"]["query_set"]
        query_set_targets = target_dict["image"]["query_set"]
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

            self.inner_loop_model.load_state_dict(self.model.state_dict())

            for key, value in self.input_layer_dict.items():
                value.reset_parameters()

            for key, value in self.feature_embedding_shape_dict.items():
                self.output_layer_dict[key] = torch.nn.Linear(
                    value,
                    int(torch.max(support_set_target[key]) + 1),
                )
                self.output_layer_dict[key].to(support_set_inputs)

            non_feature_embedding_params = list(
                self.output_layer_dict.parameters()
            )

            if self.use_input_instance_norm:
                non_feature_embedding_params += list(
                    self.input_layer_dict.parameters()
                )

            params = (
                (
                    list(self.inner_loop_model.parameters())
                    + non_feature_embedding_params
                )
                if self.fine_tune_all_layers
                else non_feature_embedding_params
            )

            if episodic_optimizer:
                del episodic_optimizer

            episodic_optimizer = hydra.utils.instantiate(
                config=self.inner_loop_optimizer_config,
                params=params,
            )

            with tqdm.tqdm(total=self.inner_loop_steps) as pbar:
                for _ in range(self.inner_loop_steps):
                    current_output_dict = self.forward(
                        support_set_input, model_module=self.inner_loop_model
                    )

                    (
                        support_set_loss,
                        current_computed_metrics,
                    ) = self.compute_metrics(
                        phase_name=phase_name,
                        set_name="support_set",
                        output_dict=current_output_dict,
                        target_dict=support_set_target,
                        task_metrics_dict=task_metrics_dict,
                        learner_metrics_dict=learner_metrics_dict,
                        episode_idx=self.episode_idx,
                    )

                    computed_task_metrics_dict.update(current_computed_metrics)

                    episodic_optimizer.zero_grad()
                    support_set_loss.backward()
                    episodic_optimizer.step()

                    self.zero_grad()

                    pbar.update(1)
                    pbar.set_description(
                        f"Support Set Loss: {support_set_loss}, "
                    )
                with torch.no_grad():
                    current_output_dict = self.forward(
                        query_set_input, model_module=self.inner_loop_model
                    )

                    (
                        query_set_loss,
                        current_computed_metrics,
                    ) = self.compute_metrics(
                        phase_name=phase_name,
                        set_name="query_set",
                        output_dict=current_output_dict,
                        target_dict=query_set_target,
                        task_metrics_dict=task_metrics_dict,
                        learner_metrics_dict=learner_metrics_dict,
                        episode_idx=self.episode_idx,
                    )
                    computed_task_metrics_dict.update(current_computed_metrics)

                for key, value in current_output_dict.items():
                    output_dict[key].append(value)

            opt_loss_list.append(query_set_loss)
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
        computed_task_metrics_dict = defaultdict(list)
        opt_loss_list = []
        for metric_key, metric_function in task_metrics_dict.items():
            for output_name, output_value in output_dict.items():
                metric_value = metric_function(
                    output_dict[output_name],
                    target_dict[output_name],
                )
                computed_task_metrics_dict[
                    f"{phase_name}/episode_{episode_idx}/"
                    f"{set_name}_{metric_key}"
                ].append(metric_value.detach().cpu().numpy())

        for (
            metric_key,
            metric_function,
        ) in learner_metrics_dict.items():
            for output_name, output_value in output_dict.items():
                metric_value = metric_function(
                    output_dict[output_name],
                    target_dict[output_name],
                )
                computed_task_metrics_dict[
                    f"{phase_name}/episode_{episode_idx}/"
                    f"{set_name}_{metric_key}"
                ].append(metric_value.detach().cpu().numpy())

                opt_loss_list.append(metric_value)

        return torch.stack(opt_loss_list).mean(), computed_task_metrics_dict

    def training_step(self, batch, batch_idx, task_metrics_dict):
        output_dict, computed_task_metrics_dict, opt_loss = self.step(
            batch,
            batch_idx,
            task_metrics_dict,
            self.learner_metrics_dict,
            phase_name="training",
        )

        computed_task_metrics_dict["training/opt_loss"] = opt_loss
        output_dict["loss"] = opt_loss

        return opt_loss, computed_task_metrics_dict

    def validation_step(self, batch, batch_idx, task_metrics_dict):

        output_dict, computed_task_metrics_dict, opt_loss = self.step(
            batch,
            batch_idx,
            task_metrics_dict,
            self.learner_metrics_dict,
            phase_name="validation",
        )

        computed_task_metrics_dict["validation/opt_loss"] = opt_loss

        return opt_loss, computed_task_metrics_dict

    def test_step(self, batch, batch_idx, task_metrics_dict):
        output_dict, computed_task_metrics_dict, opt_loss = self.step(
            batch,
            batch_idx,
            task_metrics_dict,
            self.learner_metrics_dict,
            phase_name="test",
        )

        computed_task_metrics_dict["test/opt_loss"] = opt_loss

        return opt_loss, computed_task_metrics_dict

    def predict_step(self, batch: Any, batch_idx: int, **kwargs):
        input_dict = batch
        return self.forward(input_dict)
