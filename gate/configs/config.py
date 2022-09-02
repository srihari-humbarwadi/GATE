import inspect
import os
from dataclasses import MISSING, dataclass, field
from typing import Any, List, Optional

from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf

from gate.base.utils.loggers import get_logger
from gate.configs.callbacks import add_lightning_callback_configs
from gate.configs.datamodule import add_datamodule_configs
from gate.configs.datasets import add_transform_configs
from gate.configs.hydra import add_hydra_configs
from gate.configs.learner import (
    add_learner_configs,
    add_learning_scheduler_configs,
    add_optimizer_configs,
)
from gate.configs.logger import add_logger_configs
from gate.configs.mode import add_mode_configs
from gate.configs.model import add_model_configs
from gate.configs.task import add_task_configs
from gate.configs.train_eval_agent import add_train_eval_agent_configs
from gate.configs.trainer import add_trainer_configs

log = get_logger(__name__)

defaults = [
    {"callbacks": "wandb"},
    {"logger": "wandb"},
    {"model": "timm-image-resnet18"},
    {"learner": "EpisodicPrototypicalNetwork"},
    {"datamodule": "OmniglotFewShotClassification"},
    {"task": "VariableClassClassification"},
    {"train_eval_agent": "base"},
    {"trainer": "base"},
    {"mode": "base"},
    {"additional_input_transforms": "base"},
    {"additional_target_transforms": "base"},
    {"hydra": "custom_logging_path"},
    {"_self_": None},
]

overrides = []

OmegaConf.register_new_resolver("last_bit", lambda x: x.split(".")[-1])
OmegaConf.register_new_resolver("lower", lambda x: x.lower())
OmegaConf.register_new_resolver(
    "remove_redundant_words",
    lambda x: x.replace("scheme", "").replace("module", "").replace("config", ""),
)


@dataclass
class Config:
    _self_: Any = MISSING
    callbacks: Any = MISSING
    logger: Any = MISSING
    model: Any = MISSING
    learner: Any = MISSING
    datamodule: Any = MISSING
    additional_input_transforms: Any = MISSING
    additional_target_transforms: Any = MISSING
    task: Any = MISSING
    train_eval_agent: Any = MISSING
    trainer: Any = MISSING
    mode: Any = MISSING
    hydra: Any = MISSING

    resume: bool = True
    checkpoint_path: Optional[str] = None
    # pretty print config at the start of the run using Rich library
    print_config: bool = True

    # disable python warnings if they annoy you
    ignore_warnings: bool = True
    logging_level: str = "INFO"
    # evaluate on test set, using best model weights achieved during training
    # lightning chooses best weights based on metric specified in checkpoint
    # callback
    test_after_training: bool = True

    batch_size: Optional[int] = None
    # seed for random number generators in learn2learn_hub, numpy and python.random
    seed: int = 0
    num_train_samples: int = 25000
    # top level argument that sets all the downstream configs to run an
    # experiment on this many iterations

    # path to original working directory
    # hydra hijacks working directory by changing it to the new log directory
    # so it's useful to have this path as a special variable
    # https://hydra.cc/docs/next/tutorials/basic/running_your_app/working_directory
    root_experiment_dir: str = os.environ["EXPERIMENTS_DIR"]
    # path to folder with data
    data_dir: str = os.environ["DATASET_DIR"]
    defaults: List[Any] = field(default_factory=lambda: defaults)
    overrides: List[Any] = field(default_factory=lambda: overrides)
    name: str = (
        "${remove_redundant_words:${lower:${last_bit:"
        "${datamodule.dataset_config._target_}}-${last_bit:${task._target_}}-"
        "${last_bit:${learner._target_}}-${model.model_name_to_download}-"
        "${seed}}}"
    )
    current_experiment_dir: str = "${root_experiment_dir}/${name}"
    code_dir: str = "${hydra:runtime.cwd}"


def collect_config_store():
    config_store = ConfigStore.instance()
    config_store.store(name="config", node=Config)
    config_store = add_hydra_configs(config_store)
    config_store = add_trainer_configs(config_store)
    config_store = add_model_configs(config_store)
    config_store = add_datamodule_configs(config_store)
    config_store = add_learner_configs(config_store)
    config_store = add_optimizer_configs(config_store)
    config_store = add_learning_scheduler_configs(config_store)
    config_store = add_task_configs(config_store)
    config_store = add_mode_configs(config_store)
    config_store = add_train_eval_agent_configs(config_store)
    config_store = add_logger_configs(config_store)
    config_store = add_lightning_callback_configs(config_store)
    config_store = add_transform_configs(config_store)
    return config_store
