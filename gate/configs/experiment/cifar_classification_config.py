# import os
# from dataclasses import dataclass, MISSING, field
# from typing import Any, Optional, List
#
# from hydra.core.config_store import ConfigStore
# from omegaconf import OmegaConf
#
# defaults = [
#     {"callbacks": "wandb"},
#     {"logger": "wandb"},
#     {"model": "timm-image-resnet18"},
#     {"learner": "FullModelFineTuning"},
#     {"train_eval_agent": "base"},
#     {"trainer": "base"},
#     {"mode": "default"},
# ]
#
# cifar10_overrides = [
#     {"task": "cifar10"},
#     {"datamodule": "CIFAR10StandardClassification"},
# ]
#
#
# cifar100_overrides = [
#     {"task": "cifar100"},
#     {"datamodule": "CIFAR100StandardClassification"},
# ]
#
#
# def add_task_configs(config_store: ConfigStore):
#     config_store.store(
#         group="experiment",
#         name="cifar10-standard_classification",
#         node=CIFAR10Config,
#     )
#
#     config_store.store(
#         group="experiment",
#         name="cifar100-standard_classification",
#         node=CIFAR100Config,
#     )
#
#     return config_store
