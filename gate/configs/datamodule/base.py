import multiprocessing
from dataclasses import MISSING, dataclass

# ------------------------------------------------------------------------------
# General configs
# from gate.datasets.data_utils import collate_fn_replace_corrupted
from typing import Any, Dict, List, Optional

# ------------------------------------------------------------------------------
# data loader configs
from gate.configs import get_module_import_path
from gate.configs.string_variables import (
    ADDITIONAL_INPUT_TRANSFORMS,
    ADDITIONAL_TARGET_TRANSFORMS,
    BATCH_SIZE,
)

# ------------------------------------------------------------------------------
# Config rules:

# Structured Configs use Python dataclasses to describe your configuration
# structure and types. They enable:
#
# - Runtime type checking as you compose or mutate your config
# - Static type checking when using static type checkers (mypy, PyCharm, etc.)
#
# Structured Configs supports:
# - Primitive types (int, bool, float, str, Enums)
# - Nesting of Structured Configs
# - Containers (List and Dict) containing primitives or Structured Configs
# - Optional fields
#
# Structured Configs Limitations:
# - Union types are not supported (except Optional)
# - User methods are not supported
#
# There are two primary patterns for using Structured configs
# - As a config, in place of configuration files (often a starting place)
# - As a config schema validating configuration files (better for complex use cases)
#
# With both patterns, you still get everything Hydra has to offer
# (config composition, Command line overrides etc).


@dataclass
class DataLoaderConfig:
    seed: int = 0
    train_batch_size: int = BATCH_SIZE
    val_batch_size: int = BATCH_SIZE
    test_batch_size: int = BATCH_SIZE
    num_workers: int = multiprocessing.cpu_count()
    pin_memory: bool = True
    train_drop_last: bool = False
    eval_drop_last: bool = False
    train_shuffle: bool = True
    eval_shuffle: bool = False
    prefetch_factor: int = 2
    persistent_workers: bool = True


@dataclass
class ShapeConfig:
    """
    Modality configuration for the types of processing a model can do.
    """

    image: Optional[Any] = None
    audio: Optional[Any] = None
    text: Optional[Any] = None
    video: Optional[Any] = None


# ------------------------------------------------------------------------------

# sourcery skip: remove-redundant-fstring
@dataclass
class TransformConfig:
    _target_: str = MISSING
    additional_transforms: Optional[List[Any]] = None
