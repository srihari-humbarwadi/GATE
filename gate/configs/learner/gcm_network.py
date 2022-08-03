from dataclasses import MISSING, dataclass
from typing import Any, Callable, Dict

from gate.configs import get_module_import_path
from gate.configs.learner.learning_rate_scheduler_config import (
    CosineAnnealingLRConfig,
    LRSchedulerConfig,
)
from gate.configs.learner.linear_layer_fine_tuning import LearnerConfig
from gate.configs.learner.optimizer_config import (
    AdamOptimizerConfig,
    BaseOptimizerConfig,
)
from gate.learners.GCM import ConditionalGenerativeContrastiveModelling
from gate.model_blocks.auto_builder_modules.gcm_blocks import HeadConv, HeadMLP


@dataclass
class HeadConfig:
    output_activation_fn: Any = None
    _target_: str = "HeadClassPath"


@dataclass
class HeadConvConfig(HeadConfig):
    num_output_filters: int = 512
    num_layers: int = 2
    num_hidden_filters: int = 512
    input_avg_pool_size: int = 5
    _target_: str = get_module_import_path(HeadConv)


@dataclass
class HeadMLPConfig(HeadConfig):
    num_output_filters: int = 512
    num_layers: int = 2
    num_hidden_filters: int = 512
    input_avg_pool_size: int = 2
    _target_: str = get_module_import_path(HeadMLP)


@dataclass
class ConditionalGenerativeContrastiveModellingConfig(LearnerConfig):
    _target_: str = get_module_import_path(
        ConditionalGenerativeContrastiveModelling
    )
    fine_tune_all_layers: bool = True
    use_input_instance_norm: bool = True
    head_num_layers: int = 2
    head_num_hidden_filters: int = 512
    head_num_output_filters: int = 512
    optimizer_config: BaseOptimizerConfig = AdamOptimizerConfig(lr=1e-3)
    lr_scheduler_config: LRSchedulerConfig = CosineAnnealingLRConfig()
    mean_head_config: HeadConfig = HeadMLPConfig()
    precision_head_config: HeadConfig = HeadMLPConfig()


@dataclass
class ConditionalGenerativeContrastiveModellingMLPConfig(
    ConditionalGenerativeContrastiveModellingConfig
):
    mean_head_config: HeadConfig = HeadMLPConfig()
    precision_head_config: HeadConfig = HeadMLPConfig()


@dataclass
class ConditionalGenerativeContrastiveModellingConvConfig(
    ConditionalGenerativeContrastiveModellingConfig
):
    mean_head_config: HeadConfig = HeadConvConfig()
    precision_head_config: HeadConfig = HeadConvConfig()
