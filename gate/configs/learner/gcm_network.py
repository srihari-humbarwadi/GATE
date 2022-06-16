from dataclasses import dataclass, MISSING

from gate.configs import get_module_import_path
from gate.configs.learner.learning_rate_scheduler_config import (
    CosineAnnealingLRConfig,
    LRSchedulerConfig,
)
from gate.configs.learner.linear_layer_fine_tuning import LearnerConfig
from gate.configs.learner.optimizer_config import (
    BaseOptimizerConfig,
    AdamOptimizerConfig,
)
from gate.learners.GCM import ConditionalGenerativeContrastiveModelling


@dataclass
class ConditionalGenerativeContrastiveModellingConfig(LearnerConfig):
    _target_: str = get_module_import_path(
        ConditionalGenerativeContrastiveModelling
    )
    fine_tune_all_layers: bool = True
    use_input_instance_norm: bool = True
    head_num_layers: int = 3
    head_num_hidden_filters: int = 64
    head_num_output_filters: int = 64
    optimizer_config: BaseOptimizerConfig = AdamOptimizerConfig(lr=1e-3)
    lr_scheduler_config: LRSchedulerConfig = CosineAnnealingLRConfig()
