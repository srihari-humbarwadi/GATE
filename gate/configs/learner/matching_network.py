from dataclasses import dataclass

from gate.configs import get_module_import_path
from gate.configs.learner.base import LearnerConfig
from gate.configs.learner.learning_rate_scheduler_config import (
    CosineAnnealingLRConfig,
    LRSchedulerConfig,
)
from gate.configs.learner.optimizer_config import (
    AdamOptimizerConfig,
    BaseOptimizerConfig,
)
from gate.learners.matchingnet import MatchingNetworkEpisodicTuningScheme


@dataclass
class EpisodicMatchingNetworkConfig(LearnerConfig):
    _target_: str = get_module_import_path(MatchingNetworkEpisodicTuningScheme)
    fine_tune_all_layers: bool = True
    use_input_instance_norm: bool = True
    optimizer_config: BaseOptimizerConfig = AdamOptimizerConfig(lr=1e-3)
    lr_scheduler_config: LRSchedulerConfig = CosineAnnealingLRConfig()
