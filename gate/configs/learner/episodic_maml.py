from dataclasses import MISSING, dataclass

from gate.configs import get_module_import_path
from gate.configs.learner.base import LearnerConfig
from gate.configs.learner.learning_rate_scheduler_config import (
    BiLevelLRSchedulerConfig,
)
from gate.configs.learner.optimizer_config import (
    BiLevelOptimizerConfig,
    AdamOptimizerConfig,
)
from gate.learners.maml_episodic import EpisodicMAML


@dataclass
class EpisodicMAMLSingleLinearLayerConfig(LearnerConfig):
    _target_: str = get_module_import_path(EpisodicMAML)
    fine_tune_all_layers: bool = False
    use_input_instance_norm: bool = True
    optimizer_config: BiLevelOptimizerConfig = BiLevelOptimizerConfig(
        inner_loop_optimizer_config=AdamOptimizerConfig(lr=2e-5)
    )
    lr_scheduler_config: BiLevelLRSchedulerConfig = BiLevelLRSchedulerConfig()
    inner_loop_steps: int = 1
    use_cosine_similarity: bool = True
    use_weight_norm: bool = False
    temperature: float = 10.0
    manual_optimization: bool = True
    include_coordinate_information: bool = False


@dataclass
class EpisodicMAMLFullModelConfig(LearnerConfig):
    _target_: str = get_module_import_path(EpisodicMAML)
    fine_tune_all_layers: bool = True
    use_input_instance_norm: bool = True
    optimizer_config: BiLevelOptimizerConfig = BiLevelOptimizerConfig(
        inner_loop_optimizer_config=AdamOptimizerConfig(lr=2e-5)
    )
    lr_scheduler_config: BiLevelLRSchedulerConfig = BiLevelLRSchedulerConfig()
    inner_loop_steps: int = 1
    use_cosine_similarity: bool = True
    use_weight_norm: bool = False
    temperature: float = 10.0
    manual_optimization: bool = True
    include_coordinate_information: bool = False
