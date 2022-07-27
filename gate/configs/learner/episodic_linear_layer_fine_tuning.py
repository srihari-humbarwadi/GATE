from dataclasses import MISSING, dataclass

from gate.configs import get_module_import_path
from gate.configs.learner.base import LearnerConfig
from gate.configs.learner.learning_rate_scheduler_config import (
    BiLevelLRSchedulerConfig,
)
from gate.configs.learner.optimizer_config import BiLevelOptimizerConfig
from gate.learners.single_layer_fine_tuning_episodic import (
    EpisodicLinearLayerFineTuningScheme,
)


@dataclass
class EpisodicSingleLinearLayerFineTuningSchemeConfig(LearnerConfig):
    _target_: str = get_module_import_path(EpisodicLinearLayerFineTuningScheme)
    fine_tune_all_layers: bool = False
    use_input_instance_norm: bool = True
    optimizer_config: BiLevelOptimizerConfig = BiLevelOptimizerConfig()
    lr_scheduler_config: BiLevelLRSchedulerConfig = BiLevelLRSchedulerConfig()
    inner_loop_steps: int = 100


@dataclass
class EpisodicFullModelFineTuningSchemeConfig(LearnerConfig):
    _target_: str = get_module_import_path(EpisodicLinearLayerFineTuningScheme)
    fine_tune_all_layers: bool = True
    use_input_instance_norm: bool = True
    optimizer_config: BiLevelOptimizerConfig = BiLevelOptimizerConfig()
    lr_scheduler_config: BiLevelLRSchedulerConfig = BiLevelLRSchedulerConfig()
    inner_loop_steps: int = 100
