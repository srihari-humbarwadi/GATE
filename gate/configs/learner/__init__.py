from hydra.core.config_store import ConfigStore

from .episodic_linear_layer_fine_tuning import (
    EpisodicFullModelFineTuningSchemeConfig,
    EpisodicSingleLinearLayerFineTuningSchemeConfig,
)
from .gcm_network import (
    ConditionalGenerativeContrastiveModellingConfig,
    ConditionalGenerativeContrastiveModellingConvHeadConfig,
    ConditionalGenerativeContrastiveModellingMLPHeadConfig,
    ConditionalGenerativeContrastiveModellingResNetHeadConfig,
)
from .learning_rate_scheduler_config import (
    CosineAnnealingLRConfig,
    CosineAnnealingLRWarmRestartsConfig,
    ReduceLROnPlateauConfig,
)
from .linear_layer_fine_tuning import (
    FullModelFineTuningSchemeConfig,
    SingleLinearLayerFineTuningSchemeConfig,
)
from .optimizer_config import AdamOptimizerConfig, BiLevelOptimizerConfig
from .prototypical_network import EpisodicPrototypicalNetworkConfig

LEARNING_RATE_SCHEDULER_CONFIGS = "learner/learning_rate_scheduler"
LEARNER_CONFIGS = "learner"
OPTIMIZER_CONFIGS = "learner/optimizer"


def add_learning_scheduler_configs(
    config_store: ConfigStore,
):
    config_store.store(
        group=LEARNING_RATE_SCHEDULER_CONFIGS,
        name="CosineAnnealingLR",
        node=CosineAnnealingLRConfig,
    )

    config_store.store(
        group=LEARNING_RATE_SCHEDULER_CONFIGS,
        name="CosineAnnealingLRWarmRestarts",
        node=CosineAnnealingLRWarmRestartsConfig,
    )

    config_store.store(
        group=LEARNING_RATE_SCHEDULER_CONFIGS,
        name="ReduceLROnPlateau",
        node=ReduceLROnPlateauConfig,
    )

    return config_store


def add_optimizer_configs(config_store: ConfigStore):
    config_store.store(
        group=OPTIMIZER_CONFIGS,
        name="CosineAnnealingLR",
        node=CosineAnnealingLRConfig,
    )

    config_store.store(
        group=OPTIMIZER_CONFIGS,
        name="CosineAnnealingLRWarmRestarts",
        node=CosineAnnealingLRWarmRestartsConfig,
    )

    config_store.store(
        group=OPTIMIZER_CONFIGS,
        name="ReduceLROnPlateau",
        node=ReduceLROnPlateauConfig,
    )

    return config_store


def add_learner_configs(config_store: ConfigStore):
    config_store.store(
        group=LEARNER_CONFIGS,
        name="SingleLinearLayerFineTuning",
        node=SingleLinearLayerFineTuningSchemeConfig,
    )

    config_store.store(
        group=LEARNER_CONFIGS,
        name="FullModelFineTuning",
        node=FullModelFineTuningSchemeConfig,
    )

    config_store.store(
        group=LEARNER_CONFIGS,
        name="EpisodicPrototypicalNetwork",
        node=EpisodicPrototypicalNetworkConfig,
    )

    config_store.store(
        group=LEARNER_CONFIGS,
        name="EpisodicSingleLinearLayerFineTuning",
        node=EpisodicSingleLinearLayerFineTuningSchemeConfig,
    )

    config_store.store(
        group=LEARNER_CONFIGS,
        name="EpisodicFullModelFineTuning",
        node=EpisodicFullModelFineTuningSchemeConfig,
    )

    config_store.store(
        group=LEARNER_CONFIGS,
        name="ConditionalGenerativeContrastiveModelling",
        node=ConditionalGenerativeContrastiveModellingConfig,
    )

    config_store.store(
        group=LEARNER_CONFIGS,
        name="ConditionalGenerativeContrastiveMLPHeadModelling",
        node=ConditionalGenerativeContrastiveModellingMLPHeadConfig,
    )

    config_store.store(
        group=LEARNER_CONFIGS,
        name="ConditionalGenerativeContrastiveConvHeadModelling",
        node=ConditionalGenerativeContrastiveModellingConvHeadConfig,
    )

    config_store.store(
        group=LEARNER_CONFIGS,
        name="ConditionalGenerativeContrastiveResNetHeadModelling",
        node=ConditionalGenerativeContrastiveModellingResNetHeadConfig,
    )

    return config_store
