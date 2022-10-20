from hydra.core.config_store import ConfigStore

from .base import BaseTrainer
from .gpu import DDPTrainer, DPTrainer
from .mps import MPSTrainer


def add_trainer_configs(config_store: ConfigStore):
    config_store.store(group="trainer", name="base", node=BaseTrainer)
    config_store.store(group="trainer", name="gpu-dp", node=DPTrainer)
    config_store.store(group="trainer", name="gpu-ddp", node=DDPTrainer)
    config_store.store(group="trainer", name="mps", node=MPSTrainer)

    return config_store
