from hydra.core.config_store import ConfigStore

from .base import BaseTrainer
from .ddp import DDPTrainer


def add_trainer_configs(config_store: ConfigStore):
    config_store.store(group="trainer", name="base", node=BaseTrainer)
    config_store.store(group="trainer", name="ddp", node=DDPTrainer)

    return config_store
