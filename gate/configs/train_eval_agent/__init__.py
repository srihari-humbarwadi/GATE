from hydra.core.config_store import ConfigStore

from .base import BaseTrainEvalAgent


def add_train_eval_agent_configs(
    config_store: ConfigStore,
):
    config_store.store(
        group="train_eval_agent", name="base", node=BaseTrainEvalAgent
    )

    return config_store
