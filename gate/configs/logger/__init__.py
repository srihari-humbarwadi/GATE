from hydra.core.config_store import ConfigStore

from .wandb import WeightsAndBiasesLoggerConfig


def add_logger_configs(config_store: ConfigStore):
    config_store.store(
        group="logger",
        name="wandb",
        node=dict(wandb_logger=WeightsAndBiasesLoggerConfig()),
    )

    return config_store
