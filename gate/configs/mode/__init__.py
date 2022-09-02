from hydra.core.config_store import ConfigStore

from .base import BaseMode


def add_mode_configs(config_store: ConfigStore):
    config_store.store(
        group="mode",
        name="base",
        node=BaseMode(),
    )

    return config_store
