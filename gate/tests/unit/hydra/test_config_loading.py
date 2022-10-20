import dotenv
import hydra
import pytest
from omegaconf import OmegaConf

from gate.base.utils.loggers import get_logger
from gate.base.utils.rank_zero_ops import print_config

dotenv_loaded_vars = dotenv.load_dotenv(
    override=True, verbose=True, dotenv_path="gate/tests/unit/hydra/.env-test"
)

log = get_logger(__name__)

log.info(f"Loaded dotenv variables: {dotenv_loaded_vars}")

from gate.configs.config import Config, collect_config_store

log = get_logger(__name__, set_default_handler=True)


def test_config_loading():
    config_store = collect_config_store()

    print_config(config_store.repo, resolve=True)
