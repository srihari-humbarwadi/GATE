import os
import warnings

import dotenv
import hydra
import rich
from omegaconf import DictConfig, OmegaConf
from rich.traceback import install
from rich.tree import Tree

from gate.base.utils.tf_babysitting import configure_tf_memory_growth

install()

configure_tf_memory_growth()

from gate.base.utils.loggers import get_logger
from gate.base.utils.rank_zero_ops import extras

# load environment variables from `.env-` file if it exists
# recursively searches for `.env` in all folders starting from work dir

dotenv.load_dotenv(override=True, verbose=True)
log = get_logger(__name__)

from gate.configs.config import collect_config_store

config_store = collect_config_store()


@hydra.main(version_base=None, config_name="config")
def main(config: DictConfig):
    # Imports can be nested inside @hydra.main to optimize tab completion
    # https://github.com/facebookresearch/hydra/issues/934

    from gate.train_eval import train_eval

    extras(config)

    return train_eval(config)


if __name__ == "__main__":
    main()
