import os

import dotenv
import hydra
from omegaconf import DictConfig, OmegaConf
from rich.traceback import install

from gate.base.utils.rank_zero_ops import extras, print_config

# load environment variables from `.env-` file if it exists
# recursively searches for `.env` in all folders starting from work dir

dotenv.load_dotenv(override=True, verbose=True)
install(show_locals=False, extra_lines=1, word_wrap=True, width=350)

from gate.configs.config import collect_config_store

config_store = collect_config_store()


@hydra.main(version_base=None, config_name="config")
def main(config: DictConfig):
    # Imports can be nested inside @hydra.main to optimize tab completion
    # https://github.com/facebookresearch/hydra/issues/934
    from gate.train_eval import train_eval

    os.environ["WANDB_PROGRAM"] = config.code_dir

    extras(config)

    return train_eval(config)


if __name__ == "__main__":
    main()
