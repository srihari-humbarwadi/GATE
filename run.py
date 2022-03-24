import os

import dotenv
import hydra
from omegaconf import DictConfig
from rich.traceback import install

# load environment variables from `.env` file if it exists
# recursively searches for `.env` in all folders starting from work dir
dotenv.load_dotenv(override=True)
install(show_locals=False, extra_lines=1, word_wrap=True, width=350)


@hydra.main(config_path="configs", config_name="config")
def main(config: DictConfig):

    # Imports can be nested inside @hydra.main to optimize tab completion
    # https://github.com/facebookresearch/hydra/issues/934
    from gate.base import utils
    from gate.train_eval import train_eval

    utils.extras(config)
    os.environ["WANDB_PROGRAM"] = config.code_dir

    # Pretty print config using Rich library
    if config.get("print_config"):
        utils.print_config(config, resolve=True)

    return train_eval(config)


if __name__ == "__main__":
    main()
