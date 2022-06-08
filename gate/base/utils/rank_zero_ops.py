import logging
import warnings
from typing import List

import rich
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import (
    Callback,
    LightningDataModule,
    LightningModule,
    Trainer,
    loggers,
)
from pytorch_lightning.utilities import rank_zero_only
from rich.tree import Tree


def get_logger_rank_zero_only(name=__name__) -> logging.Logger:
    """Initializes multi-GPU-friendly python command line logger."""

    logger = logging.getLogger(name)

    # this ensures all logging levels get marked with the rank zero decorator
    # otherwise logs would get multiplied for each GPU process in multi-GPU setup
    for level in (
        "debug",
        "info",
        "warning",
        "error",
        "exception",
        "fatal",
        "critical",
    ):
        from pytorch_lightning.utilities import rank_zero_only

        setattr(logger, level, rank_zero_only(getattr(logger, level)))

    return logger


def extras(config: DictConfig) -> None:
    """Applies optional utilities, controlled by config flags.

    Utilities:
    - Ignoring python warnings
    - Rich config printing
    """
    log = get_logger_rank_zero_only(__name__)

    # disable python warnings if <config.ignore_warnings=True>
    # disable python warnings if <config.ignore_warnings=True>
    if config.get("ignore_warnings"):
        log.info("Disabling python warnings! <config.ignore_warnings=True>")
        warnings.filterwarnings("ignore")

    # pretty print config tree using Rich library if <config.print_config=True>
    if config.get("print_config"):
        log.info("Printing config tree with Rich! <config.print_config=True>")
        print_config(config, resolve=True)


def generate_config_tree(config: DictConfig, resolve: bool = True):
    style = "dim"

    tree = Tree("CONFIG", style=style, guide_style=style)

    for field in config.keys():
        branch = tree.add(field, style=style, guide_style=style)

        config_section = config.get(field)
        branch_content = str(config_section)
        if isinstance(config_section, DictConfig):

            branch_content = OmegaConf.to_yaml(config_section, resolve=resolve)

        branch.add(rich.syntax.Syntax(branch_content, "yaml"))

    return tree


@rank_zero_only
def print_config(
    config: DictConfig,
    resolve: bool = True,
) -> None:
    """Prints content of DictConfig using Rich library and its tree structure.

    Args:
        config (DictConfig): Configuration composed by Hydra.
        fields (Sequence[str], optional): Determines which main fields from
        config will
        be printed and in what order.
        resolve (bool, optional): Whether to resolve reference fields
        of DictConfig.
    """
    tree = generate_config_tree(config, resolve=resolve)
    rich.print(tree)

    with open("config_tree.log", "w") as fp:
        rich.print(tree, file=fp)


@rank_zero_only
def log_hyperparameters(
    config: DictConfig,
    model: LightningModule,
    datamodule: LightningDataModule,
    trainer: Trainer,
    callbacks: List[Callback],
    logger: List[loggers.LightningLoggerBase],
) -> None:
    """This method controls which parameters from Hydra config are saved by Lightning loggers.

    Additionaly saves:
        - number of model parameters
    """

    hparams = {
        "trainer": config["trainer"],
        "model": config["model"],
        "datamodule": config["datamodule"],
    }

    # choose which parts of hydra config will be saved to loggers

    if "seed" in config:
        hparams["seed"] = config["seed"]
    if "callbacks" in config:
        hparams["callbacks"] = config["callbacks"]

    # save number of model parameters
    hparams["model/params/total"] = sum(p.numel() for p in model.parameters())
    hparams["model/params/trainable"] = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    hparams["model/params/non_trainable"] = sum(
        p.numel() for p in model.parameters() if not p.requires_grad
    )

    # send hparams to all loggers
    trainer.logger.log_hyperparams(hparams)
