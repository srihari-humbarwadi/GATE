import os
import pathlib
from typing import List, Optional

import hydra
import torch
from omegaconf import DictConfig
from pytorch_lightning import (
    Callback,
    seed_everything,
    Trainer,
)
from pytorch_lightning.loggers import LightningLoggerBase
from pytorch_lightning.tuner.tuning import Tuner
from rich.traceback import install
from wandb.util import generate_id

from gate.base.utils.loggers import get_logger
from gate.datamodules.base import DataModule
from gate.train_eval_agents.base import TrainingEvaluationAgent

log = get_logger(__name__)

install(show_locals=False, extra_lines=1, word_wrap=True, width=350)


def checkpoint_setup(config):
    checkpoint_path = None

    if config.resume:

        log.info("Continue from existing checkpoint")

        if not pathlib.Path(f"{config.current_experiment_dir}").exists():
            os.makedirs(f"{config.current_experiment_dir}", exist_ok=True)

        checkpoint_path = f"{config.current_experiment_dir}/checkpoints/last.ckpt"

        log.info(checkpoint_path)

        if not pathlib.Path(checkpoint_path).exists():
            checkpoint_path = None

    else:

        log.info("Starting from scratch")
        if not pathlib.Path(f"{config.current_experiment_dir}").exists():
            os.makedirs(f"{config.current_experiment_dir}", exist_ok=True)

    return checkpoint_path


def train_eval(config: DictConfig):
    """Contains training pipeline.
    Instantiates all PyTorch Lightning objects from config.

    Args:
        config (DictConfig): Configuration composed by Hydra.

    Returns:
        Optional[float]: Metric score for hyperparameter optimization.
    """

    if config.get("seed"):
        seed_everything(config.seed, workers=True)
    # --------------------------------------------------------------------------------
    # Create or recover checkpoint path to resume from
    checkpoint_path = checkpoint_setup(config)
    # --------------------------------------------------------------------------------
    # Instantiate Lightning DataModule for task
    log.info(f"Instantiating datamodule <{config.datamodule._target_}>")
    datamodule: DataModule = hydra.utils.instantiate(
        config.datamodule, _recursive_=False
    )
    datamodule.setup(stage="fit")
    # --------------------------------------------------------------------------------
    # Instantiate Lightning TrainingEvaluationAgent for task
    log.info(f"Instantiating model <{config.model._target_}>")

    train_eval_agent: TrainingEvaluationAgent = hydra.utils.instantiate(
        config.train_eval_agent, datamodule=datamodule, _recursive_=False
    )
    # --------------------------------------------------------------------------------
    # Instantiate Lightning Learner using a dummy data dict with the
    # data names and shapes
    x_dummy_data_dict, y_dummy_data_dict = datamodule.dummy_batch()

    x_dummy_data_dict = {
        key: value
        for key, value in x_dummy_data_dict.items()
        if isinstance(value, torch.Tensor)
    }

    dummy_data_device_dict = {
        key: value.device
        for key, value in x_dummy_data_dict.items()
        if isinstance(value, torch.Tensor)
    }

    log.info(f"Data shape description: {dummy_data_device_dict}")
    _ = train_eval_agent.forward(x_dummy_data_dict)
    # --------------------------------------------------------------------------------
    # Instantiate Lightning Callbacks
    # --------------------------------------------------------------------------------
    callbacks: List[Callback] = []
    if "callbacks" in config:
        for _, cb_conf in config.callbacks.items():
            if "_target_" in cb_conf:
                if (
                    cb_conf["_target_"]
                    == "tali.base.callbacks.wandb_callbacks.LogConfigInformation"
                ):
                    cb_conf["config"] = dict(config)
                    log.info(f"Instantiating callback <{cb_conf._target_}>")
                    callbacks.append(
                        hydra.utils.instantiate(cb_conf, _recursive_=False)
                    )
                else:
                    log.info(f"Instantiating callback <{cb_conf._target_}>")
                    callbacks.append(hydra.utils.instantiate(cb_conf))

    os.environ["WANDB_RESUME"] = "allow"
    os.environ["WANDB_RUN_ID"] = generate_id()
    # --------------------------------------------------------------------------------
    # Instantiate Experiment Logger
    # --------------------------------------------------------------------------------
    # Init lightning loggers
    logger: List[LightningLoggerBase] = []
    if "logger" in config:
        for _, lg_conf in config.logger.items():
            if "_target_" in lg_conf:
                log.info(f"Instantiating logger <{lg_conf._target_}>")
                logger.append(hydra.utils.instantiate(lg_conf))

    # --------------------------------------------------------------------------------
    # Instantiate Lightning Trainer
    # --------------------------------------------------------------------------------
    log.info(f"Instantiating trainer <{config.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(
        config.trainer,
        callbacks=callbacks,
        logger=logger,
        _convert_="partial",
    )

    # --------------------------------------------------------------------------------
    # If auto_scale_batch_size is set, we need to tune the batch size using
    # the Lightning Tuner class, starting from given batch size and increasing
    # in powers of 2
    if config.trainer.auto_scale_batch_size:
        tuner = Tuner(trainer)
        new_batch_size = tuner.scale_batch_size(
            train_eval_agent,
            datamodule=datamodule,
            mode="power",
            init_val=2 * torch.cuda.device_count(),
        )
        datamodule.batch_size = new_batch_size
        config.datamodule.batch_size = new_batch_size

    # --------------------------------------------------------------------------------
    # Start training
    if config.mode.fit:
        log.info("Starting training!")
        trainer.validate(
            model=train_eval_agent, datamodule=datamodule, ckpt_path=checkpoint_path
        )

        trainer.fit(
            model=train_eval_agent, datamodule=datamodule, ckpt_path=checkpoint_path
        )

    # --------------------------------------------------------------------------------
    # Start evaluation on test set
    if config.mode.test and not config.trainer.get("fast_dev_run"):
        datamodule.setup(stage="test")

        if config.mode.fit is False:
            trainer._restore_modules_and_callbacks(checkpoint_path=checkpoint_path)

        log.info(f"Starting testing ! 🧪")

        trainer.test(
            model=train_eval_agent,
            datamodule=datamodule,
        )

    # Make sure everything closed properly
    log.info("Finalizing! 😺")
    # Print path to best checkpoint
    if not config.trainer.get("fast_dev_run"):
        log.info(f"Best model ckpt at {trainer.checkpoint_callback.best_model_path}")