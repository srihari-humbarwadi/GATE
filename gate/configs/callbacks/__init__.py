import inspect

from hydra.core.config_store import ConfigStore

from .base import (
    LearningRateMonitor,
    LogConfigInformation,
    LogGrads,
    ModelSummaryConfig,
    RichProgressBar,
    UploadCheckpointsAsArtifact,
    UploadCodeAsArtifact,
    model_checkpoint_eval,
    model_checkpoint_train,
)

base_callbacks = dict(
    model_checkpoint_eval=model_checkpoint_eval,
    model_checkpoint_train=model_checkpoint_train,
    model_summary=ModelSummaryConfig(),
    progress_bar=RichProgressBar(),
    lr_monitor=LearningRateMonitor(),
)


wandb_callbacks = dict(
    model_checkpoint_eval=model_checkpoint_eval,
    model_checkpoint_train=model_checkpoint_train,
    model_summary=ModelSummaryConfig(),
    progress_bar=RichProgressBar(),
    lr_monitor=LearningRateMonitor,
    code_upload=UploadCodeAsArtifact(),
    log_grads=LogGrads(),
    log_config=LogConfigInformation(),
)


def add_lightning_callback_configs(config_store: ConfigStore):
    config_store.store(
        group="callbacks",
        name="base",
        node=base_callbacks,
    )

    config_store.store(
        group="callbacks",
        name="wandb",
        node=wandb_callbacks,
    )
    return config_store
