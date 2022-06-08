from dataclasses import MISSING, dataclass
from datetime import timedelta
from typing import Dict, Optional

from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    RichModelSummary,
    TQDMProgressBar,
    LearningRateMonitor,
)

from gate.base.callbacks.wandb_callbacks import (
    UploadCodeAsArtifact,
    UploadCheckpointsAsArtifact,
    LogConfigInformation,
    LogGrads,
)
from gate.configs import get_module_import_path
from gate.configs.string_variables import CHECKPOINT_DIR


@dataclass
class TimerConfig:
    _target_: str = get_module_import_path(timedelta)
    minutes: int = 15


@dataclass
class ModelCheckpointingConfig:
    monitor: str = MISSING
    mode: str = MISSING
    save_top_k: int = MISSING
    save_last: bool = MISSING
    verbose: bool = MISSING
    filename: str = MISSING
    auto_insert_metric_name: bool = MISSING
    save_on_train_epoch_end: Optional[bool] = None
    train_time_interval: Optional[TimerConfig] = None
    _target_: str = get_module_import_path(ModelCheckpoint)
    dirpath: str = CHECKPOINT_DIR


@dataclass
class ModelSummaryConfig:
    _target_: str = get_module_import_path(RichModelSummary)
    max_depth: int = 7


@dataclass
class RichProgressBar:
    _target_: str = get_module_import_path(TQDMProgressBar)
    refresh_rate: int = 1
    process_position: int = 0


@dataclass
class LearningRateMonitor:
    _target_: str = get_module_import_path(LearningRateMonitor)
    logging_interval: str = "step"


@dataclass
class UploadCodeAsArtifact:
    _target_: str = get_module_import_path(UploadCodeAsArtifact)
    code_dir: str = "${code_dir}"


@dataclass
class UploadCheckpointsAsArtifact:
    _target_: str = get_module_import_path(UploadCheckpointsAsArtifact)
    ckpt_dir: str = "${current_experiment_dir}/checkpoints/"
    upload_best_only: bool = False


@dataclass
class LogGrads:
    _target_: str = get_module_import_path(LogGrads)
    refresh_rate: int = 100


@dataclass
class LogConfigInformation:
    _target_: str = get_module_import_path(LogConfigInformation)
    config: Optional[Dict] = None


model_checkpoint_eval: ModelCheckpointingConfig = ModelCheckpointingConfig(
    monitor="validation/opt_loss_epoch",
    mode="min",
    save_top_k=3,
    save_last=False,
    verbose=False,
    dirpath=CHECKPOINT_DIR,
    filename="eval_epoch",
    auto_insert_metric_name=False,
)

model_checkpoint_train = ModelCheckpointingConfig(
    _target_=get_module_import_path(ModelCheckpoint),
    monitor="training/opt_loss_step",
    save_on_train_epoch_end=True,
    save_top_k=0,
    save_last=True,
    train_time_interval=TimerConfig(),
    mode="min",
    verbose=False,
    dirpath=CHECKPOINT_DIR,
    filename="last",
    auto_insert_metric_name=False,
)
