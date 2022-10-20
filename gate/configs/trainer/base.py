from dataclasses import dataclass

from pytorch_lightning import Trainer

from gate.configs import get_module_import_path
from gate.configs.string_variables import (
    CURRENT_EXPERIMENT_DIR,
    NUM_TRAIN_SAMPLES,
)


@dataclass
class BaseTrainer:
    _target_: str = get_module_import_path(Trainer)
    gpus: int = 0
    accelerator: str = "cpu"
    enable_checkpointing: bool = True
    default_root_dir: str = CURRENT_EXPERIMENT_DIR
    progress_bar_refresh_rate: int = 1
    enable_progress_bar: bool = True
    val_check_interval: float = 0.02
    max_steps: int = NUM_TRAIN_SAMPLES
    log_every_n_steps: int = 1
    precision: int = 32
    num_sanity_val_steps: int = 2
    auto_scale_batch_size: bool = False
