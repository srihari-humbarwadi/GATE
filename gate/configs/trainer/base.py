from dataclasses import dataclass

from pytorch_lightning import Trainer

from gate.configs import get_module_import_path


@dataclass
class BaseTrainer:
    _target_: str = get_module_import_path(Trainer)
    gpus: int = -1
    enable_checkpointing: bool = True
    strategy: str = "dp"
    default_root_dir: str = "${current_experiment_dir}"
    progress_bar_refresh_rate: int = 1
    enable_progress_bar: bool = True
    val_check_interval: float = 1.0
    max_epochs: int = 5
    log_every_n_steps: int = 1
    precision: int = 32
    num_sanity_val_steps: int = 2
    auto_scale_batch_size: bool = False
