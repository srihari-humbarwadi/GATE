# https://wandb.ai
import os
from dataclasses import dataclass, field

from typing import List

from pytorch_lightning.loggers import WandbLogger

from gate.configs import get_module_import_path


@dataclass
class WeightsAndBiasesLoggerConfig:
    _target_: str = get_module_import_path(WandbLogger)
    project: str = os.environ["WANDB_PROJECT"]
    offline: bool = False  # set True to store all logs only locally
    resume: str = "allow"  # allow, True, False, must
    save_dir: str = "${current_experiment_dir}/"
    log_model: str = "False"
    prefix: str = ""
    job_type: str = "train"
    group: str = ""
    tags: List[str] = field(default_factory=list)
