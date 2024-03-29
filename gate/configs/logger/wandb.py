# https://wandb.ai
import os
from dataclasses import dataclass, field
from typing import List, Optional

from pytorch_lightning.loggers import WandbLogger

from gate.configs import get_module_import_path
from gate.configs.string_variables import CURRENT_EXPERIMENT_DIR


@dataclass
class WeightsAndBiasesLoggerConfig:
    _target_: str = get_module_import_path(WandbLogger)
    project: str = os.environ["WANDB_PROJECT"]
    offline: bool = False  # set True to store all logs only locally
    resume: str = "allow"  # allow, True, False, must
    save_dir: str = CURRENT_EXPERIMENT_DIR
    log_model: Optional[str] = None
    prefix: str = ""
    job_type: str = "train"
    group: str = ""
    tags: List[str] = field(default_factory=list)
