from dataclasses import dataclass
from typing import Any

from pytorch_lightning.plugins import DDPPlugin

from gate.configs import get_module_import_path
from gate.configs.trainer.base import BaseTrainer


@dataclass
class MPSTrainer(BaseTrainer):
    accelerator: str = "mps"
    gpus: int = 0
