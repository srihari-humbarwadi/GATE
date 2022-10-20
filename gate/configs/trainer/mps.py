from dataclasses import dataclass

from gate.configs.trainer.base import BaseTrainer


@dataclass
class MPSTrainer(BaseTrainer):
    accelerator: str = "mps"
    gpus: int = 0
