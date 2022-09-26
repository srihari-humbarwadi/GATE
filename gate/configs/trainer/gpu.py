from dataclasses import dataclass
from typing import Any

from pytorch_lightning.plugins import DDPPlugin

from gate.configs import get_module_import_path
from gate.configs.trainer.base import BaseTrainer


@dataclass
class DDPPlugin:
    _target_: str = get_module_import_path(DDPPlugin)
    find_unused_parameters: bool = False


@dataclass
class DDPTrainer(BaseTrainer):
    accelerator: str = "gpu"
    strategy: Any = None
    replace_sampler_ddp: bool = True
    sync_batchnorm: bool = True
    auto_scale_batch_size: bool = False
    plugins: Any = DDPPlugin()


@dataclass
class DPTrainer(BaseTrainer):
    accelerator: str = "gpu"
    strategy: str = "dp"
    auto_scale_batch_size: bool = False
