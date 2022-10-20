from dataclasses import MISSING, dataclass, field
from typing import List

from gate.configs import get_module_import_path
from torch.optim import Adam, AdamW


@dataclass
class BaseOptimizerConfig:
    lr: float = MISSING
    _target_: str = MISSING


@dataclass
class AdamWOptimizerConfig(BaseOptimizerConfig):
    _target_: str = get_module_import_path(AdamW)
    lr: float = 2e-5
    weight_decay: float = 0.00001
    amsgrad: bool = False
    betas: List = field(default_factory=lambda: [0.9, 0.999])


@dataclass
class AdamOptimizerConfig(BaseOptimizerConfig):
    _target_: str = get_module_import_path(Adam)
    lr: float = 2e-5
    weight_decay: float = 0.00001
    amsgrad: bool = False
    betas: List = field(default_factory=lambda: [0.9, 0.999])


@dataclass
class BiLevelOptimizerConfig:
    outer_loop_optimizer_config: BaseOptimizerConfig = AdamWOptimizerConfig()
    inner_loop_optimizer_config: BaseOptimizerConfig = AdamWOptimizerConfig(lr=2e-5)
