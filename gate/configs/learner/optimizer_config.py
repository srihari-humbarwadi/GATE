from dataclasses import MISSING, dataclass, field
from typing import List

from torch.optim import Adam

from gate.configs import get_module_import_path


@dataclass
class BaseOptimizerConfig:
    lr: float = MISSING
    _target_: str = MISSING


@dataclass
class AdamOptimizerConfig(BaseOptimizerConfig):
    _target_: str = get_module_import_path(Adam)
    lr: float = 1e-3
    weight_decay: float = 0.00001
    amsgrad: bool = False
    betas: List = field(default_factory=lambda: [0.9, 0.999])


@dataclass
class BiLevelOptimizerConfig:
    outer_loop_optimizer_config: BaseOptimizerConfig = AdamOptimizerConfig()
    inner_loop_optimizer_config: BaseOptimizerConfig = AdamOptimizerConfig(
        lr=2e-5
    )
