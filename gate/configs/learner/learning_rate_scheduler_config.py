from dataclasses import dataclass

from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
    ReduceLROnPlateau,
)

from gate.configs import get_module_import_path
from gate.configs.string_variables import BATCH_SIZE, MaxDurationTypes


@dataclass
class LRSchedulerConfig:
    _target_: str = get_module_import_path(CosineAnnealingLR)


@dataclass
class UpdateIntervalOptions:
    EPOCH: str = "epoch"
    STEP: str = "step"


@dataclass
class CosineAnnealingLRConfig(LRSchedulerConfig):
    _target_: str = get_module_import_path(CosineAnnealingLR)
    T_max: int = MaxDurationTypes.MAX_STEPS
    eta_min: int = 0
    verbose: bool = False
    batch_size: int = BATCH_SIZE
    num_train_samples: int = MaxDurationTypes.MAX_STEPS
    update_interval: str = UpdateIntervalOptions.STEP


@dataclass
class CosineAnnealingLRWarmRestartsConfig(LRSchedulerConfig):
    _target_: str = get_module_import_path(CosineAnnealingWarmRestarts)
    T_0: int = 10000
    verbose: bool = False
    eta_min: int = BATCH_SIZE


@dataclass
class ReduceLROnPlateauConfig(LRSchedulerConfig):
    _target_: str = get_module_import_path(ReduceLROnPlateau)
    verbose: bool = False
    mode: str = "min"
    factor: float = 0.5
    patience: int = 100
    threshold: float = 0.0001
    threshold_mode: str = "rel"
    cooldown: int = 0
    min_lr: float = 0.0
    eps: float = 1e-08


@dataclass
class BiLevelLRSchedulerConfig:
    inner_loop_lr_scheduler_config: LRSchedulerConfig = (
        CosineAnnealingLRConfig()
    )
    outer_loop_lr_scheduler_config: LRSchedulerConfig = (
        CosineAnnealingLRConfig()
    )
