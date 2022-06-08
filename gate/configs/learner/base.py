from dataclasses import MISSING, dataclass


@dataclass
class LearnerConfig:
    _target_: str = MISSING
