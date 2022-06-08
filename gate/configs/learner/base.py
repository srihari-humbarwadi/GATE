from dataclasses import dataclass, MISSING


@dataclass
class LearnerConfig:
    _target_: str = MISSING
