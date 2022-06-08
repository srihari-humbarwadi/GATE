from dataclasses import dataclass
from typing import Any

from gate.configs import get_module_import_path
from gate.configs.learner.linear_layer_fine_tuning import LearnerConfig
from gate.configs.task.image_classification import TaskConfig
from gate.train_eval_agents.base import TrainingEvaluationAgent


@dataclass
class ModalityConfig:
    image: bool = True
    audio: bool = False
    text: bool = False
    video: bool = False
    image_text: bool = False
    audio_text: bool = False
    video_text: bool = False
    image_audio: bool = False
    image_video: bool = False
    audio_video: bool = False
    image_audio_text: bool = False
    image_video_text: bool = False
    audio_video_text: bool = False
    image_audio_video: bool = False
    image_audio_video_text: bool = False


@dataclass
class BaseTrainEvalAgent:
    _target_: str = get_module_import_path(TrainingEvaluationAgent)
    model_config: Any = "${model}"
    learner_config: LearnerConfig = "${learner}"
    task_config: TaskConfig = "${task}"
    modality_config: ModalityConfig = ModalityConfig()
