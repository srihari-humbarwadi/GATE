import logging

import pytest
import torch
from rich.logging import RichHandler

from gate.class_configs.base import (
    CIFAR10DatasetConfig,
    DataLoaderConfig,
    CIFAR100DatasetConfig,
)
from gate.datamodules.cifar import CIFAR10DataModule, CIFAR100DataModule
from gate.tasks.classification import ImageClassificationTaskModule

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
ch = RichHandler()
ch.setLevel(logging.INFO)

# create formatter
formatter = logging.Formatter("%(levelname)s - %(message)s")

# add formatter to ch
ch.setFormatter(formatter)

# add ch to logger
log.addHandler(ch)


@pytest.mark.parametrize(
    "task",
    [
        ImageClassificationTaskModule,
    ],
)
@pytest.mark.parametrize(
    "output_shape_dict", [{"image": (10,)}, {"image": (100,)}, {"image": (1000,)}]
)  # , 4, 8, 16, 32, 64, 128, 256, 512])
def test_image_classification_tasks(task, output_shape_dict):
    """
    Test the ImageClassificationTaskModule
    """
    # test the ImageClassificationTaskModule
    task_module = task(output_shape_dict=output_shape_dict)
    # test the forward method
    assert task_module is not None
