import pytest

from gate.base.utils.loggers import get_logger
from gate.tasks.standard_classification import ImageClassificationTaskModule

log = get_logger(__name__, set_default_handler=True)


@pytest.mark.parametrize(
    "task",
    [
        ImageClassificationTaskModule,
    ],
)
@pytest.mark.parametrize(
    "output_shape_dict",
    [{"image": (10,)}, {"image": (100,)}, {"image": (1000,)}],
)  # , 4, 8, 16, 32, 64, 128, 256, 512])
def test_image_classification_tasks(task, output_shape_dict):
    """
    Test the ImageClassificationTaskModule
    """
    # test the ImageClassificationTaskModule
    task_module = task(output_shape_dict=output_shape_dict)
    # test the forward method
    assert task_module is not None
