import inspect
import pathlib

import pytest
from pytorch_lightning import seed_everything

from gate.base.utils.loggers import get_logger
from gate.datasets.tf_hub.few_shot.omniglot import (
    OmniglotFewShotClassificationDataset,
)

log = get_logger(__name__, set_default_handler=True)


@pytest.mark.parametrize(
    "dataset",
    [
        OmniglotFewShotClassificationDataset,
    ],
)
@pytest.mark.parametrize("split_name", ["train", "test"])
@pytest.mark.parametrize("variable_num_samples_per_class", [True, False])
@pytest.mark.parametrize("variable_num_classes_per_set", [True, False])
@pytest.mark.parametrize("download", [True, False])
def test_omniglot_datasets(
    dataset,
    split_name,
    variable_num_samples_per_class,
    variable_num_classes_per_set,
    download,
):
    seed_everything(42, workers=True)
    log.info("Testing dataset: %s", dataset.__name__)
    input_transforms = None

    argument_names = inspect.signature(dataset.__init__).parameters.keys()
    log.info(f"Items: {argument_names} {'input_transform' in argument_names}")
    target_transforms = None

    dataset_instance = dataset(
        dataset_root=pathlib.Path("tests/data/omniglot/"),
        split_name=split_name,
        download=True,
        num_classes_per_set=100,
        num_samples_per_class=15,
        variable_num_samples_per_class=variable_num_samples_per_class,
        variable_num_classes_per_set=variable_num_classes_per_set,
        num_classes_per_set=6,
        num_samples_per_class=6,
        support_set_input_transform=input_transforms,
        support_set_target_transform=target_transforms,
        query_set_input_transform=input_transforms,
        query_set_target_transform=target_transforms,
        num_episodes=1000,
    )

    for i in range(len(dataset_instance)):
        item = dataset_instance[i]
        x, y = item
        for key, value in x.image.items():
            value = value.float()
            log.info(f"{key} {value.shape}")

        for key, value in y.image.items():
            value = value.float()
            log.info(f"{key} {value.shape}")

        break
