import inspect
import pathlib

import pytest
import torch
import torchvision.transforms as transforms

from gate.base.utils.loggers import get_logger
from gate.datasets.tf_hub.standard.omniglot import OmniglotClassificationDataset

log = get_logger(__name__, set_default_handler=True)


@pytest.mark.parametrize(
    "dataset",
    [
        OmniglotClassificationDataset,
    ],
)
@pytest.mark.parametrize("split_name", ["train", "test"])
@pytest.mark.parametrize("download", [True, False])
def test_omniglot_datasets(dataset, split_name, download):
    log.info("Testing dataset: %s", dataset.__name__)
    input_transforms = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )

    argument_names = inspect.signature(dataset.__init__).parameters.keys()
    log.info(f"Items: {argument_names} {'input_transform' in argument_names}")
    target_transforms = None

    dataset_instance = dataset(
        dataset_root=pathlib.Path("tests/data/omniglot/"),
        input_transform=input_transforms,
        target_transform=target_transforms,
        split_name=split_name,
        download=True,
    )

    for i in range(len(dataset_instance)):
        item = dataset_instance[i]
        x, y = item
        assert len(x["image"].shape) == 3
        assert torch.is_tensor(x["image"])
        assert torch.is_tensor(y["image"])
        break
