import multiprocessing

import hydra.utils
import pytest

from gate.base.utils.loggers import get_logger
from gate.configs.datamodule.base import DataLoaderConfig
from gate.configs.datamodule.few_shot_classification import (
    DTDFewShotDatasetConfig,
    FewShotDataModuleConfig,
    FewShotTransformConfig,
)
from gate.configs.datasets.transforms import (
    dtd_query_set_transforms,
    dtd_support_set_transforms,
)

log = get_logger(__name__, set_default_handler=True)


@pytest.mark.parametrize(
    "datamodule_config_class",
    [
        FewShotDataModuleConfig,
    ],
)
@pytest.mark.parametrize("batch_size", [multiprocessing.cpu_count() * 2])
@pytest.mark.parametrize("num_workers", [multiprocessing.cpu_count()])
@pytest.mark.parametrize("pin_memory", [True])
@pytest.mark.parametrize("drop_last", [True])
@pytest.mark.parametrize("shuffle", [True])
@pytest.mark.parametrize("prefetch_factor", [2])
@pytest.mark.parametrize("persistent_workers", [True])
@pytest.mark.parametrize("variable_num_samples_per_class", [True, False])
@pytest.mark.parametrize("variable_num_classes_per_set", [True, False])
def test_aircraft_fewshot_datamodules(
    datamodule_config_class,
    batch_size,
    num_workers,
    pin_memory,
    drop_last,
    shuffle,
    prefetch_factor,
    persistent_workers,
    variable_num_samples_per_class,
    variable_num_classes_per_set,
):
    dataset_config = DTDFewShotDatasetConfig(
        dataset_root=".test/datasets/dtd",
        split_name="train",
        download=True,
        num_episodes=100,
        variable_num_samples_per_class=variable_num_samples_per_class,
        variable_num_classes_per_set=variable_num_classes_per_set,
        num_classes_per_set=6,
        num_samples_per_class=6,
        support_to_query_ratio=0.75,
        rescan_cache=False,
    )

    transform_train = FewShotTransformConfig(
        support_set_input_transform=dtd_support_set_transforms(),
        query_set_input_transform=dtd_query_set_transforms(),
        support_set_target_transform=None,
        query_set_target_transform=None,
    )

    data_loader_config = DataLoaderConfig(
        train_batch_size=1,
        val_batch_size=1,
        test_batch_size=1,
        pin_memory=pin_memory,
        train_drop_last=drop_last,
        eval_drop_last=drop_last,
        train_shuffle=shuffle,
        eval_shuffle=shuffle,
        prefetch_factor=prefetch_factor,
        persistent_workers=persistent_workers,
        num_workers=num_workers,
    )

    datamodule_config = datamodule_config_class(
        dataset_config=dataset_config,
        data_loader_config=data_loader_config,
        transform_train=transform_train,
        transform_eval=transform_train,
    )

    datamodule = hydra.utils.instantiate(datamodule_config, _recursive_=False)

    datamodule.setup(stage="fit")
    datamodule.setup(stage="test")

    for idx, item in enumerate(datamodule.test_dataloader()):
        x, y = item
        support_set_inputs = x["image"]["support_set"]
        support_set_targets = y["image"]["support_set"]
        query_set_inputs = x["image"]["query_set"]
        query_set_targets = y["image"]["query_set"]

        assert support_set_inputs.shape[1] == support_set_targets.shape[1]
        assert query_set_inputs.shape[1] == query_set_targets.shape[1]

        assert set(
            int(item) for item in list(support_set_targets[0].numpy())
        ) == set(int(item) for item in list(query_set_targets[0].numpy()))

        break

    for idx, item in enumerate(datamodule.train_dataloader()):
        x, y = item
        support_set_inputs = x["image"]["support_set"]
        support_set_targets = y["image"]["support_set"]
        query_set_inputs = x["image"]["query_set"]
        query_set_targets = y["image"]["query_set"]

        assert support_set_inputs.shape[1] == support_set_targets.shape[1]
        assert query_set_inputs.shape[1] == query_set_targets.shape[1]

        assert set(
            int(item) for item in list(support_set_targets[0].numpy())
        ) == set(int(item) for item in list(query_set_targets[0].numpy()))

        break

    for idx, item in enumerate(datamodule.val_dataloader()):
        x, y = item
        support_set_inputs = x["image"]["support_set"]
        support_set_targets = y["image"]["support_set"]
        query_set_inputs = x["image"]["query_set"]
        query_set_targets = y["image"]["query_set"]

        assert support_set_inputs.shape[1] == support_set_targets.shape[1]
        assert query_set_inputs.shape[1] == query_set_targets.shape[1]

        assert set(
            int(item) for item in list(support_set_targets[0].numpy())
        ) == set(int(item) for item in list(query_set_targets[0].numpy()))

        break
