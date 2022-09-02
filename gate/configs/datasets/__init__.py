from hydra.core.config_store import ConfigStore

from gate.configs.datasets.transforms import (
    RandomCropResizeCustomTransform,
    SuperClassExistingLabelsTransform,
)


def add_transform_configs(config_store: ConfigStore):
    config_store.store(
        group="additional_target_transforms",
        name="SuperClassExistingLabels",
        node=[SuperClassExistingLabelsTransform],
    )

    config_store.store(
        group="additional_input_transforms",
        name="RandomCropResizeCustomTransform",
        node=[RandomCropResizeCustomTransform],
    )

    config_store.store(
        group="additional_input_transforms",
        name="base",
        node=[],
    )

    config_store.store(
        group="additional_target_transforms",
        name="base",
        node=[],
    )

    return config_store
