from hydra.core.config_store import ConfigStore

from .few_shot_classification import (
    AircraftFewShotDataModuleConfig,
    AircraftMultiViewFewShotDatasetConfig,
    CUB200FewShotDataModuleConfig,
    CUB200MultiViewFewShotDatasetConfig,
    DTDFewShotDataModuleConfig,
    DTDMultiViewFewShotDatasetConfig,
    FungiFewShotDataModuleConfig,
    GermanTrafficSignsFewShotDataModuleConfig,
    GermanTrafficSignsMultiViewFewShotDatasetConfig,
    MSCOCOFewShotDataModuleConfig,
    OmniglotFewShotDataModuleConfig,
    OmniglotMultiViewFewShotDatasetConfig,
    QuickDrawFewShotDataModuleConfig,
    QuickDrawMultiViewFewShotDatasetConfig,
    VGGFlowersFewShotDataModuleConfig,
    VGGFlowersMultiViewFewShotDatasetConfig,
    OmniglotMultiViewFewShotDataModuleConfig,
    CUB200MultiViewFewShotDataModuleConfig,
    AircraftMultiViewFewShotDataModuleConfig,
    QuickDrawMultiViewFewShotDataModuleConfig,
    DTDMultiViewFewShotDataModuleConfig,
    GermanTrafficSignsMultiViewFewShotDataModuleConfig,
    VGGFlowersMultiViewFewShotDataModuleConfig,
)
from .standard_classification import (
    CIFAR10DataModuleConfig,
    CIFAR100DataModuleConfig,
    OmniglotDataModuleConfig,
)


def add_datamodule_configs(config_store: ConfigStore):
    config_store.store(
        group="datamodule",
        name="OmniglotStandardClassification",
        node=OmniglotDataModuleConfig,
    )

    config_store.store(
        group="datamodule",
        name="CIFAR10StandardClassification",
        node=CIFAR10DataModuleConfig,
    )

    config_store.store(
        group="datamodule",
        name="CIFAR100StandardClassification",
        node=CIFAR100DataModuleConfig,
    )

    config_store.store(
        group="datamodule",
        name="OmniglotFewShotClassification",
        node=OmniglotFewShotDataModuleConfig,
    )

    config_store.store(
        group="datamodule",
        name="OmniglotMultiViewFewShotClassification",
        node=OmniglotMultiViewFewShotDataModuleConfig,
    )

    config_store.store(
        group="datamodule",
        name="CUB200FewShotClassification",
        node=CUB200FewShotDataModuleConfig,
    )

    config_store.store(
        group="datamodule",
        name="CUB200MultiViewFewShotClassification",
        node=CUB200MultiViewFewShotDataModuleConfig,
    )

    config_store.store(
        group="datamodule",
        name="AircraftFewShotClassification",
        node=AircraftFewShotDataModuleConfig,
    )

    config_store.store(
        group="datamodule",
        name="AircraftMultiViewFewShotClassification",
        node=AircraftMultiViewFewShotDataModuleConfig,
    )

    config_store.store(
        group="datamodule",
        name="QuickDrawFewShotClassification",
        node=QuickDrawFewShotDataModuleConfig,
    )

    config_store.store(
        group="datamodule",
        name="QuickDrawMultiViewFewShotClassification",
        node=QuickDrawMultiViewFewShotDataModuleConfig,
    )

    config_store.store(
        group="datamodule",
        name="DTDFewShotClassification",
        node=DTDFewShotDataModuleConfig,
    )

    config_store.store(
        group="datamodule",
        name="DTDMultiViewFewShotClassification",
        node=DTDMultiViewFewShotDataModuleConfig,
    )

    config_store.store(
        group="datamodule",
        name="GermanTrafficSignsFewShotClassification",
        node=GermanTrafficSignsFewShotDataModuleConfig,
    )

    config_store.store(
        group="datamodule",
        name="GermanTrafficSignsMultiViewFewShotClassification",
        node=GermanTrafficSignsMultiViewFewShotDataModuleConfig,
    )

    config_store.store(
        group="datamodule",
        name="VGGFlowersFewShotClassification",
        node=VGGFlowersFewShotDataModuleConfig,
    )

    config_store.store(
        group="datamodule",
        name="VGGFlowersMultiViewFewShotClassification",
        node=VGGFlowersMultiViewFewShotDataModuleConfig,
    )

    config_store.store(
        group="datamodule",
        name="FungiFewShotClassification",
        node=FungiFewShotDataModuleConfig,
    )

    config_store.store(
        group="datamodule",
        name="MSCOCOFewShotClassification",
        node=MSCOCOFewShotDataModuleConfig,
    )

    return config_store
