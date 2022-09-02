from hydra.core.config_store import ConfigStore

from .few_shot_classification import (AircraftFewShotDataModuleConfig,
                                      CUB200FewShotDataModuleConfig,
                                      DTDFewShotDataModuleConfig,
                                      FungiFewShotDataModuleConfig,
                                      GermanTrafficSignsFewShotDataModuleConfig,
                                      MSCOCOFewShotDataModuleConfig,
                                      OmniglotFewShotDataModuleConfig,
                                      QuickDrawFewShotDataModuleConfig,
                                      VGGFlowersFewShotDataModuleConfig)
from .standard_classification import (CIFAR10DataModuleConfig,
                                      CIFAR100DataModuleConfig,
                                      OmniglotDataModuleConfig)


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
        name="CUB200FewShotClassification",
        node=CUB200FewShotDataModuleConfig,
    )

    config_store.store(
        group="datamodule",
        name="AircraftFewShotClassification",
        node=AircraftFewShotDataModuleConfig,
    )

    config_store.store(
        group="datamodule",
        name="QuickDrawFewShotClassification",
        node=QuickDrawFewShotDataModuleConfig,
    )

    config_store.store(
        group="datamodule",
        name="DTDFewShotClassification",
        node=DTDFewShotDataModuleConfig,
    )

    config_store.store(
        group="datamodule",
        name="GermanTrafficSignsFewShotClassification",
        node=GermanTrafficSignsFewShotDataModuleConfig,
    )

    config_store.store(
        group="datamodule",
        name="VGGFlowersFewShotClassification",
        node=VGGFlowersFewShotDataModuleConfig,
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
