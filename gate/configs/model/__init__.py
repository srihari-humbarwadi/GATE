from hydra.core.config_store import ConfigStore

from .clip_model_config import (
    CLIPModelGenericPretrainedConfig,
    CLIPModelGenericScratchConfig,
)

from .tali_model_config import (
    TALIModelGenericPretrainedConfig,
    TALIModelGenericScratchConfig,
)

from .timm_model_configs import (
    TimmImageResNet18Config,
)


def add_model_configs(config_store: ConfigStore):
    config_store.store(
        group="model",
        name="clip-generic-pretrained",
        node=CLIPModelGenericPretrainedConfig,
    )

    config_store.store(
        group="model",
        name="clip-generic-scratch",
        node=CLIPModelGenericScratchConfig,
    )

    config_store.store(
        group="model",
        name="tali-generic-pretrained",
        node=TALIModelGenericPretrainedConfig,
    )

    config_store.store(
        group="model",
        name="tali-generic-scratch",
        node=TALIModelGenericScratchConfig,
    )

    config_store.store(
        group="model",
        name="timm-image-resnet18",
        node=TimmImageResNet18Config,
    )

    return config_store
