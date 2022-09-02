from dataclasses import MISSING, dataclass
from typing import Any, Dict, List

from gate.configs import get_module_import_path
from gate.models.timm_hub import (
    TimmImageModel,
    TimmImageModelConfigurableDepth,
    TimmImageModelConfigurableDepthAndHead,
)

input_size_dict_224_image = dict(
    image=dict(shape=dict(channels=3, width=224, height=224), dtype="float32")
)

input_size_dict_224_audio_image = dict(
    image=dict(shape=dict(channels=3, width=224, height=224), dtype="float32"),
    audio=dict(shape=dict(channels=2, width=44100), dtype="float32"),
)


@dataclass
class TimmImageModelConfig:
    pretrained: bool = MISSING
    model_name_to_download: str = MISSING
    input_shape_dict: Dict = MISSING
    global_pool: bool = False
    _target_: str = get_module_import_path(TimmImageModel)


@dataclass
class TimmImageModelWithRemovedLayersConfig(TimmImageModelConfig):
    list_of_layer_prefix_to_remove: List[str] = None
    _target_: str = get_module_import_path(TimmImageModelConfigurableDepth)


@dataclass
class TimmImageModelConfigurableDepthAndHeadConfig(
    TimmImageModelWithRemovedLayersConfig
):
    head_num_output_filters: int = 512
    head_num_hidden_filters: int = 512
    head_output_activation_fn: Any = None
    use_twin_head: bool = False
    _target_: str = get_module_import_path(
        TimmImageModelConfigurableDepthAndHead
    )


TimmImageResNet18PoolConfig = TimmImageModelConfig(
    model_name_to_download="resnet18",
    pretrained=True,
    input_shape_dict=input_size_dict_224_image,
    global_pool=True,
)

TimmImageResNet18Config = TimmImageModelConfig(
    model_name_to_download="resnet18",
    pretrained=True,
    input_shape_dict=input_size_dict_224_image,
    global_pool=False,
)

TimmImageResNet18WithRemovedLayersConfig = (
    TimmImageModelWithRemovedLayersConfig(
        model_name_to_download="resnet18",
        pretrained=True,
        input_shape_dict=input_size_dict_224_image,
        list_of_layer_prefix_to_remove=["layer4", "fc", "global_pool"],
        global_pool=False,
    )
)

TimmImageResNet18PoolWithRemovedLayersConfig = (
    TimmImageModelWithRemovedLayersConfig(
        model_name_to_download="resnet18",
        pretrained=True,
        input_shape_dict=input_size_dict_224_image,
        list_of_layer_prefix_to_remove=["layer4", "fc", "global_pool"],
        global_pool=True,
    )
)

TimmImageModelConfigurableDepthAndHeadSingleHeadConfig = (
    TimmImageModelConfigurableDepthAndHeadConfig(
        model_name_to_download="resnet18",
        pretrained=True,
        input_shape_dict=input_size_dict_224_image,
        list_of_layer_prefix_to_remove=["layer4", "fc", "global_pool"],
        global_pool=False,
        head_num_output_filters=512,
        head_num_hidden_filters=512,
        use_twin_head=False,
    )
)

TimmImageModelConfigurableDepthAndHeadTwinHeadConfig = (
    TimmImageModelConfigurableDepthAndHeadConfig(
        model_name_to_download="resnet18",
        pretrained=True,
        input_shape_dict=input_size_dict_224_image,
        list_of_layer_prefix_to_remove=["layer4", "fc", "global_pool"],
        global_pool=False,
        head_num_output_filters=512,
        head_num_hidden_filters=512,
        use_twin_head=True,
    )
)

TimmImageModelPoolConfigurableDepthAndHeadSingleHeadConfig = (
    TimmImageModelConfigurableDepthAndHeadConfig(
        model_name_to_download="resnet18",
        pretrained=True,
        input_shape_dict=input_size_dict_224_image,
        list_of_layer_prefix_to_remove=["layer4", "fc", "global_pool"],
        global_pool=True,
        head_num_output_filters=512,
        head_num_hidden_filters=512,
        use_twin_head=False,
    )
)

TimmImageModelPoolConfigurableDepthAndHeadTwinHeadConfig = (
    TimmImageModelConfigurableDepthAndHeadConfig(
        model_name_to_download="resnet18",
        pretrained=True,
        input_shape_dict=input_size_dict_224_image,
        list_of_layer_prefix_to_remove=["layer4", "fc", "global_pool"],
        global_pool=True,
        head_num_output_filters=512,
        head_num_hidden_filters=512,
        use_twin_head=True,
    )
)
