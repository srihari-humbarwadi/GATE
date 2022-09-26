from dataclasses import dataclass
from typing import Any, Optional

from gate.configs import get_module_import_path
from gate.datasets.learn2learn_hub.few_shot.fungi import (
    FungiFewShotClassificationDataset,
)
from gate.datasets.tf_hub.few_shot.aircraft import (
    AircraftFewShotClassificationDataset,
    AircraftMultiViewFewShotClassificationDataset,
)
from gate.datasets.tf_hub.few_shot.base import (
    FewShotClassificationDatasetTFDS,
    MultiViewFewShotClassificationDatasetTFDS,
)
from gate.datasets.tf_hub.few_shot.cu_birds import (
    CUB200FewShotClassificationDataset,
    CUB200MultiViewFewShotClassificationDataset,
)
from gate.datasets.tf_hub.few_shot.dtd import (
    DTDFewShotClassificationDataset,
    DTDMultiViewFewShotClassificationDataset,
)
from gate.datasets.tf_hub.few_shot.german_traffic_signs import (
    GermanTrafficSignsFewShotClassificationDataset,
    GermanTrafficSignsMultiViewFewShotClassificationDataset,
)
from gate.datasets.tf_hub.few_shot.mscoco import MSCOCOFewShotClassificationDataset
from gate.datasets.tf_hub.few_shot.omniglot import (
    OmniglotFewShotClassificationDataset,
    OmniglotMultiViewFewShotClassificationDataset,
)
from gate.datasets.tf_hub.few_shot.quickdraw import (
    QuickDrawFewShotClassificationDataset,
    QuickDrawMultiViewFewShotClassificationDataset,
)
from gate.datasets.tf_hub.few_shot.vgg_flowers import (
    VGGFlowersFewShotClassificationDataset,
    VGGFlowersMultiViewFewShotClassificationDataset,
)


@dataclass
class FewShotDatasetConfig:
    """
    Class for configuring a few shot dataset
    """

    dataset_root: str
    split_name: Optional[str] = None
    download: bool = True
    num_episodes: int = 600

    # min_num_classes_per_set: int = 1 #default 5
    # min_num_samples_per_class: int = 3 # default: 2
    # min_num_queries_per_class: int = 2 # default 1
    # num_classes_per_set: int = 5  # n_way, default 50
    # num_samples_per_class: int = 3  # n_shot, default: 10
    # num_queries_per_class: int = 2 # default 3
    # variable_num_samples_per_class: bool = False
    min_num_classes_per_set: int = 5
    min_num_samples_per_class: int = 2  # default: 2
    min_num_queries_per_class: int = 1
    num_classes_per_set: int = 50  # n_way
    num_samples_per_class: int = 10  # n support shot, default: 10
    num_queries_per_class: int = 10  # n_query, default: 10
    variable_num_samples_per_class: bool = True
    variable_num_queries_per_class: bool = False
    # TODO: Add a config default for 1-3 num query variable sampling
    variable_num_classes_per_set: bool = True
    support_set_input_transform: Any = None
    query_set_input_transform: Any = None
    support_set_target_transform: Any = None
    query_set_target_transform: Any = None
    rescan_cache: bool = False
    _target_: Any = FewShotClassificationDatasetTFDS


@dataclass
class MultiViewFewShotDatasetConfig(FewShotDatasetConfig):
    dataset_root = str = ".gate/datasets"
    min_num_classes_per_set: int = 1  # default 5
    min_num_samples_per_class: int = 3  # default: 2
    min_num_queries_per_class: int = 2  # default 1
    num_classes_per_set: int = 5  # n_way, default 50
    num_samples_per_class: int = 3  # n_shot, default: 10
    num_queries_per_class: int = 2  # default 3
    variable_num_samples_per_class: bool = False
    _target_ = get_module_import_path(MultiViewFewShotClassificationDatasetTFDS)


@dataclass
class OmniglotFewShotDatasetConfig(FewShotDatasetConfig):
    """
    Class for configuring a few shot dataset
    """

    _target_: Any = get_module_import_path(OmniglotFewShotClassificationDataset)


@dataclass
class OmniglotMultiViewFewShotDatasetConfig(MultiViewFewShotDatasetConfig):
    """
    Class for configuring a few shot dataset
    """

    _target_: Any = get_module_import_path(
        OmniglotMultiViewFewShotClassificationDataset
    )


@dataclass
class CUB200FewShotDatasetConfig(FewShotDatasetConfig):
    """
    Class for configuring a few shot dataset
    """

    _target_: Any = get_module_import_path(CUB200FewShotClassificationDataset)


@dataclass
class CUB200MultiViewFewShotDatasetConfig(MultiViewFewShotDatasetConfig):
    """
    Class for configuring a few shot dataset
    """

    _target_: Any = get_module_import_path(CUB200MultiViewFewShotClassificationDataset)


@dataclass
class AircraftFewShotDatasetConfig(FewShotDatasetConfig):
    """
    Class for configuring a few shot dataset
    """

    _target_: Any = get_module_import_path(AircraftFewShotClassificationDataset)


@dataclass
class AircraftMultiViewFewShotDatasetConfig(MultiViewFewShotDatasetConfig):
    """
    Class for configuring a few shot dataset
    """

    _target_: Any = get_module_import_path(
        AircraftMultiViewFewShotClassificationDataset
    )


@dataclass
class DTDFewShotDatasetConfig(FewShotDatasetConfig):
    """
    Class for configuring a few shot dataset
    """

    _target_: Any = get_module_import_path(DTDFewShotClassificationDataset)


@dataclass
class DTDMultiViewFewShotDatasetConfig(MultiViewFewShotDatasetConfig):
    """
    Class for configuring a few shot dataset
    """

    _target_: Any = get_module_import_path(DTDMultiViewFewShotClassificationDataset)


@dataclass
class GermanTrafficSignsFewShotDatasetConfig(FewShotDatasetConfig):
    """
    Class for configuring a few shot dataset
    """

    _target_: Any = get_module_import_path(
        GermanTrafficSignsFewShotClassificationDataset
    )


@dataclass
class GermanTrafficSignsMultiViewFewShotDatasetConfig(FewShotDatasetConfig):
    """
    Class for configuring a few shot dataset
    """

    _target_: Any = get_module_import_path(
        GermanTrafficSignsMultiViewFewShotClassificationDataset
    )


@dataclass
class VGGFlowersFewShotDatasetConfig(FewShotDatasetConfig):
    """
    Class for configuring a few shot dataset
    """

    _target_: Any = get_module_import_path(VGGFlowersFewShotClassificationDataset)


class VGGFlowersMultiViewFewShotDatasetConfig(MultiViewFewShotDatasetConfig):
    """
    Class for configuring a few shot dataset
    """

    _target_: Any = get_module_import_path(
        VGGFlowersMultiViewFewShotClassificationDataset
    )


@dataclass
class QuickDrawFewShotDatasetConfig(FewShotDatasetConfig):
    """
    Class for configuring a few shot dataset
    """

    _target_: Any = get_module_import_path(QuickDrawFewShotClassificationDataset)


class QuickDrawMultiViewFewShotDatasetConfig(MultiViewFewShotDatasetConfig):
    """
    Class for configuring a few shot dataset
    """

    _target_: Any = get_module_import_path(
        QuickDrawMultiViewFewShotClassificationDataset
    )


@dataclass
class FungiFewShotDatasetConfig(FewShotDatasetConfig):
    """
    Class for configuring a few shot dataset
    """

    _target_: Any = get_module_import_path(FungiFewShotClassificationDataset)


@dataclass
class MSCOCOFewShotDatasetConfig(FewShotDatasetConfig):
    """
    Class for configuring a few shot dataset
    """

    _target_: Any = get_module_import_path(MSCOCOFewShotClassificationDataset)
