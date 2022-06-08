# import inspect
# import pathlib
#
# import pytest
# from pytorch_lightning import seed_everything
#
# from gate.base.utils.loggers import get_logger
# from gate.datasets.data_utils import (
#     FewShotSuperSplitSetOptions,
#     FewShotSplitSetOptions,
# )
#
# log = get_logger(__name__, set_default_handler=True)
#
#
# @pytest.mark.parametrize("variable_num_samples_per_class", [True])
# @pytest.mark.parametrize("download", [True])
# @pytest.mark.parametrize(
#     "super_split_name",
#     [
#         FewShotSuperSplitSetOptions.TRAIN,
#     ],
# )
# def test_omniglot_datasets(
#     super_split_name, variable_num_samples_per_class, download
# ):
#     seed_everything(42, workers=True)
#     input_transforms = None
#
#     argument_names = inspect.signature(
#         SingleSetOmniglotFewShotClassificationDataset.__init__
#     ).parameters.keys()
#     log.info(f"Items: {argument_names} {'input_transform' in argument_names}")
#     target_transforms = None
#
#     dataset_instance = SingleSetOmniglotFewShotClassificationDataset(
#         dataset_root=pathlib.Path("tests/data/omniglot/"),
#         super_split_name=super_split_name,
#         split_name=FewShotSplitSetOptions.SUPPORT_SET,
#         download=True,
#         num_classes_per_set=[2, 100],
#         num_samples_per_class=[1, 15],
#         variable_num_samples_per_class=variable_num_samples_per_class,
#         support_set_input_transform=input_transforms,
#         support_set_target_transform=target_transforms,
#         query_set_input_transform=input_transforms,
#         query_set_target_transform=target_transforms,
#         set_seed=0,
#     )
#
#     for i in range(len(dataset_instance)):
#         item = dataset_instance[i]
#         x, y = item
#
#         value = x.image
#         log.info(f"{value.shape}")
#
#         value = y.image
#         log.info(f"{value.shape}")
#
#         break
#
#     log.info(f"Support set length {len(dataset_instance)}")
#
#     del dataset_instance
#     dataset_instance = SingleSetOmniglotFewShotClassificationDataset(
#         dataset_root=pathlib.Path("tests/data/omniglot/"),
#         super_split_name=super_split_name,
#         split_name=FewShotSplitSetOptions.DEV_SET,
#         download=True,
#         num_classes_per_set=[2, 100],
#         num_samples_per_class=[1, 15],
#         variable_num_samples_per_class=variable_num_samples_per_class,
#         support_set_input_transform=input_transforms,
#         support_set_target_transform=target_transforms,
#         query_set_input_transform=input_transforms,
#         query_set_target_transform=target_transforms,
#         set_seed=0,
#     )
#
#     for i in range(len(dataset_instance)):
#         item = dataset_instance[i]
#         x, y = item
#
#         value = x.image
#         log.info(f"{value.shape}")
#
#         value = y.image
#         log.info(f"{value.shape}")
#
#         break
#
#     log.info(f"Val set length {len(dataset_instance)}")
#
#     del dataset_instance
#
#     dataset_instance = SingleSetOmniglotFewShotClassificationDataset(
#         dataset_root=pathlib.Path("tests/data/omniglot/"),
#         super_split_name=super_split_name,
#         split_name=FewShotSplitSetOptions.QUERY_SET,
#         download=True,
#         num_classes_per_set=[2, 100],
#         num_samples_per_class=[1, 15],
#         variable_num_samples_per_class=variable_num_samples_per_class,
#         support_set_input_transform=input_transforms,
#         support_set_target_transform=target_transforms,
#         query_set_input_transform=input_transforms,
#         query_set_target_transform=target_transforms,
#         set_seed=0,
#     )
#
#     for i in range(len(dataset_instance)):
#         item = dataset_instance[i]
#         x, y = item
#
#         value = x.image
#         log.info(f"{value.shape}")
#
#         value = y.image
#         log.info(f"{value.shape}")
#
#         break
#
#     log.info(f"Query set length {len(dataset_instance)}")
