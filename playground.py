import dotenv
import hydra
import tensorflow_datasets as tfds
from omegaconf import OmegaConf

from gate.base.utils.loggers import get_logger
from gate.base.utils.rank_zero_ops import print_config

dotenv_loaded_vars = dotenv.load_dotenv(override=True, verbose=True)

log = get_logger(__name__)

log.info(f"Loaded dotenv variables: {dotenv_loaded_vars}")

dataset = tfds.load(
    "visual_domain_decathlon/vgg-flowers",
    split="test",
    shuffle_files=False,
    download=True,
    as_supervised=False,
    data_dir="playground/datasets/vgg_flowers",
    with_info=True,
)
