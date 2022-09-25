import dotenv
import hydra
import tensorflow_datasets as tfds
from omegaconf import OmegaConf

from gate.base.utils.loggers import get_logger
from gate.base.utils.rank_zero_ops import print_config

dotenv_loaded_vars = dotenv.load_dotenv(override=True, verbose=True)

log = get_logger(__name__)

log.info(f"Loaded dotenv variables: {dotenv_loaded_vars}")


dataset = load_dataset("wikimedia/wit_base", cache_dir="/mnt/nas/datasets/")
log.info(f"Loaded dataset: {dataset}")
