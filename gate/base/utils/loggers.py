import logging

from rich.logging import RichHandler


def get_logger(name=__name__) -> logging.Logger:
    """Initializes multi-GPU-friendly python command line logger."""

    logger = logging.getLogger(name)

    logger.setLevel(logging.INFO)
    ch = RichHandler()
    ch.setLevel(logging.INFO)

    # create formatter
    formatter = logging.Formatter("%(levelname)s - %(message)s")

    # add formatter to ch
    ch.setFormatter(formatter)

    # add ch to logger
    logger.addHandler(ch)

    # this ensures all logging levels get marked with the rank zero decorator
    # otherwise logs would get multiplied for each GPU process in multi-GPU setup

    return logger
