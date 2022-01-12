import logging

from rich.logging import RichHandler


def get_logging(logger_level):
    logger_format = "%(message)s"
    logging.basicConfig(
        level=logger_level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler()],
    )

    return logging.getLogger("rich")
