import logging

from rich.logging import RichHandler


def get_logging(level):
    FORMAT = "%(message)s"
    logging.basicConfig(
        level=level, format=FORMAT, datefmt="[%X]", handlers=[RichHandler()]
    )

    return logging.getLogger("rich")
