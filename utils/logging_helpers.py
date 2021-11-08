import logging

from rich.logging import RichHandler


def get_logging(level):
    FORMAT = "%(message)s"
    logging.basicConfig(
        level=level, format=FORMAT, datefmt="[%X]", handlers=[RichHandler()]
    )

    log = logging.getLogger("rich")
    log.info("Hello, World!")
    return log
