import logging
import os
from typing import Optional


def create_logger(logging_dir: Optional[str], rank: int = 0) -> logging.Logger:
    """
    Create a rank-aware logger that writes to stdout and an optional log file.
    Non-zero ranks get a NullHandler to avoid log spam.
    """
    logger = logging.getLogger(f"rae.rank{rank}")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if logger.handlers:
        return logger

    if rank != 0 or logging_dir is None:
        logger.addHandler(logging.NullHandler())
        return logger

    os.makedirs(logging_dir, exist_ok=True)
    formatter = logging.Formatter(
        fmt="[\033[34m%(asctime)s\033[0m] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(os.path.join(logging_dir, "log.txt"))
    file_handler.setFormatter(formatter)

    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    return logger
