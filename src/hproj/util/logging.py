import logging
import os
from pathlib import Path
import sys


def setup_logging(log_file: Path | None) -> logging.Logger:
    """
    Configure a logger that writes to stdout and (optionally) a file.
    Returns a module-level logger instance.
    """
    logger = logging.getLogger("hdims.projection")
    logger.setLevel(logging.INFO)

    # avoid duplicate handlers if Click re-invokes or tests import repeatedly.
    if logger.handlers:
        return logger

    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")

    # stdout handler
    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.INFO)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    # file handler (optional)
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.INFO)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    # prevent propagation to root logger (avoids double printing if root configured elsewhere)
    logger.propagate = False
    return logger
