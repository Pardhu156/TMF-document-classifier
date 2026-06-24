"""Application logging configuration."""

from __future__ import annotations

import logging
from pathlib import Path


LOG_DIRECTORY = Path(__file__).resolve().parents[1] / "logs"
LOG_DIRECTORY.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIRECTORY / "tmf_classifier.log"

LOG_FORMAT = "%(asctime)s | %(levelname)s | %(filename)s:%(lineno)d | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def _configure_logger() -> logging.Logger:
    """Create and return the shared project logger without duplicate handlers."""
    project_logger = logging.getLogger("tmf_classifier")
    project_logger.setLevel(logging.INFO)
    project_logger.propagate = False

    if project_logger.handlers:
        return project_logger

    formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    project_logger.addHandler(console_handler)
    project_logger.addHandler(file_handler)
    return project_logger


logger = _configure_logger()

