"""Centralized logging setup for Krilly.

Call :func:`setup_logging` once at application/script start-up.
"""

from __future__ import annotations

import logging
import os

_DEFAULT_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"
_DEFAULT_DATEFMT = "%H:%M:%S"


def setup_logging(level: int | str | None = None) -> None:
    """Configure root logging.

    The level can be overridden by the ``KRILLY_LOG_LEVEL`` environment
    variable (e.g. ``DEBUG``, ``INFO``). Defaults to ``INFO``.
    """
    if level is None:
        level = os.environ.get("KRILLY_LOG_LEVEL", "INFO")
    logging.basicConfig(level=level, format=_DEFAULT_FORMAT, datefmt=_DEFAULT_DATEFMT)


def get_logger(name: str) -> logging.Logger:
    """Return a module logger (thin wrapper around :func:`logging.getLogger`)."""
    return logging.getLogger(name)
