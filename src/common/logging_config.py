"""Shared logging setup (PLAN.md rule 8)."""

import logging
import sys

_LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"
_configured = False


def _configure_root(level: int) -> None:
    global _configured
    if _configured:
        return
    logging.basicConfig(
        level=level,
        format=_LOG_FORMAT,
        stream=sys.stdout,
    )
    _configured = True


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Return a module-level logger with consistent formatting across the project."""
    _configure_root(level)
    logger = logging.getLogger(name)
    logger.setLevel(level)
    return logger
