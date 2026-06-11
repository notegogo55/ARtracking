"""Structured logging setup (stdlib logging; no print statements in pipeline code)."""

from __future__ import annotations

import logging
import sys

_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"


def setup_logging(level: int | str = logging.INFO) -> None:
    """Configure root logging once; safe to call repeatedly (idempotent)."""
    root = logging.getLogger()
    if root.handlers:  # already configured (e.g. by pytest or a second CLI call)
        root.setLevel(level)
        return
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter(_FORMAT, datefmt="%Y-%m-%d %H:%M:%S"))
    root.addHandler(handler)
    root.setLevel(level)
    # Third-party chatter we never want at INFO
    for noisy in ("urllib3", "matplotlib"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
