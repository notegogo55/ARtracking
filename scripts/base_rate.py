"""Thin wrapper: `python scripts/base_rate.py` == `solarflare base-rate`."""

import sys

from solarflare.cli import app

if __name__ == "__main__":
    sys.argv = [sys.argv[0], "base-rate", *sys.argv[1:]]
    app()
