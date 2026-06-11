"""Thin wrapper: `python scripts/check_credentials.py` == `solarflare check-credentials`."""

import sys

from solarflare.cli import app

if __name__ == "__main__":
    sys.argv = [sys.argv[0], "check-credentials", *sys.argv[1:]]
    app()
