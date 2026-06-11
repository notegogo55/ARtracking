"""Append-only CSV experiment log.

Each row records one experiment/metric event. Columns are the union of all keys
ever logged; missing values are left blank, so the schema can grow over time.
"""

from __future__ import annotations

import logging
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)


def _git_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5, check=True,
        )
        return out.stdout.strip()
    except Exception:
        return "unknown"


def append_experiment_row(csv_path: str | Path, record: dict) -> Path:
    """Append `record` to the experiment CSV, adding timestamp_utc and git_sha.

    Creates the file (and parent dirs) on first use; merges columns if the
    record introduces new keys.
    """
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    row = {
        "timestamp_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "git_sha": _git_sha(),
        **record,
    }
    new_df = pd.DataFrame([row])
    if csv_path.exists():
        existing = pd.read_csv(csv_path)
        new_df = pd.concat([existing, new_df], ignore_index=True)
    new_df.to_csv(csv_path, index=False)
    log.info("experiment row appended to %s", csv_path)
    return csv_path
