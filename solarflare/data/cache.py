"""Per-sample cache: float32 npy stacks + CSV tables + meta.json in one directory.

Layout of a sample directory (one AR over one window):
    hmi_magnetogram.npy, hmi_continuum.npy, aia_0094.npy, ...   (T, H, W) float32
    times.csv     frame_idx, time_utc (the common HMI-anchored timeline)
    qa.csv        one row per (frame, channel) with QA flags
    labels.csv    GOES events for this AR (EVENT_COLUMNS)
    meta.json     ids, shapes, channel stats, config hash, provenance
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

META_NAME = "meta.json"

#: Derived per-sample products (not instrument channels). Excluded from
#: SampleData.arrays so re-segmenting can overwrite them on Windows, where a
#: live read-only mmap blocks writing the file.
DERIVED_ARRAYS = frozenset({"ar_masks"})


def channel_key(channel: int) -> str:
    return f"aia_{channel:04d}"


@dataclass
class SampleData:
    """A loaded sample. Arrays are memory-mapped read-only."""

    sample_dir: Path
    arrays: dict[str, np.ndarray]
    times: pd.DataFrame
    qa: pd.DataFrame
    labels: pd.DataFrame
    meta: dict = field(default_factory=dict)

    @property
    def n_frames(self) -> int:
        return len(self.times)


def write_sample(
    sample_dir: str | Path,
    arrays: dict[str, np.ndarray],
    times: pd.DataFrame,
    qa: pd.DataFrame,
    labels: pd.DataFrame,
    meta: dict,
) -> Path:
    """Write a complete sample cache; overwrites existing files in place."""
    sample_dir = Path(sample_dir)
    sample_dir.mkdir(parents=True, exist_ok=True)
    shapes = {}
    for key, arr in arrays.items():
        arr = np.asarray(arr, dtype=np.float32)
        np.save(sample_dir / f"{key}.npy", arr)
        shapes[key] = list(arr.shape)
    times.to_csv(sample_dir / "times.csv", index=False)
    qa.to_csv(sample_dir / "qa.csv", index=False)
    labels.to_csv(sample_dir / "labels.csv", index=False)
    meta = {**meta, "array_shapes": shapes}
    (sample_dir / META_NAME).write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    log.info("sample cached to %s (%d arrays, %d frames)", sample_dir, len(arrays), len(times))
    return sample_dir


def load_sample(sample_dir: str | Path) -> SampleData:
    """Load a cached sample (arrays memory-mapped; empty tables tolerated)."""
    sample_dir = Path(sample_dir)
    meta_path = sample_dir / META_NAME
    if not meta_path.exists():
        raise FileNotFoundError(f"not a sample cache (no {META_NAME}): {sample_dir}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    arrays = {
        p.stem: np.load(p, mmap_mode="r")
        for p in sorted(sample_dir.glob("*.npy"))
        if p.stem not in DERIVED_ARRAYS
    }
    times = pd.read_csv(sample_dir / "times.csv", parse_dates=["time_utc"])

    def _read_optional(name: str) -> pd.DataFrame:
        path = sample_dir / name
        if not path.exists() or path.stat().st_size == 0:
            return pd.DataFrame()
        try:
            return pd.read_csv(path)
        except pd.errors.EmptyDataError:
            return pd.DataFrame()

    return SampleData(
        sample_dir=sample_dir,
        arrays=arrays,
        times=times,
        qa=_read_optional("qa.csv"),
        labels=_read_optional("labels.csv"),
        meta=meta,
    )
