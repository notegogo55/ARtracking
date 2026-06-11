"""Preprocessing: timestamps, exposure normalization, WCS reprojection, QA flags.

Geometry note: each AIA cutout frame is reprojected onto the WCS of the
time-matched HMI SHARP CEA frame. The HARP patch itself tracks the AR across
the disk (differential rotation handled by the SHARP pipeline), and the AIA
cutouts are requested with JSOC tracking, so the resulting (T, H, W) stacks are
co-rotating and pixel-aligned with the magnetogram. `aiapy.calibrate.register`
is intentionally NOT applied: it is a full-disk lev1->1.5 helper, and the
reprojection here absorbs the same rotation/plate-scale alignment via WCS.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


def read_date_obs(path: str | Path) -> pd.Timestamp:
    """DATE-OBS of a FITS file (checks HDU 1 for Rice-compressed JSOC files, then 0)."""
    from astropy.io import fits

    for hdu_index in (1, 0):
        try:
            header = fits.getheader(path, hdu_index)
        except IndexError:
            continue
        value = header.get("DATE-OBS") or header.get("T_OBS")
        if value:
            return pd.Timestamp(str(value).replace("_TAI", "").replace("Z", ""))
    raise ValueError(f"no DATE-OBS/T_OBS found in {path}")


def match_nearest(
    target_times: pd.Series, candidate_times: pd.Series, tolerance_seconds: float
) -> np.ndarray:
    """For each target time, index of the nearest candidate within tolerance, else -1."""
    targets = pd.to_datetime(target_times).to_numpy()
    cands = pd.to_datetime(candidate_times).to_numpy()
    out = np.full(len(targets), -1, dtype=int)
    if len(cands) == 0:
        return out
    order = np.argsort(cands)
    sorted_cands = cands[order]
    pos = np.searchsorted(sorted_cands, targets)
    tol = np.timedelta64(int(tolerance_seconds * 1e9), "ns")
    for i, p in enumerate(pos):
        best, best_dt = -1, tol
        for j in (p - 1, p):
            if 0 <= j < len(sorted_cands):
                dt = abs(targets[i] - sorted_cands[j])
                if dt <= best_dt:
                    best, best_dt = order[j], dt
        out[i] = best
    return out


def exposure_normalize(smap) -> np.ndarray:
    """Map data in DN/s (float32). Frames with unusable exposure become all-NaN.

    EXPTIME = 0 happens in bulk during SDO eclipse seasons (Earth partially
    occults the detector); such frames are also photometrically invalid, so
    masking them (NaN -> QA high_nan flag, hourly resampling skips them) is
    correct. Returning raw DN here would silently mix units (~3x scale).
    """
    data = np.asarray(smap.data, dtype=np.float32)
    exptime = smap.exposure_time
    seconds = float(exptime.to_value("s")) if exptime is not None else np.nan
    if not np.isfinite(seconds) or seconds <= 0:
        log.warning("invalid exposure time %r for %s; masking frame as NaN",
                    seconds, smap.name)
        return np.full(data.shape, np.nan, dtype=np.float32)
    return data / np.float32(seconds)


def reproject_to_target(src_map, target_map) -> tuple[np.ndarray, float]:
    """Reproject `src_map` onto `target_map`'s WCS grid.

    Returns (float32 array with target shape, coverage = fraction of target
    pixels that received valid source data).
    """
    out_map, footprint = src_map.reproject_to(target_map.wcs, return_footprint=True)
    data = np.asarray(out_map.data, dtype=np.float32)
    coverage = float(np.mean(footprint > 0))
    return data, coverage


def frame_qa(
    data: np.ndarray,
    quality: int | None,
    max_nan_fraction: float,
    coverage: float | None = None,
    min_coverage: float = 0.0,
) -> dict:
    """QA flags for one frame. Flags mark frames; nothing is dropped here."""
    nan_fraction = float(np.mean(~np.isfinite(data))) if data.size else 1.0
    flags = {
        "nan_fraction": round(nan_fraction, 4),
        "quality": int(quality) if quality is not None else 0,
        "coverage": round(coverage, 4) if coverage is not None else 1.0,
        "bad_quality": bool(quality),
        "high_nan": nan_fraction > max_nan_fraction,
        "low_coverage": coverage is not None and coverage < min_coverage,
    }
    flags["flagged"] = flags["bad_quality"] or flags["high_nan"] or flags["low_coverage"]
    return flags


def robust_stats(stack: np.ndarray) -> dict:
    """Per-channel robust statistics for later normalization (computed on finite pixels)."""
    finite = stack[np.isfinite(stack)]
    if finite.size == 0:
        return {"median": None, "p01": None, "p99": None, "p999": None, "finite_frac": 0.0}
    # subsample very large stacks for speed; percentiles are insensitive to this
    if finite.size > 2_000_000:
        rng = np.random.default_rng(0)
        finite = rng.choice(finite, 2_000_000, replace=False)
    p01, med, p99, p999 = np.percentile(finite, [1, 50, 99, 99.9])
    return {
        "median": float(med),
        "p01": float(p01),
        "p99": float(p99),
        "p999": float(p999),
        "finite_frac": float(np.mean(np.isfinite(stack))),
    }
