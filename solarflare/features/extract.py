"""Per-AR, per-frame feature extraction from the co-aligned sample cache.

Every channel in a sample is already pixel-aligned on the SHARP CEA grid
(Phase 1) and `ar_masks.npy` (Phase 2) lives on the same grid, so projecting
the AR boundary onto AIA is a boolean index — the WCS work is done upstream.

Feature policy (master prompt, non-negotiable): per-channel MAX intensity
inside the AR boundary, NOT the mean (the mean dilutes pre-flare signal with
quiet-Sun pixels), plus magnetic flux and area.

Leakage-relevant detail: hourly resampling labels bins by their RIGHT edge
(closed right), so the feature row stamped t contains only data from (t-1h, t].
A left-labeled bin would smuggle up to 59 min of the future into each row.
Gradients are backward differences only.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

FEATURE_VERSION = "1"

#: features aggregated with max when resampling (transients must survive)
_MAX_AGG_SUFFIXES = ("_max", "b_peak")


def frame_features(channels: dict[str, np.ndarray], mask: np.ndarray) -> dict:
    """Features for one timestep. `channels` maps array key -> 2D frame.

    Single-mask propagation contract (the project's core concept): `mask` is the
    ONE HMI-rooted AR mask, and it is the *only* region every layer is read
    through. The same boolean index drives the magnetogram (flux/area) and each
    AIA channel (max-in-mask) — there is NO per-channel re-segmentation. Every
    channel must share the mask's grid (asserted below), since a propagated mask
    is only valid on the co-registered grid. Guarded by `tests/test_propagation.py`.

    AIA keys ('aia_*') produce '<key>_max' (max inside mask). The magnetogram
    produces unsigned/signed flux (G*px), peak |B| (G) and area (px) — native
    pixel units; the CEA grid is equal-area so a physical conversion is one
    constant multiplier, deferred to the data dictionary.
    """
    mask = mask.astype(bool)
    for key, frame in channels.items():
        if np.asarray(frame).shape != mask.shape:
            raise ValueError(
                f"channel {key!r} shape {np.asarray(frame).shape} != mask shape "
                f"{mask.shape}: the HMI-rooted mask is not co-registered onto this "
                "layer (propagation contract violated)"
            )
    out: dict[str, float] = {"area_px": float(mask.sum())}
    blos = channels.get("hmi_magnetogram")
    if blos is not None:
        masked = blos[mask]
        finite = masked[np.isfinite(masked)]
        if finite.size:
            out["flux_total"] = float(np.sum(np.abs(finite)))
            out["signed_flux"] = float(np.sum(finite))
            out["b_peak"] = float(np.max(np.abs(finite)))
        else:
            out["flux_total"] = out["signed_flux"] = out["b_peak"] = np.nan
    for key, frame in channels.items():
        if not key.startswith("aia_"):
            continue
        masked = frame[mask]
        finite = masked[np.isfinite(masked)]
        out[f"{key}_max"] = float(np.max(finite)) if finite.size else np.nan
    return out


def extract_sample_features(sample, masks: np.ndarray) -> pd.DataFrame:
    """Per-frame features for a whole sample: one row per native timestep."""
    keys = sorted(sample.arrays)
    rows = []
    for i in range(sample.n_frames):
        channels = {k: np.asarray(sample.arrays[k][i]) for k in keys}
        row = {"time": pd.Timestamp(sample.times["time_utc"].iloc[i])}
        row.update(frame_features(channels, masks[i]))
        rows.append(row)
    df = pd.DataFrame(rows)
    log.info("extracted %d frame-feature rows, %d features", len(df), df.shape[1] - 1)
    return df


def resample_features(frame_df: pd.DataFrame, cadence_minutes: int) -> pd.DataFrame:
    """Fixed-cadence rows, RIGHT-edge labeled: row t summarizes (t-cadence, t].

    Max-type features take the bin max (preserve transients); flux/area take
    the bin median (slowly varying, robust to single bad frames).
    """
    df = frame_df.set_index("time").sort_index()
    rule = f"{cadence_minutes}min"
    agg = {col: ("max" if col.endswith(_MAX_AGG_SUFFIXES) else "median") for col in df.columns}
    out = df.resample(rule, label="right", closed="right").agg(agg)
    out = out.dropna(how="all").reset_index().rename(columns={"index": "time"})
    return out


def add_gradients(df: pd.DataFrame, exclude: tuple[str, ...] = ("time",)) -> pd.DataFrame:
    """Backward 1-step differences for every feature column (suffix _d1)."""
    out = df.copy()
    for col in df.columns:
        if col in exclude:
            continue
        out[f"{col}_d1"] = df[col].diff()
    return out


def build_frame_pipeline(frame_df: pd.DataFrame, cadence_minutes: int) -> pd.DataFrame:
    """frame features -> fixed cadence -> gradients (the full leak-safe chain)."""
    return add_gradients(resample_features(frame_df, cadence_minutes))
