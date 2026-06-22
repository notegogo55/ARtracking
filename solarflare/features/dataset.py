"""Labeled per-AR sequence dataset: leakage-safe windowing + GOES labeling.

Label rule (DeFN-style): a sequence issued at t0 is POSITIVE iff its AR
produces a GOES flare of class >= the threshold with peak_time strictly inside
(t0, t0 + lead_hours]. Features use rows with time <= t0 only — verified by an
explicit poison-the-future test in tests/test_sequences.py.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from solarflare.data.goes_events import goes_class_to_flux
from solarflare.features.extract import FEATURE_VERSION

log = logging.getLogger(__name__)


def _biggest_in_window(
    events: pd.DataFrame, noaa: int, t0: pd.Timestamp, lead_hours: float
) -> tuple[float, str]:
    """(peak GOES flux, its class string) for the AR strictly inside (t0, t0+lead].

    Returns (0.0, "") when no qualifying flare. The peak flux makes the binary
    label for ANY class threshold C derivable as ``flux >= goes_class_to_flux(C)``.
    """
    if events.empty:
        return 0.0, ""
    peaks = pd.to_datetime(events["peak_time"])
    in_window = (
        (events["noaa_ar"].astype(int) == int(noaa))
        & (peaks > t0)
        & (peaks <= t0 + pd.Timedelta(hours=lead_hours))
    )
    subset = events[in_window]
    if subset.empty:
        return 0.0, ""
    fluxes = subset["goes_class"].map(goes_class_to_flux)
    idx = fluxes.idxmax()
    return float(fluxes.loc[idx]), str(subset.loc[idx, "goes_class"])


def label_for_issuance(
    events: pd.DataFrame,
    noaa: int,
    t0: pd.Timestamp,
    lead_hours: float,
    min_class: str = "M1.0",
) -> tuple[int, str]:
    """(label, biggest class in lead window) for one issuance time + class threshold."""
    flux, biggest = _biggest_in_window(events, noaa, t0, lead_hours)
    if not biggest:
        return 0, ""
    return int(flux >= goes_class_to_flux(min_class)), biggest


def grid_label_name(lead_hours: float, min_class: str) -> str:
    """Canonical per-cell label column name, e.g. (24, 'M1.0') -> 'label_24h_M1.0'."""
    return f"label_{lead_hours:g}h_{min_class}"


def build_sequences(
    features: pd.DataFrame,
    events: pd.DataFrame,
    noaa: int,
    harp: int,
    window_name: str,
    lookback_steps: int,
    lead_hours: float,
    min_class: str = "M1.0",
    lon_series: pd.Series | None = None,
    max_lon_deg: float = 65.0,
    min_valid_fraction: float = 0.8,
    lead_grid: list[float] | None = None,
    class_grid: list[str] | None = None,
) -> tuple[np.ndarray, pd.DataFrame, list[str]]:
    """Sliding leakage-safe windows over fixed-cadence features.

    `features` must be time-sorted at fixed cadence with a 'time' column.
    `lon_series` (optional) is indexed by time and gives the AR's Stonyhurst
    longitude; issuance times with |lon| > max_lon_deg are skipped.

    The primary `label` column uses (lead_hours, min_class). `lead_grid` x
    `class_grid` (default: just the primary cell) additionally emit one binary
    column per (horizon, class) cell, named `label_{H}h_{C}` — the {24,72}h x
    {>=M,>=X} evaluation grid. Labels never enter X, so the leakage guard holds
    for every cell.
    Returns (X [n, steps, n_feat] float32, samples table, feature names).
    """
    lead_grid = lead_grid or [lead_hours]
    class_grid = class_grid or [min_class]
    grid_leads = sorted({lead_hours, *lead_grid})
    class_flux = {c: goes_class_to_flux(c) for c in {min_class, *class_grid}}

    feature_cols = [c for c in features.columns if c != "time"]
    times = pd.to_datetime(features["time"]).reset_index(drop=True)
    values = features[feature_cols].to_numpy(dtype=np.float32)

    sequences: list[np.ndarray] = []
    rows: list[dict] = []
    for end_idx in range(lookback_steps - 1, len(features)):
        t0 = times[end_idx]
        window = values[end_idx - lookback_steps + 1 : end_idx + 1]
        window_times = times[end_idx - lookback_steps + 1 : end_idx + 1]
        assert (window_times <= t0).all(), "windowing bug: future row leaked into X"

        lon_t0 = np.nan
        if lon_series is not None and len(lon_series):
            # normalize both sides to int64 nanoseconds: pandas 3 may carry
            # datetime indexes at us resolution while Timestamp.value is ns
            xp = pd.DatetimeIndex(lon_series.index).as_unit("ns").asi8
            x = pd.Timestamp(t0).as_unit("ns").value
            lon_t0 = float(np.interp(x, xp, lon_series.to_numpy(dtype=float)))
            if abs(lon_t0) > max_lon_deg:
                continue
        valid_fraction = float(np.mean(np.isfinite(window)))
        if valid_fraction < min_valid_fraction:
            continue
        label, biggest = label_for_issuance(events, noaa, t0, lead_hours, min_class)
        # peak flux per horizon -> binary label for every (horizon, class) cell
        peak_flux = {h: _biggest_in_window(events, noaa, t0, h)[0] for h in grid_leads}
        grid_cols = {
            grid_label_name(h, c): int(peak_flux[h] >= class_flux[c])
            for h in lead_grid
            for c in class_grid
        }
        sequences.append(window)
        rows.append(
            {
                "sample_id": f"harp{harp:05d}_{t0:%Y%m%dT%H%M}",
                "window": window_name,
                "harp": harp,
                "noaa": noaa,
                "t0": t0,
                "t_first": window_times.iloc[0],
                "label": label,
                "biggest_class_in_lead": biggest,
                "lon_t0_deg": lon_t0,
                "valid_fraction": round(valid_fraction, 4),
                **grid_cols,
            }
        )
    X = (
        np.stack(sequences).astype(np.float32)
        if sequences
        else np.empty((0, lookback_steps, len(feature_cols)), dtype=np.float32)
    )
    return X, pd.DataFrame(rows), feature_cols


FEATURE_DESCRIPTIONS = {
    "aia_*_max": "max DN/s inside the AR mask for that AIA channel (NOT mean)",
    "flux_total": "sum of |B_los| over the AR mask [G*px; CEA grid is equal-area]",
    "signed_flux": "sum of B_los over the AR mask [G*px]",
    "b_peak": "max |B_los| inside the AR mask [G]",
    "area_px": "AR mask area [CEA pixels]",
    "*_d1": "backward 1-step difference of the base feature (per cadence step)",
}


def write_dataset(
    out_dir: str | Path,
    X: np.ndarray,
    samples: pd.DataFrame,
    feature_names: list[str],
    config_meta: dict,
) -> dict:
    """Write X.npz + samples.parquet + data_dictionary.json + stats.json."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / "X.npz", X=X, feature_names=np.array(feature_names))
    samples.to_parquet(out_dir / "samples.parquet", index=False)

    n = len(samples)
    n_pos = int(samples["label"].sum()) if n else 0
    missing_per_feature = {
        name: float(np.mean(~np.isfinite(X[:, :, j]))) if n else np.nan
        for j, name in enumerate(feature_names)
    }
    grid_cols = [c for c in samples.columns if c.startswith("label_")] if n else []
    grid_positive = {
        c: {"n_positive": int(samples[c].sum()), "rate": float(samples[c].mean())}
        for c in sorted(grid_cols)
    }
    stats = {
        "n_sequences": n,
        "n_positive": n_pos,
        "n_negative": n - n_pos,
        "positive_rate": (n_pos / n) if n else np.nan,
        "n_ars": int(samples["harp"].nunique()) if n else 0,
        "windows": sorted(samples["window"].unique().tolist()) if n else [],
        "missing_fraction_overall": float(np.mean(~np.isfinite(X))) if n else np.nan,
        "missing_fraction_per_feature": missing_per_feature,
        "grid_label_positive": grid_positive,
    }
    dictionary = {
        "dataset_shape": list(X.shape),
        "feature_version": FEATURE_VERSION,
        "feature_names": feature_names,
        "feature_descriptions": FEATURE_DESCRIPTIONS,
        "label": (
            "1 iff a GOES flare of the AR with class >= threshold peaks strictly "
            "inside (t0, t0+lead_hours]; features use rows with time <= t0 only"
        ),
        "leakage_guard": (
            "right-edge-labeled resampling, backward-only gradients, "
            "poison-the-future test in tests/test_sequences.py"
        ),
        "grid_labels": (
            "label_{H}h_{C} = 1 iff a GOES flare >= class C peaks strictly inside "
            "(t0, t0+H h]. One column per (horizon, class) evaluation cell "
            "(forecast.lead_grid x forecast.class_grid); the primary `label` aliases "
            "the (lead_hours, flare_class_threshold) cell."
        ),
        "samples_table": {
            "sample_id": "harp + issuance time",
            "t0": "issuance time (UTC); last feature timestamp in the window",
            "t_first": "first feature timestamp in the window",
            "label": "binary target (see 'label')",
            "label_{H}h_{C}": "per-cell binary target (see 'grid_labels')",
            "biggest_class_in_lead": "largest GOES class in the lead window ('' if none)",
            "lon_t0_deg": "AR Stonyhurst longitude at t0 (NaN if unavailable)",
            "valid_fraction": "fraction of finite cells in the feature window",
        },
        **config_meta,
    }
    (out_dir / "data_dictionary.json").write_text(
        json.dumps(dictionary, indent=2, default=str), encoding="utf-8"
    )
    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2, default=str), encoding="utf-8")
    log.info("dataset written to %s: %s", out_dir, stats)
    return stats
