"""Bootstrap AR detection labels from HARP metadata — no hand-labeling, no images.

A keyword-only JSOC query returns, for every HARP alive at each sample time, the
patch's Stonyhurst longitude/latitude bounds (LON_MIN/LON_MAX/LAT_MIN/LAT_MAX,
verified live 2026-06-11 against AR 11158's documented position). These boxes are
the reference for temporal tracking and the HARP overlay in the full-disk views.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

HARP_BOX_KEYS = (
    "HARPNUM, T_REC, NOAA_AR, NOAA_ARS, "
    "LON_MIN, LON_MAX, LAT_MIN, LAT_MAX, LON_FWT, LAT_FWT, OMEGA_DT"
)

BOX_COLUMNS = [
    "time",
    "harpnum",
    "noaa_ar",
    "lon_min",
    "lon_max",
    "lat_min",
    "lat_max",
    "lon_fwt",
    "lat_fwt",
    "omega_dt",
]


def parse_tai(t_rec: pd.Series) -> pd.Series:
    """JSOC T_REC strings ('2011.02.15_02:00:00_TAI') -> pandas timestamps.

    The 37 s TAI-UTC offset is irrelevant at box-label precision and is ignored.
    """
    cleaned = t_rec.str.replace("_TAI", "", regex=False).str.replace(".", "-", regex=False)
    return pd.to_datetime(cleaned, format="%Y-%m-%d_%H:%M:%S")


def tidy_harp_keywords(raw: pd.DataFrame) -> pd.DataFrame:
    """drms keyword table -> tidy box table (BOX_COLUMNS), dropping unusable rows."""
    df = raw.rename(columns=str.upper).copy()
    out = pd.DataFrame(
        {
            "time": parse_tai(df["T_REC"].astype(str)),
            "harpnum": pd.to_numeric(df["HARPNUM"], errors="coerce"),
            "noaa_ar": pd.to_numeric(df["NOAA_AR"], errors="coerce").fillna(0).astype(int),
            "lon_min": pd.to_numeric(df["LON_MIN"], errors="coerce"),
            "lon_max": pd.to_numeric(df["LON_MAX"], errors="coerce"),
            "lat_min": pd.to_numeric(df["LAT_MIN"], errors="coerce"),
            "lat_max": pd.to_numeric(df["LAT_MAX"], errors="coerce"),
            "lon_fwt": pd.to_numeric(df["LON_FWT"], errors="coerce"),
            "lat_fwt": pd.to_numeric(df["LAT_FWT"], errors="coerce"),
            "omega_dt": pd.to_numeric(df["OMEGA_DT"], errors="coerce"),
        }
    )
    n0 = len(out)
    out = out.dropna(subset=["harpnum", "lon_min", "lon_max", "lat_min", "lat_max"])
    out = out[(out["lon_max"] > out["lon_min"]) & (out["lat_max"] > out["lat_min"])]
    out["harpnum"] = out["harpnum"].astype(int)
    if n0 - len(out):
        log.info("dropped %d/%d HARP rows with missing/degenerate geometry", n0 - len(out), n0)
    return out.sort_values(["time", "harpnum"]).reset_index(drop=True)


def fetch_harp_boxes(
    start: datetime,
    end: datetime,
    cadence_hours: int,
    cache_dir: str | Path,
    series: str = "hmi.sharp_720s",
) -> pd.DataFrame:
    """All HARP boxes alive in [start, end] at the given cadence (keyword query only)."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / (
        f"harp_boxes_{start:%Y%m%dT%H%M}_{end:%Y%m%dT%H%M}_{cadence_hours}h.csv"
    )
    if cache_file.exists():
        return pd.read_csv(cache_file, parse_dates=["time"])

    import drms

    from solarflare.data.jsoc_fetch import tai_str

    ds = f"{series}[][{tai_str(start)}-{tai_str(end)}@{cadence_hours}h]"
    log.info("querying HARP boxes: %s", ds)
    raw = drms.Client().query(ds, key=HARP_BOX_KEYS)
    if raw is None or len(raw) == 0:
        boxes = pd.DataFrame(columns=BOX_COLUMNS)
    else:
        boxes = tidy_harp_keywords(raw)
    boxes.to_csv(cache_file, index=False)
    log.info("cached %d HARP boxes to %s", len(boxes), cache_file)
    return boxes


def boxes_to_pixels(boxes: pd.DataFrame, fulldisk_map) -> pd.DataFrame:
    """Project Stonyhurst boxes onto a full-disk map's pixel grid via its WCS.

    Returns a copy with x_min/x_max/y_min/y_max pixel columns (handles HMI's
    CROTA2=180 automatically because the map WCS does). Boxes are the pixel
    bounding box of the 4 heliographic corners.
    """
    import astropy.units as u
    from astropy.coordinates import SkyCoord
    from sunpy.coordinates import HeliographicStonyhurst

    out = boxes.copy()
    if out.empty:
        for col in ("x_min", "x_max", "y_min", "y_max"):
            out[col] = []
        return out
    frame = HeliographicStonyhurst(obstime=fulldisk_map.date)
    lons = np.column_stack([out["lon_min"], out["lon_max"], out["lon_min"], out["lon_max"]])
    lats = np.column_stack([out["lat_min"], out["lat_min"], out["lat_max"], out["lat_max"]])
    corners = SkyCoord(lons * u.deg, lats * u.deg, frame=frame)
    x, y = fulldisk_map.wcs.world_to_pixel(corners)
    out["x_min"] = np.min(x, axis=1)
    out["x_max"] = np.max(x, axis=1)
    out["y_min"] = np.min(y, axis=1)
    out["y_max"] = np.max(y, axis=1)
    return out
