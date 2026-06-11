"""GOES flare-event access: class parsing and HEK event-list retrieval with CSV cache.

Event lists come from the Heliophysics Event Knowledgebase (HEK) via SunPy's Fido,
restricted to GOES observations — the same catalog used later for labels.
Network access happens only in `fetch_goes_events`; everything else is pure.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)

#: GOES class letter -> peak X-ray flux (W/m^2) of the class lower bound.
_CLASS_FLUX = {"A": 1e-8, "B": 1e-7, "C": 1e-6, "M": 1e-5, "X": 1e-4}

#: Canonical event-table columns used throughout the pipeline.
EVENT_COLUMNS = ["start_time", "peak_time", "end_time", "goes_class", "noaa_ar"]


def goes_class_to_flux(goes_class: str) -> float:
    """Convert a GOES class string (e.g. 'M1.0', 'X9.3', 'C5') to peak flux in W/m^2.

    A bare letter (e.g. 'M') is treated as multiplier 1.0.
    Raises ValueError for malformed input.
    """
    s = str(goes_class).strip().upper()
    if not s or s[0] not in _CLASS_FLUX:
        raise ValueError(f"invalid GOES class: {goes_class!r}")
    base = _CLASS_FLUX[s[0]]
    rest = s[1:].strip()
    if not rest:
        return base
    try:
        mult = float(rest)
    except ValueError as err:
        raise ValueError(f"invalid GOES class multiplier in {goes_class!r}") from err
    if mult <= 0:
        raise ValueError(f"GOES class multiplier must be positive: {goes_class!r}")
    return base * mult


def load_events_csv(path: str | Path) -> pd.DataFrame:
    """Load an event CSV with EVENT_COLUMNS, parsing time columns."""
    df = pd.read_csv(path)
    missing = set(EVENT_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"event CSV {path} missing columns: {sorted(missing)}")
    for col in ("start_time", "peak_time", "end_time"):
        df[col] = pd.to_datetime(df[col])
    return df


def select_ar_events(
    events: pd.DataFrame,
    noaa: int,
    start,
    end,
) -> pd.DataFrame:
    """Events attributed to one NOAA AR with peak inside [start, end), sorted by peak."""
    if events.empty:
        return events
    peaks = pd.to_datetime(events["peak_time"])
    mask = (
        (events["noaa_ar"].astype(int) == int(noaa))
        & (peaks >= pd.Timestamp(start))
        & (peaks < pd.Timestamp(end))
    )
    return events[mask].sort_values("peak_time").reset_index(drop=True)


def fetch_goes_events(
    start: datetime,
    end: datetime,
    min_class: str = "M1.0",
    cache_dir: str | Path | None = None,
) -> pd.DataFrame:
    """Fetch GOES flare events >= `min_class` in [start, end) from HEK, with CSV cache.

    Returns a DataFrame with EVENT_COLUMNS, sorted by peak_time. Requires network
    on cache miss (run `solarflare check-credentials` first if unsure).
    """
    if cache_dir is not None:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = cache_dir / (
            f"goes_flares_{start:%Y%m%dT%H%M}_{end:%Y%m%dT%H%M}_{min_class}.csv"
        )
        if cache_file.exists():
            log.info("loading cached GOES events: %s", cache_file)
            return load_events_csv(cache_file)

    log.info("querying HEK for GOES flares >= %s in %s .. %s", min_class, start, end)
    from sunpy.net import Fido
    from sunpy.net import attrs as a

    result = Fido.search(
        a.Time(start, end),
        a.hek.EventType("FL"),
        a.hek.FL.GOESCls >= min_class,
        a.hek.OBS.Observatory == "GOES",
    )
    hek_table = result["hek"]
    if len(hek_table) == 0:
        df = pd.DataFrame(columns=EVENT_COLUMNS)
    else:
        sub = hek_table["event_starttime", "event_peaktime", "event_endtime",
                        "fl_goescls", "ar_noaanum"]
        df = pd.DataFrame(
            {
                "start_time": pd.to_datetime([str(v) for v in sub["event_starttime"]]),
                "peak_time": pd.to_datetime([str(v) for v in sub["event_peaktime"]]),
                "end_time": pd.to_datetime([str(v) for v in sub["event_endtime"]]),
                "goes_class": [str(v).strip() for v in sub["fl_goescls"]],
                "noaa_ar": [int(v) if str(v).strip().isdigit() else 0 for v in sub["ar_noaanum"]],
            }
        )
        # Defense in depth: HEK filters server-side, but enforce the threshold locally too.
        min_flux = goes_class_to_flux(min_class)
        df = df[df["goes_class"].map(goes_class_to_flux) >= min_flux]
        df = df.sort_values("peak_time").reset_index(drop=True)

    if cache_dir is not None:
        df.to_csv(cache_file, index=False)
        log.info("cached %d events to %s", len(df), cache_file)
    return df
