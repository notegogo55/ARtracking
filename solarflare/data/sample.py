"""End-to-end sample builder: fetch -> co-align -> normalize -> QA -> cache -> label.

`build_sample` is the one entry point behind `solarflare fetch`: for one study
window + one target AR it downloads SHARP CEA segments and tracked AIA cutouts,
reprojects every AIA frame onto the time-matched HMI CEA grid, normalizes AIA
to DN/s, flags bad frames, attaches the GOES event list for the AR, and writes
the npy/CSV/JSON sample cache.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from solarflare import __version__
from solarflare.config import Config, StudyTarget, StudyWindow
from solarflare.data.cache import channel_key, write_sample
from solarflare.data.goes_events import fetch_goes_events, select_ar_events
from solarflare.data.harps import fetch_harp_noaa_mapping, resolve_harp
from solarflare.data.jsoc_fetch import (
    cutout_corners_from_map,
    fetch_aia_cutouts,
    fetch_sharp_cutouts,
)
from solarflare.data.preprocess import (
    exposure_normalize,
    frame_qa,
    match_nearest,
    read_date_obs,
    reproject_to_target,
    robust_stats,
)

log = logging.getLogger(__name__)

#: GOES catalog floor for the per-AR label table (>=M labeling happens in Phase D).
LABEL_CATALOG_MIN_CLASS = "C1.0"


def resolve_window_target(
    cfg: Config, window_name: str, noaa: int | None = None
) -> tuple[StudyWindow, StudyTarget]:
    window = next((w for w in cfg.study.windows if w.name == window_name), None)
    if window is None:
        raise ValueError(f"unknown study window {window_name!r}; "
                         f"known: {[w.name for w in cfg.study.windows]}")
    if not window.targets:
        raise ValueError(f"window {window_name!r} has no AR targets (quiet window?)")
    if noaa is None:
        return window, window.targets[0]
    target = next((t for t in window.targets if t.noaa == noaa), None)
    if target is None:
        raise ValueError(f"NOAA {noaa} not among targets of window {window_name!r}")
    return window, target


def sample_dir_name(harp: int, window_name: str) -> str:
    return f"harp{harp:05d}_{window_name}"


def _load_hmi_stacks(
    files: list[Path], segments: list[str]
) -> tuple[dict[str, np.ndarray], list, pd.DataFrame]:
    """Group SHARP FITS by segment, sort by time, stack. Returns (stacks, target maps, times).

    The returned maps list (magnetogram segment) defines the per-frame target WCS.
    """
    import sunpy.map

    by_segment: dict[str, list] = {seg: [] for seg in segments}
    for path in files:
        name = path.name.lower()
        seg = next((s for s in segments if s.lower() in name), None)
        if seg is None:
            log.warning("cannot assign segment for %s; skipping", path.name)
            continue
        by_segment[seg].append(sunpy.map.Map(str(path)))
    for seg, maps in by_segment.items():
        if not maps:
            raise RuntimeError(f"no files matched HMI segment {seg!r}")
        maps.sort(key=lambda m: m.date.datetime)

    anchor_seg = segments[0]
    anchor_maps = by_segment[anchor_seg]
    times = pd.DataFrame(
        {
            "frame_idx": range(len(anchor_maps)),
            "time_utc": [pd.Timestamp(m.date.datetime) for m in anchor_maps],
        }
    )
    # Definitive HARP patches keep a constant pixel size over the disk passage;
    # crop defensively to the common minimum if that assumption is ever violated.
    min_shape = tuple(
        min(m.data.shape[axis] for maps in by_segment.values() for m in maps)
        for axis in (0, 1)
    )
    stacks: dict[str, np.ndarray] = {}
    for seg, maps in by_segment.items():
        if len(maps) != len(anchor_maps):
            raise RuntimeError(
                f"segment {seg!r} has {len(maps)} frames vs {len(anchor_maps)} "
                f"for {anchor_seg!r}; JSOC export incomplete?"
            )
        arr = np.stack(
            [np.asarray(m.data[: min_shape[0], : min_shape[1]], dtype=np.float32) for m in maps]
        )
        stacks[f"hmi_{seg}"] = arr
    return stacks, anchor_maps, times


def build_sample(
    cfg: Config,
    window_name: str,
    noaa: int | None = None,
    start: datetime | None = None,
    end: datetime | None = None,
    channels: list[int] | None = None,
    email: str = "",
    skip_aia: bool = False,
    overwrite: bool = False,
) -> Path:
    """Build and cache the co-aligned sample for one AR/window. Returns the cache dir."""
    window, target = resolve_window_target(cfg, window_name, noaa)
    t0 = start or window.start
    t1 = end or window.end
    channels = channels or cfg.data.aia_channels
    if not email:
        raise ValueError("a JSOC-registered email is required (set JSOC_EMAIL or --email)")

    harp = target.harp
    if harp is None:
        mapping = fetch_harp_noaa_mapping(cfg.paths.cache_dir)
        harp = resolve_harp(mapping, target.noaa)
        log.info("resolved NOAA %d -> HARP %d via JSOC mapping", target.noaa, harp)

    raw_root = Path(cfg.paths.data_root) / "raw" / window_name
    qa_rows: list[dict] = []

    # --- HMI SHARP segments (define the timeline and the target CEA grids) ---
    sharp_files = fetch_sharp_cutouts(
        cfg.data.hmi_sharp_series, harp, t0, t1, cfg.data.hmi_segments,
        email=email, out_dir=raw_root / f"sharp_{harp}", overwrite=overwrite,
    )
    stacks, hmi_maps, times = _load_hmi_stacks(sharp_files, cfg.data.hmi_segments)
    n_frames = len(hmi_maps)
    log.info("HMI timeline: %d frames, %s .. %s, patch %s",
             n_frames, times["time_utc"].iloc[0], times["time_utc"].iloc[-1],
             stacks[f"hmi_{cfg.data.hmi_segments[0]}"].shape[1:])
    for seg in cfg.data.hmi_segments:
        arr = stacks[f"hmi_{seg}"]
        for i in range(n_frames):
            quality = hmi_maps[i].meta.get("quality", 0) if seg == cfg.data.hmi_segments[0] else 0
            qa_rows.append({
                "frame_idx": i, "channel": f"hmi_{seg}",
                "time_utc": times["time_utc"].iloc[i], "dt_seconds": 0.0,
                **frame_qa(arr[i], quality, cfg.qa.max_nan_fraction),
            })

    # --- AIA cutouts, reprojected per-frame onto the HMI CEA grid ---
    if not skip_aia:
        ref_map = hmi_maps[n_frames // 2]
        bottom_left, top_right = cutout_corners_from_map(ref_map, cfg.data.cutout_pad_arcsec)
        target_shape = stacks[f"hmi_{cfg.data.hmi_segments[0]}"].shape[1:]
        import sunpy.map

        for ch in channels:
            files = fetch_aia_cutouts(
                ch, t0, t1, cfg.data.aia_cadence_seconds, bottom_left, top_right,
                email=email, out_dir=raw_root / f"aia_{ch:04d}",
                euv_series=cfg.data.aia_euv_series, uv_series=cfg.data.aia_uv_series,
                overwrite=overwrite,
            )
            file_times = pd.Series([read_date_obs(f) for f in files])
            matches = match_nearest(
                times["time_utc"], file_times, cfg.data.aia_match_tolerance_seconds
            )
            arr = np.full((n_frames, *target_shape), np.nan, dtype=np.float32)
            for i, j in enumerate(matches):
                row = {"frame_idx": i, "channel": channel_key(ch),
                       "time_utc": times["time_utc"].iloc[i]}
                if j < 0:
                    qa_rows.append({**row, "dt_seconds": np.nan,
                                    **frame_qa(arr[i], None, cfg.qa.max_nan_fraction),
                                    "flagged": True})
                    continue
                aia_map = sunpy.map.Map(str(files[j]))
                normalized = exposure_normalize(aia_map)
                normalized_map = sunpy.map.Map(normalized, aia_map.meta)
                data, coverage = reproject_to_target(normalized_map, hmi_maps[i])
                arr[i] = data[: target_shape[0], : target_shape[1]]
                dt = abs((file_times.iloc[j] - times["time_utc"].iloc[i]).total_seconds())
                qa_rows.append({**row, "dt_seconds": dt,
                                **frame_qa(arr[i], aia_map.meta.get("quality", 0),
                                           cfg.qa.max_nan_fraction, coverage,
                                           cfg.qa.min_coverage)})
            stacks[channel_key(ch)] = arr
            log.info("AIA %d A: stacked %d/%d frames matched", ch,
                     int((matches >= 0).sum()), n_frames)

    # --- GOES labels for this AR (catalog floor C1.0; >=M labeling in Phase D) ---
    events = fetch_goes_events(
        t0 - timedelta(hours=24), t1 + timedelta(hours=24),
        min_class=LABEL_CATALOG_MIN_CLASS, cache_dir=cfg.paths.cache_dir,
    )
    labels = select_ar_events(events, target.noaa, t0 - timedelta(hours=24),
                              t1 + timedelta(hours=24))
    log.info("labels: %d GOES events attributed to NOAA %d", len(labels), target.noaa)

    meta = {
        "window": window_name,
        "noaa": target.noaa,
        "harp": harp,
        "start": t0.isoformat(),
        "end": t1.isoformat(),
        "channels": channels if not skip_aia else [],
        "hmi_segments": cfg.data.hmi_segments,
        "label_catalog_min_class": LABEL_CATALOG_MIN_CLASS,
        "stats": {key: robust_stats(arr) for key, arr in stacks.items()},
        "config_hash": cfg.short_hash(),
        "package_version": __version__,
        "created_utc": pd.Timestamp.utcnow().isoformat(),
    }
    sample_dir = Path(cfg.paths.cache_dir) / "samples" / sample_dir_name(harp, window_name)
    return write_sample(sample_dir, stacks, times, pd.DataFrame(qa_rows), labels, meta)
