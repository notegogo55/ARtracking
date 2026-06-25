"""Full-disk segmentation: threshold the whole solar disk, reproject mask to HARP CEA.

Operational flow (more realistic than HARP-patch segmenters):
  1. Load a rebinned full-disk HMI magnetogram (data/raw/fulldisk/<window>/,
     pre-fetched by `solarflare fetch-fulldisk`).
  2. Threshold |B_los| > cfg.bfield_threshold_gauss on the entire disk.
  3. Reproject the binary mask from helioprojective (TX/TY arcsec) onto the HARP
     CEA WCS grid via nearest-neighbour resampling (order=0, reproject_interp).
  4. Write ar_masks.npy — same (T, H, W) uint8 format as all other segmenters;
     features/extract.py is unchanged.

Prerequisite: `solarflare fetch-fulldisk --window <name> --email <email>`
"""

from __future__ import annotations

import logging
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ts_from_sharp_name(path: Path) -> pd.Timestamp | None:
    """Parse T_REC from a SHARP FITS filename (avoids a full header read).

    Pattern: hmi.sharp_cea_720s.<HARP>.<YYYYMMDD>_<HHMMSS>_TAI.<seg>.fits
    """
    m = re.search(r"\.(\d{8}_\d{6})_TAI\.", path.name)
    if m:
        return pd.Timestamp(datetime.strptime(m.group(1), "%Y%m%d_%H%M%S"))
    return None


def _build_fulldisk_index(fulldisk_dir: Path) -> list[tuple[pd.Timestamp, Path]]:
    """Sorted (timestamp, path) pairs for every FITS file in fulldisk_dir."""
    from solarflare.data.preprocess import read_date_obs

    fits_files = sorted(fulldisk_dir.glob("*.fits"))
    if not fits_files:
        raise RuntimeError(
            f"No full-disk frames found in {fulldisk_dir}.\n"
            "Run:  solarflare fetch-fulldisk --window <window-name> --email <email>"
        )
    index = []
    for f in fits_files:
        try:
            index.append((read_date_obs(f), f))
        except Exception as exc:  # noqa: BLE001
            log.warning("cannot read timestamp from %s (%s); skipping", f.name, exc)
    if not index:
        raise RuntimeError(f"All FITS in {fulldisk_dir} failed timestamp read.")
    index.sort(key=lambda x: x[0])
    return index


def _nearest(index: list[tuple[pd.Timestamp, Path]], t: pd.Timestamp) -> tuple[pd.Timestamp, Path]:
    """Entry in index whose timestamp is nearest to t."""
    t_ns = t.value
    best_ts, best_path = index[0]
    best_dt = abs(best_ts.value - t_ns)
    for ts, path in index[1:]:
        dt = abs(ts.value - t_ns)
        if dt < best_dt:
            best_dt, best_ts, best_path = dt, ts, path
    return best_ts, best_path


# ---------------------------------------------------------------------------
# Reprojection
# ---------------------------------------------------------------------------


def _reproject_mask(
    full_mask: np.ndarray,
    fd_map,  # sunpy Map — helioprojective WCS + observer position
    harp_map,  # sunpy Map — HARP CEA WCS
    target_shape: tuple[int, int],
) -> np.ndarray:
    """Nearest-neighbour reproject a binary full-disk mask onto the HARP CEA grid."""
    import sunpy.map
    from reproject import reproject_interp

    mask_map = sunpy.map.Map(full_mask.astype(np.float32), fd_map.meta)
    reprojected, _ = reproject_interp(
        (mask_map.data, mask_map.wcs),
        harp_map.wcs,
        shape_out=target_shape,
        order=0,  # nearest-neighbour — preserves binary values
    )
    return (reprojected > 0.5).astype(np.uint8)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def segment_sample_fulldisk(sample, cfg) -> tuple[Path, pd.DataFrame]:
    """Threshold full-disk HMI and reproject the mask onto the HARP CEA grid.

    Reads full-disk frames from  data/raw/fulldisk/<window>/  and SHARP magnetogram
    FITS from  data/raw/<window>/sharp_<harp>/  (both already present on disk).
    Output ar_masks.npy is (T, H, W) uint8 — same contract as all other segmenters.
    """
    import sunpy.map

    from solarflare.data.preprocess import match_nearest, read_date_obs
    from solarflare.detect.segment import active_mask

    window_name = sample.meta["window"]
    harp = sample.meta["harp"]
    # sample_dir = <data_root>/cache/samples/<name>;  parents[2] == data_root
    data_root = Path(sample.sample_dir).parents[2]
    fulldisk_dir = data_root / "raw" / "fulldisk" / window_name
    sharp_dir = data_root / "raw" / window_name / f"sharp_{harp}"

    # --- full-disk index --------------------------------------------------------
    fd_index = _build_fulldisk_index(fulldisk_dir)
    fd_cache: dict[Path, object] = {}  # lazy sunpy Map cache (only ~32 files)

    # --- SHARP magnetogram files for per-frame WCS ------------------------------
    sharp_mag = sorted(f for f in sharp_dir.glob("*.fits") if "magnetogram" in f.name.lower())
    if not sharp_mag:
        raise RuntimeError(f"No SHARP magnetogram FITS in {sharp_dir}")

    sharp_times = pd.Series([_ts_from_sharp_name(f) or read_date_obs(f) for f in sharp_mag])
    frame_times = sample.times["time_utc"].reset_index(drop=True)
    # 3600 s tolerance matches the hourly cadence of the SHARP subsample
    sharp_matches = match_nearest(frame_times, sharp_times, tolerance_seconds=3600)

    target_shape: tuple[int, int] = tuple(sample.arrays["hmi_magnetogram"].shape[1:])
    n_frames = sample.n_frames
    masks = np.zeros((n_frames, *target_shape), dtype=np.uint8)
    rows: list[dict] = []

    for i, t in enumerate(frame_times):
        t_ts = pd.Timestamp(t)

        # Nearest full-disk frame
        fd_ts, fd_path = _nearest(fd_index, t_ts)
        dt_h = abs((fd_ts - t_ts).total_seconds()) / 3600
        if dt_h > 3:
            log.warning("frame %d: nearest full-disk is %.1f h away", i, dt_h)
        if fd_path not in fd_cache:
            fd_cache[fd_path] = sunpy.map.Map(str(fd_path))
        fd_map = fd_cache[fd_path]

        # SHARP FITS for target WCS (closest match to this frame's timestamp)
        si = sharp_matches[i]
        harp_fits = sharp_mag[si] if si >= 0 else sharp_mag[min(i, len(sharp_mag) - 1)]
        harp_map = sunpy.map.Map(str(harp_fits))

        # Threshold on full-disk magnetogram
        fd_blos = np.asarray(fd_map.data, dtype=np.float32)
        full_mask = active_mask(
            fd_blos,
            cfg.bfield_threshold_gauss,
            cfg.min_region_pixels,
            cfg.morph_radius_px,
        )

        # Reproject to HARP CEA grid
        cea_mask = _reproject_mask(full_mask, fd_map, harp_map, target_shape)
        masks[i] = cea_mask
        rows.append(
            {
                "frame_idx": i,
                "spot_pixels": 0,  # hmi.m_720s has no continuum channel
                "active_pixels": int(cea_mask.sum()),
                "ar_pixels": int(cea_mask.sum()),
            }
        )

    areas = pd.DataFrame(rows)
    masks_path = Path(sample.sample_dir) / "ar_masks.npy"
    np.save(masks_path, masks)
    areas.to_csv(Path(sample.sample_dir) / "ar_mask_areas.csv", index=False)
    log.info(
        "full-disk seg: %d frames, HARP CEA %s, median AR %d px",
        n_frames,
        "×".join(str(s) for s in target_shape),
        int(areas["ar_pixels"].median()),
    )
    return masks_path, areas
