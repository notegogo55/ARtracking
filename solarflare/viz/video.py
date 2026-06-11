"""Frame-by-frame videos of a cached AR sample (detection/segmentation view).

Renders continuum + magnetogram + one AIA channel side by side with the AR
mask contour and GOES flare annotations, then encodes MP4 via OpenCV (mp4v —
no ffmpeg dependency). The same compositor feeds the Streamlit app.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


def _scale_gray(frame: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    out = (np.clip(frame, vmin, vmax) - vmin) / max(vmax - vmin, 1e-6)
    return np.nan_to_num(out, nan=0.0)


def _apply_colormap(scaled: np.ndarray, cmap_name: str) -> np.ndarray:
    import matplotlib

    rgba = matplotlib.colormaps[cmap_name](scaled)
    return (rgba[..., :3] * 255).astype(np.uint8)


def _draw_mask_outline(rgb: np.ndarray, mask: np.ndarray,
                       color: tuple[int, int, int]) -> np.ndarray:
    import cv2

    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    return cv2.drawContours(rgb.copy(), contours, -1, color, 1)


class SampleScaling:
    """Per-channel display ranges computed ONCE over the whole stack, so the
    video does not flicker as per-frame percentiles move."""

    def __init__(self, sample, channel_key: str) -> None:
        cont = np.asarray(sample.arrays["hmi_continuum"][::10])
        mag = np.asarray(sample.arrays["hmi_magnetogram"][::10])
        aia = np.asarray(sample.arrays[channel_key][::10])
        self.cont = (float(np.nanpercentile(cont, 1)), float(np.nanpercentile(cont, 99.9)))
        vmax = float(np.nanpercentile(np.abs(mag), 99.5)) or 1.0
        self.mag = (-vmax, vmax)
        finite = aia[np.isfinite(aia)]
        if finite.size:
            lo, hi = np.percentile(finite, [1, 99.7])
        else:
            lo, hi = 0.0, 1.0
        self.aia = (float(lo), float(hi))


def compose_frame(
    sample, masks: np.ndarray, frame_idx: int, channel_key: str,
    scaling: SampleScaling, flare_note: str = "",
) -> np.ndarray:
    """One (H, 3W+pad, 3) RGB frame: continuum | magnetogram | AIA, mask outlined."""
    import cv2

    mask = masks[frame_idx].astype(bool)
    cont = _apply_colormap(_scale_gray(np.asarray(sample.arrays["hmi_continuum"][frame_idx]),
                                       *scaling.cont), "afmhot")
    mag = _apply_colormap(_scale_gray(np.asarray(sample.arrays["hmi_magnetogram"][frame_idx]),
                                      *scaling.mag), "gray")
    aia_raw = np.asarray(sample.arrays[channel_key][frame_idx], dtype=np.float32)
    aia = _apply_colormap(np.sqrt(_scale_gray(aia_raw, *scaling.aia)), "inferno")

    cont = _draw_mask_outline(cont, mask, (0, 255, 255))
    mag = _draw_mask_outline(mag, mask, (255, 60, 60))
    aia = _draw_mask_outline(aia, mask, (0, 255, 120))

    pad = np.full((cont.shape[0], 4, 3), 30, dtype=np.uint8)
    row = np.concatenate([cont, pad, mag, pad, aia], axis=1)
    row = np.flipud(row)  # FITS bottom-left origin -> video top-left

    time_utc = pd.Timestamp(sample.times["time_utc"].iloc[frame_idx])
    header = np.zeros((28, row.shape[1], 3), dtype=np.uint8)
    label = (f"NOAA {sample.meta.get('noaa')} / HARP {sample.meta.get('harp')}   "
             f"{time_utc:%Y-%m-%d %H:%M} UT   continuum | B_los | "
             f"{channel_key.replace('aia_', 'AIA ')} A")
    cv2.putText(header, label, (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (255, 255, 255), 1, cv2.LINE_AA)
    if flare_note:
        cv2.putText(header, flare_note, (row.shape[1] - 170, 19),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 80, 255), 2, cv2.LINE_AA)
    return np.concatenate([header, row], axis=0)


def flare_annotations(sample, window_minutes: float = 45.0) -> dict[int, str]:
    """frame_idx -> 'X2.2' for frames within +/-window of a >=M flare peak."""
    if sample.labels.empty:
        return {}
    times = pd.to_datetime(sample.times["time_utc"])
    notes: dict[int, str] = {}
    for _, ev in sample.labels.iterrows():
        cls = str(ev["goes_class"])
        if not cls or cls[0].upper() not in ("M", "X"):
            continue
        peak = pd.Timestamp(ev["peak_time"])
        near = (times - peak).abs() <= pd.Timedelta(minutes=window_minutes)
        for idx in np.where(near.to_numpy())[0]:
            notes[int(idx)] = f"{cls} flare!"
    return notes


def render_sample_video(
    sample, masks: np.ndarray, out_path: str | Path,
    channel: int = 171, start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None, fps: int = 12,
    max_width: int = 1920,
) -> Path:
    """Encode the sample as MP4 over [start, end] (defaults: full range)."""
    import cv2

    from solarflare.data.cache import channel_key as ck

    key = ck(channel)
    if key not in sample.arrays:
        raise KeyError(f"channel {channel} not in sample (have {sorted(sample.arrays)})")
    times = pd.to_datetime(sample.times["time_utc"])
    sel = np.ones(len(times), dtype=bool)
    if start is not None:
        sel &= (times >= pd.Timestamp(start)).to_numpy()
    if end is not None:
        sel &= (times <= pd.Timestamp(end)).to_numpy()
    frame_ids = np.where(sel)[0]
    if len(frame_ids) == 0:
        raise ValueError("no frames in the requested time range")

    scaling = SampleScaling(sample, key)
    notes = flare_annotations(sample)
    first = compose_frame(sample, masks, int(frame_ids[0]), key, scaling)
    scale = min(1.0, max_width / first.shape[1])
    size = (int(first.shape[1] * scale), int(first.shape[0] * scale))

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    if not writer.isOpened():
        raise RuntimeError("OpenCV VideoWriter failed to open (mp4v)")
    try:
        for n, idx in enumerate(frame_ids):
            rgb = compose_frame(sample, masks, int(idx), key, scaling,
                                notes.get(int(idx), ""))
            if scale < 1.0:
                rgb = cv2.resize(rgb, size, interpolation=cv2.INTER_AREA)
            writer.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            if (n + 1) % 100 == 0:
                log.info("rendered %d/%d frames", n + 1, len(frame_ids))
    finally:
        writer.release()
    log.info("video: %s (%d frames @ %d fps, %s)", out_path, len(frame_ids), fps, size)
    return out_path
