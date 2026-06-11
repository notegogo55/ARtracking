"""QA overlay: HMI magnetogram + AIA channel side by side on the shared CEA grid.

Because every channel in a sample cache lives on the same reprojected grid,
visual alignment of magnetogram contours over the AIA image is a direct check
that co-registration worked (Gate G1's "visually lines up").
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from solarflare.data.cache import SampleData, channel_key

log = logging.getLogger(__name__)


def pick_overlay_frame(sample: SampleData) -> int:
    """Frame nearest the strongest labeled flare peak, else the middle frame."""
    times = pd.to_datetime(sample.times["time_utc"])
    if not sample.labels.empty:
        from solarflare.data.goes_events import goes_class_to_flux

        labels = sample.labels.copy()
        labels["flux"] = labels["goes_class"].map(goes_class_to_flux)
        peak = pd.Timestamp(labels.sort_values("flux").iloc[-1]["peak_time"])
        if times.iloc[0] <= peak <= times.iloc[-1]:
            return int((times - peak).abs().idxmin())
    return len(times) // 2


def build_overlay(
    sample: SampleData,
    channel: int = 171,
    frame_idx: int | None = None,
    out_path: str | Path | None = None,
) -> Path:
    """Render the QA overlay PNG; returns the output path."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if frame_idx is None:
        frame_idx = pick_overlay_frame(sample)
    mag = np.asarray(sample.arrays["hmi_magnetogram"][frame_idx], dtype=np.float32)
    key = channel_key(channel)
    if key not in sample.arrays:
        raise KeyError(f"channel {channel} not in sample (have {sorted(sample.arrays)})")
    aia = np.asarray(sample.arrays[key][frame_idx], dtype=np.float32)
    time_utc = pd.Timestamp(sample.times["time_utc"].iloc[frame_idx])
    meta = sample.meta

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    vmax = float(np.nanpercentile(np.abs(mag), 99.5)) or 1.0
    im0 = axes[0].imshow(mag, cmap="gray", origin="lower", vmin=-vmax, vmax=vmax)
    axes[0].set_title(f"HMI magnetogram (CEA)  {time_utc:%Y-%m-%d %H:%M} UT")
    fig.colorbar(im0, ax=axes[0], shrink=0.8, label="B_los [G]")

    finite = aia[np.isfinite(aia)]
    if finite.size:
        v1, v99 = np.percentile(finite, [1, 99.5])
        shown = np.sqrt(np.clip(aia - v1, 0, None))
        vmax_aia = np.sqrt(max(v99 - v1, 1e-6))
    else:
        shown, vmax_aia = aia, 1.0
    im1 = axes[1].imshow(shown, cmap="inferno", origin="lower", vmin=0, vmax=vmax_aia)
    contour_level = max(300.0, 0.3 * vmax)
    axes[1].contour(mag, levels=[-contour_level, contour_level],
                    colors=["cyan", "lime"], linewidths=0.8)
    axes[1].set_title(f"AIA {channel} A (DN/s, reprojected) + B_los contours")
    fig.colorbar(im1, ax=axes[1], shrink=0.8, label="sqrt(DN/s)")

    flare_note = ""
    if not sample.labels.empty:
        biggest = sample.labels.iloc[-1]
        flare_note = f" | GOES events: {len(sample.labels)} (latest {biggest['goes_class']})"
    fig.suptitle(
        f"NOAA {meta.get('noaa')} / HARP {meta.get('harp')} — frame {frame_idx}{flare_note}"
    )

    if out_path is None:
        out_path = sample.sample_dir / f"qa_overlay_{channel:04d}_f{frame_idx:04d}.png"
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    log.info("QA overlay written to %s", out_path)
    return out_path
