"""Segmentation baseline: intensity threshold + morphology (no training required).

Two masks per frame on the SHARP CEA patch:
- sunspot mask: continuum below a fraction of the quiet-Sun level (the patch
  median is a robust quiet-Sun proxy because spots cover a small area fraction);
- active mask:  |B_los| above a Gauss threshold.
The AR mask (their union) is the boundary Phase C uses for max-intensity
feature extraction. This is the "threshold" `Segmenter`; the trained U-Net and
the surya/sam2 stubs sit behind the same registry (solarflare.detect.segmenter),
so `segment_sample_auto` dispatches on cfg.model.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


def _cleanup(mask: np.ndarray, min_pixels: int, radius: int) -> np.ndarray:
    from skimage import morphology

    footprint = morphology.disk(radius)
    mask = morphology.closing(mask, footprint)
    # max_size removes objects <= value, so this keeps regions of >= min_pixels
    mask = morphology.remove_small_objects(mask, max_size=min_pixels - 1)
    mask = morphology.opening(mask, footprint)
    return mask.astype(bool)


def sunspot_mask(
    continuum: np.ndarray,
    spot_threshold: float = 0.85,
    min_pixels: int = 64,
    morph_radius: int = 2,
) -> np.ndarray:
    """Boolean sunspot mask from a continuum frame (threshold vs patch median)."""
    finite = np.isfinite(continuum)
    if not finite.any():
        return np.zeros(continuum.shape, dtype=bool)
    quiet = np.median(continuum[finite])
    if not np.isfinite(quiet) or quiet <= 0:
        return np.zeros(continuum.shape, dtype=bool)
    mask = np.nan_to_num(continuum, nan=np.inf) < spot_threshold * quiet
    return _cleanup(mask, min_pixels, morph_radius)


def active_mask(
    blos: np.ndarray,
    bfield_threshold: float = 100.0,
    min_pixels: int = 64,
    morph_radius: int = 2,
) -> np.ndarray:
    """Boolean active-pixel mask from a LOS magnetogram frame."""
    mask = np.nan_to_num(np.abs(blos), nan=0.0) > bfield_threshold
    return _cleanup(mask, min_pixels, morph_radius)


def ar_mask(continuum: np.ndarray, blos: np.ndarray, cfg) -> np.ndarray:
    """AR boundary mask = sunspots OR strong-field pixels (cfg: SegmentConfig)."""
    return sunspot_mask(
        continuum, cfg.spot_threshold, cfg.min_region_pixels, cfg.morph_radius_px
    ) | active_mask(blos, cfg.bfield_threshold_gauss, cfg.min_region_pixels, cfg.morph_radius_px)


def segment_sample(sample, cfg) -> tuple[Path, pd.DataFrame]:
    """Segment every frame of a cached sample; write masks + area table into it.

    Returns (masks.npy path, per-frame area DataFrame). Masks are uint8 (T,H,W).
    """
    continuum = sample.arrays["hmi_continuum"]
    blos = sample.arrays["hmi_magnetogram"]
    n_frames = sample.n_frames
    masks = np.zeros(blos.shape, dtype=np.uint8)
    rows = []
    for i in range(n_frames):
        spot = sunspot_mask(
            np.asarray(continuum[i]), cfg.spot_threshold, cfg.min_region_pixels, cfg.morph_radius_px
        )
        active = active_mask(
            np.asarray(blos[i]),
            cfg.bfield_threshold_gauss,
            cfg.min_region_pixels,
            cfg.morph_radius_px,
        )
        masks[i] = (spot | active).astype(np.uint8)
        rows.append(
            {
                "frame_idx": i,
                "spot_pixels": int(spot.sum()),
                "active_pixels": int(active.sum()),
                "ar_pixels": int(masks[i].sum()),
            }
        )
    areas = pd.DataFrame(rows)
    masks_path = Path(sample.sample_dir) / "ar_masks.npy"
    np.save(masks_path, masks)
    areas.to_csv(Path(sample.sample_dir) / "ar_mask_areas.csv", index=False)
    log.info(
        "segmented %d frames -> %s (median AR area %d px)",
        n_frames,
        masks_path,
        int(areas["ar_pixels"].median()),
    )
    return masks_path, areas


def segment_sample_auto(sample, cfg) -> tuple[Path, pd.DataFrame]:
    """Segment via the `Segmenter` selected by cfg.model (registry; same outputs)."""
    from solarflare.detect.segmenter import get_segmenter

    return get_segmenter(cfg).segment_sample(sample)


def segmentation_qa_plot(
    sample, masks: np.ndarray, frame_idx: int, out_path: str | Path | None = None
) -> Path:
    """Continuum + magnetogram with the AR mask contour, for visual QA."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cont = np.asarray(sample.arrays["hmi_continuum"][frame_idx])
    blos = np.asarray(sample.arrays["hmi_magnetogram"][frame_idx])
    mask = masks[frame_idx].astype(bool)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    axes[0].imshow(cont, cmap="afmhot", origin="lower")
    axes[0].contour(mask, levels=[0.5], colors="cyan", linewidths=0.8)
    axes[0].set_title(f"continuum + AR mask (frame {frame_idx})")
    vmax = float(np.nanpercentile(np.abs(blos), 99.5)) or 1.0
    axes[1].imshow(blos, cmap="gray", origin="lower", vmin=-vmax, vmax=vmax)
    axes[1].contour(mask, levels=[0.5], colors="red", linewidths=0.8)
    axes[1].set_title("B_los + AR mask")
    if out_path is None:
        out_path = Path(sample.sample_dir) / f"segment_qa_f{frame_idx:04d}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return Path(out_path)
