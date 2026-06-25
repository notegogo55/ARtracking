"""Visualise full-disk HMI magnetogram with AR segmentation mask overlaid.

Usage:
    uv run python scripts/plot_fulldisk_segmented.py
    uv run python scripts/plot_fulldisk_segmented.py --fits data/raw/fulldisk/ar12673_sep2017/hmi.m_720s.20170906_120000_TAI.3.magnetogram.fits
    uv run python scripts/plot_fulldisk_segmented.py --all-windows
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np


def segment_fulldisk(blos: np.ndarray, threshold_gauss: float = 100,
                     min_region_pixels: int = 64, morph_radius_px: int = 2) -> np.ndarray:
    from solarflare.detect.segment import active_mask
    return active_mask(blos, threshold_gauss, min_region_pixels, morph_radius_px)


def make_disk_alpha(data: np.ndarray) -> np.ndarray:
    """True where pixels are on the solar disk (non-NaN, not fill-value ≈ ±1e9)."""
    on_disk = np.isfinite(data) & (np.abs(data) < 5000)
    return on_disk


def plot_fulldisk(fits_path: Path, out_path: Path | None = None,
                  threshold: float = 100, vclip: float = 500) -> Path:
    import sunpy.map

    print(f"Loading {fits_path.name} ...")
    fd_map = sunpy.map.Map(str(fits_path))
    blos = np.asarray(fd_map.data, dtype=np.float32)

    print("Segmenting ...")
    mask = segment_fulldisk(blos, threshold_gauss=threshold)
    disk = make_disk_alpha(blos)

    # ---- figure ----------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 7), facecolor="black")
    date_str = fits_path.name.split(".")[2].replace("_TAI", "").replace("_", " ")

    for ax in axes:
        ax.set_facecolor("black")
        ax.set_xticks([])
        ax.set_yticks([])

    # Left: magnetogram clipped at ±vclip G
    display = np.where(disk, np.clip(blos, -vclip, vclip), np.nan)
    im = axes[0].imshow(display, cmap="RdBu_r", vmin=-vclip, vmax=vclip, origin="lower")
    # AR mask contour
    axes[0].contour(mask.astype(float), levels=[0.5], colors=["lime"], linewidths=0.8)
    axes[0].set_title(f"HMI B_los  |  {date_str} TAI", color="white", fontsize=11)
    cb = fig.colorbar(im, ax=axes[0], fraction=0.046, pad=0.01)
    cb.set_label("B_los  [G]", color="white")
    cb.ax.yaxis.set_tick_params(color="white")
    plt.setp(cb.ax.yaxis.get_ticklabels(), color="white")

    # Right: mask overlay on grayscale magnetogram
    gray = np.where(disk, np.clip(blos, -vclip, vclip), np.nan)
    axes[1].imshow(gray, cmap="gray", vmin=-vclip, vmax=vclip, origin="lower")
    # Colour AR pixels
    rgba = np.zeros((*mask.shape, 4), dtype=np.float32)
    rgba[mask > 0] = [1.0, 0.9, 0.0, 0.55]   # yellow, semi-transparent
    axes[1].imshow(rgba, origin="lower")
    axes[1].contour(mask.astype(float), levels=[0.5], colors=["yellow"], linewidths=0.8)

    n_ar = int(mask.sum())
    n_regions = _count_regions(mask)
    patch = mpatches.Patch(color="yellow", alpha=0.55,
                           label=f"AR mask  ({n_ar:,} px, {n_regions} region(s))")
    axes[1].legend(handles=[patch], loc="lower right",
                   facecolor="#111", edgecolor="gray", labelcolor="white", fontsize=9)
    axes[1].set_title(f"|B_los| > {threshold:.0f} G  —  full-disk threshold", color="white", fontsize=11)

    fig.suptitle(f"Full-disk AR segmentation  ·  AR 12673  ·  {date_str} TAI",
                 color="white", fontsize=13, y=1.01)
    plt.tight_layout()

    if out_path is None:
        stem = fits_path.stem.replace("hmi.m_720s.", "").replace("_TAI.3.magnetogram", "")
        out_path = Path("outputs") / "segment" / f"fulldisk_seg_{stem}.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="black")
    plt.close(fig)
    print(f"Saved: {out_path}")
    return out_path


def _count_regions(mask: np.ndarray) -> int:
    from scipy.ndimage import label
    _, n = label(mask)
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fits", type=Path,
        default=Path("data/raw/fulldisk/ar12673_sep2017/hmi.m_720s.20170906_120000_TAI.3.magnetogram.fits"))
    ap.add_argument("--all-windows", action="store_true",
        help="Generate one representative frame per window")
    ap.add_argument("--threshold", type=float, default=100)
    ap.add_argument("--vclip", type=float, default=500)
    args = ap.parse_args()

    if args.all_windows:
        windows = {
            "ar11158_feb2011": "hmi.m_720s.20110215_000000_TAI.1.magnetogram.fits",
            "ar11429_mar2012": "hmi.m_720s.20120307_000000_TAI.1.magnetogram.fits",
            "ar12673_sep2017": "hmi.m_720s.20170906_120000_TAI.3.magnetogram.fits",
        }
        for win, fname in windows.items():
            p = Path("data/raw/fulldisk") / win / fname
            if p.exists():
                plot_fulldisk(p, threshold=args.threshold, vclip=args.vclip)
            else:
                print(f"[skip] {p} not found")
    else:
        plot_fulldisk(args.fits, threshold=args.threshold, vclip=args.vclip)


if __name__ == "__main__":
    main()
