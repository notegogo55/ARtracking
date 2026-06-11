"""YOLO dataset builder: bounded rebinned full-disk magnetograms + HARP boxes.

Data discipline: only the configured study windows are downloaded, at coarse
cadence (default 6 h) and JSOC-side rebinned 4096->1024 (verified live), so the
whole detection dataset stays at a few hundred MB. Splits are window-blocked.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from solarflare.config import Config
from solarflare.detect.bootstrap import boxes_to_pixels, fetch_harp_boxes

log = logging.getLogger(__name__)

CLASS_NAME = "active_region"


def magnetogram_to_png_array(data: np.ndarray, clip_gauss: float = 300.0) -> np.ndarray:
    """Float magnetogram -> uint8 image, FITS bottom-left origin flipped to PNG top-left.

    Linear map [-clip, +clip] -> [0, 255]; NaN (off-disk) -> 127 (neutral, 0 G).
    """
    scaled = (np.clip(data, -clip_gauss, clip_gauss) + clip_gauss) / (2 * clip_gauss)
    out = np.where(np.isfinite(data), scaled * 255.0, 127.0).astype(np.uint8)
    return np.flipud(out)


def yolo_label_lines(
    px_boxes: pd.DataFrame, width: int, height: int, min_box_px: float
) -> list[str]:
    """Pixel boxes (FITS orientation) -> YOLO txt lines (class 0, normalized cxcywh).

    The vertical flip applied to the PNG is mirrored here: cy_png = 1 - cy_fits.
    Boxes are clipped to the frame; boxes smaller than min_box_px are dropped.
    """
    lines = []
    for _, b in px_boxes.iterrows():
        x0 = max(float(b["x_min"]), -0.5)
        x1 = min(float(b["x_max"]), width - 0.5)
        y0 = max(float(b["y_min"]), -0.5)
        y1 = min(float(b["y_max"]), height - 0.5)
        if x1 - x0 < min_box_px or y1 - y0 < min_box_px:
            continue
        cx = (0.5 * (x0 + x1) + 0.5) / width
        cy = 1.0 - (0.5 * (y0 + y1) + 0.5) / height
        w = (x1 - x0) / width
        h = (y1 - y0) / height
        lines.append(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    return lines


def fetch_fulldisk_frames(cfg: Config, window, email: str, out_dir: Path) -> list[Path]:
    """Bounded rebinned full-disk export for one study window (reused if present)."""
    from solarflare.data.jsoc_fetch import _existing_fits, tai_str

    out_dir.mkdir(parents=True, exist_ok=True)
    existing = _existing_fits(out_dir)
    if existing:
        log.info("reusing %d full-disk frames in %s", len(existing), out_dir)
        return existing

    import drms

    ds = (f"{cfg.detect.fulldisk_series}"
          f"[{tai_str(window.start)}-{tai_str(window.end)}"
          f"@{cfg.detect.bootstrap_cadence_hours}h]"
          f"{{{cfg.detect.fulldisk_segment}}}")
    log.info("JSOC full-disk export (rebin %.2f): %s", cfg.detect.rebin_scale, ds)
    request = drms.Client().export(
        ds, method="url", protocol="fits", email=email,
        process={"rebin": {"method": "boxcar", "scale": cfg.detect.rebin_scale}},
    )
    request.wait()
    if not request.has_succeeded():
        raise RuntimeError(f"full-disk export failed for {ds!r}")
    request.download(str(out_dir))
    files = _existing_fits(out_dir)
    if not files:
        raise RuntimeError(f"full-disk export {ds!r} produced no files")
    return files


def build_yolo_dataset(cfg: Config, email: str, out_root: str | Path) -> pd.DataFrame:
    """Build images/ + labels/ + dataset.yaml for all configured splits.

    Returns a per-(split, window) summary table. Quiet windows contribute
    background images (empty label files), which YOLO uses as negatives.
    """
    import sunpy.map
    from PIL import Image

    out_root = Path(out_root)
    splits = {
        "train": cfg.detect.train_windows,
        "val": cfg.detect.val_windows,
        "test": cfg.detect.test_windows,
    }
    windows_by_name = {w.name: w for w in cfg.study.windows}
    raw_root = Path(cfg.paths.data_root) / "raw" / "fulldisk"
    rows = []
    for split, window_names in splits.items():
        img_dir = out_root / "images" / split
        lbl_dir = out_root / "labels" / split
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)
        for name in window_names:
            window = windows_by_name[name]
            frames = fetch_fulldisk_frames(cfg, window, email, raw_root / name)
            boxes = fetch_harp_boxes(
                window.start, window.end, cfg.detect.bootstrap_cadence_hours,
                cfg.paths.cache_dir,
            )
            n_boxes = 0
            tol = pd.Timedelta(hours=cfg.detect.bootstrap_cadence_hours / 2)
            for fits_path in frames:
                smap = sunpy.map.Map(str(fits_path))
                t = pd.Timestamp(smap.date.datetime)
                if len(boxes):
                    near = boxes[(pd.to_datetime(boxes["time"]) - t).abs() <= tol]
                else:
                    near = boxes
                px = boxes_to_pixels(near, smap)
                arcsec_per_px = abs(float(smap.scale[0].value))
                min_box_px = cfg.detect.min_box_arcsec / arcsec_per_px
                lines = yolo_label_lines(px, smap.data.shape[1], smap.data.shape[0],
                                         min_box_px)
                stem = f"{name}_{t:%Y%m%dT%H%M%S}"
                png = magnetogram_to_png_array(np.asarray(smap.data, dtype=np.float32),
                                               cfg.detect.image_clip_gauss)
                Image.fromarray(png, mode="L").save(img_dir / f"{stem}.png")
                (lbl_dir / f"{stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""),
                                                     encoding="utf-8")
                n_boxes += len(lines)
            rows.append({"split": split, "window": name,
                         "n_images": len(frames), "n_boxes": n_boxes})
            log.info("dataset: %s/%s -> %d images, %d boxes", split, name,
                     len(frames), n_boxes)

    yaml_text = (
        f"path: {out_root.resolve().as_posix()}\n"
        "train: images/train\nval: images/val\ntest: images/test\n"
        f"names:\n  0: {CLASS_NAME}\n"
    )
    (out_root / "dataset.yaml").write_text(yaml_text, encoding="utf-8")
    summary = pd.DataFrame(rows)
    summary.to_csv(out_root / "dataset_summary.csv", index=False)
    return summary


def load_yolo_labels(dataset_root: str | Path, split: str) -> pd.DataFrame:
    """Read back YOLO txt labels as pixel boxes [image, x_min, x_max, y_min, y_max]."""
    from PIL import Image

    dataset_root = Path(dataset_root)
    rows = []
    for img_path in sorted((dataset_root / "images" / split).glob("*.png")):
        with Image.open(img_path) as im:
            width, height = im.size
        lbl = dataset_root / "labels" / split / f"{img_path.stem}.txt"
        if not lbl.exists():
            continue
        for line in lbl.read_text().splitlines():
            parts = line.split()
            if len(parts) != 5:
                continue
            _, cx, cy, w, h = map(float, parts)
            rows.append({
                "image": img_path.name,
                "x_min": (cx - w / 2) * width, "x_max": (cx + w / 2) * width,
                "y_min": (cy - h / 2) * height, "y_max": (cy + h / 2) * height,
            })
    return pd.DataFrame(rows)
