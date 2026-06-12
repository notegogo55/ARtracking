"""JSOC-style "Tracked AR (HARP)" full-disk maps: PNG frames + MP4.

Mimics the classic http://jsoc.stanford.edu HARP tracking image: gray solar
disk on white, strong-|B| pixels as white speckles, each HARP's strong-field
pixels filled with a per-HARP color, black numbered bounding boxes, blue NOAA
crosses, and NOAA/HARP legends. Built entirely from the project's cached
full-disk magnetograms and the SHARP keyword bootstrap (no JSOC round-trip).
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# saturated, well-separated palette (RGB); chosen to echo the JSOC map look
PALETTE = [
    (139, 28, 28),  # dark red / maroon
    (218, 64, 212),  # magenta
    (0, 121, 107),  # teal
    (87, 144, 56),  # green
    (70, 130, 180),  # steel blue
    (230, 126, 34),  # orange
    (93, 58, 14),  # dark brown
    (227, 38, 54),  # red
    (94, 53, 177),  # purple
    (30, 144, 255),  # bright blue
    (130, 119, 23),  # olive
    (216, 27, 96),  # pink
]

DISK_GRAY = (168, 168, 168)
NOAA_BLUE = (0, 0, 230)


def harp_color(harpnum: int) -> tuple[int, int, int]:
    """Deterministic RGB color for a HARP (stable across frames and runs)."""
    return PALETTE[int(harpnum) % len(PALETTE)]


def render_harp_disk(
    fulldisk_map,
    boxes_px: pd.DataFrame,
    size: int = 880,
    field_threshold: float = 100.0,
    spot_threshold: float = 700.0,
) -> np.ndarray:
    """Gray disk with white strong-field speckles and per-HARP colored masks.

    Pixels with |B| above field_threshold are speckled white; inside a HARP
    box they take the HARP color instead, with |B| above spot_threshold kept
    white (sunspot cores), matching the JSOC map's look.
    """
    import cv2

    data = np.asarray(fulldisk_map.data, dtype=np.float32)
    h, w = data.shape
    yy, xx = np.mgrid[0:h, 0:w]
    cx = float(fulldisk_map.wcs.wcs.crpix[0]) - 1
    cy = float(fulldisk_map.wcs.wcs.crpix[1]) - 1
    r_pix = float(fulldisk_map.rsun_obs.to_value("arcsec") / abs(fulldisk_map.scale[0].value))
    on_disk = np.hypot(xx - cx, yy - cy) <= r_pix

    babs = np.abs(np.nan_to_num(data, nan=0.0))
    strong = (babs >= field_threshold) & on_disk

    rgb = np.full((h, w, 3), 255, dtype=np.uint8)  # white background
    rgb[on_disk] = DISK_GRAY
    rgb[strong] = (255, 255, 255)  # speckles

    for _, b in boxes_px.iterrows():
        x0 = max(int(b["x_min"]), 0)
        x1 = min(int(b["x_max"]), w - 1)
        y0 = max(int(b["y_min"]), 0)
        y1 = min(int(b["y_max"]), h - 1)
        if x1 <= x0 or y1 <= y0:
            continue
        box = np.zeros_like(strong)
        box[y0 : y1 + 1, x0 : x1 + 1] = True
        member = strong & box
        rgb[member & (babs < spot_threshold)] = harp_color(int(b["harpnum"]))
        # spot cores stay white

    rgb = np.flipud(rgb)  # FITS bottom-left -> image top-left
    return cv2.resize(rgb, (size, size), interpolation=cv2.INTER_NEAREST)


def compose_harpmap_frame(
    fulldisk_map,
    boxes_px: pd.DataFrame,
    time_utc: pd.Timestamp,
    size: int = 880,
    legend_width: int = 240,
    field_threshold: float = 100.0,
    spot_threshold: float = 700.0,
) -> np.ndarray:
    """One JSOC-style frame: disk + boxes + NOAA crosses + side legend."""
    import cv2

    disk = render_harp_disk(fulldisk_map, boxes_px, size, field_threshold, spot_threshold)
    h_native = fulldisk_map.data.shape[0]
    scale = size / h_native

    entries = []  # (harpnum, noaa, color) for the legend
    for _, b in boxes_px.iterrows():
        harp = int(b["harpnum"])
        color = harp_color(harp)
        x0, x1 = int(b["x_min"] * scale), int(b["x_max"] * scale)
        y0 = int(size - b["y_max"] * scale)  # vertical flip, as in dashboard
        y1 = int(size - b["y_min"] * scale)
        cv2.rectangle(disk, (x0, y0), (x1, y1), (0, 0, 0), 1)
        cv2.putText(
            disk,
            str(harp),
            (max(x0, x1 - 52), max(12, y0 - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )
        noaa = int(b.get("noaa_ar", 0) or 0)
        cxp, cyp = (x0 + x1) // 2, (y0 + y1) // 2
        cv2.drawMarker(disk, (cxp, cyp), NOAA_BLUE, cv2.MARKER_CROSS, 12, 2)
        if noaa:
            cv2.putText(
                disk,
                str(noaa),
                (cxp - 24, min(size - 6, y1 + 16)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                NOAA_BLUE,
                1,
                cv2.LINE_AA,
            )
        entries.append((harp, noaa, color))

    # header (drawn over the white margin of the disk panel)
    cv2.putText(
        disk,
        "SDO/HMI Tracked AR (HARP)",
        (10, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 0, 0),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        disk,
        f"{time_utc:%Y/%m/%d}",
        (10, 44),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 0, 0),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        disk,
        f"{time_utc:%H:%M}",
        (10, 64),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 0, 0),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        disk,
        f"H = {len(entries)}",
        (10, size - 14),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 0, 0),
        1,
        cv2.LINE_AA,
    )

    # right-hand legend: NOAA ARs (blue) | HARPs with color chips
    legend = np.full((size, legend_width, 3), 255, dtype=np.uint8)
    cv2.putText(
        legend, "NOAA ARs", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.5, NOAA_BLUE, 1, cv2.LINE_AA
    )
    cv2.putText(
        legend, "HARPs", (130, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA
    )
    y = 46
    for harp, noaa, color in sorted(entries, key=lambda e: -e[0]):
        if y > size - 10:
            break
        if noaa:
            cv2.putText(
                legend,
                str(noaa),
                (10, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                NOAA_BLUE,
                1,
                cv2.LINE_AA,
            )
        cv2.rectangle(legend, (130, y - 11), (144, y + 1), color, -1)
        cv2.putText(
            legend, str(harp), (150, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA
        )
        y += 22

    return np.concatenate([disk, legend], axis=1)


def render_harpmap(
    cfg,
    window_name: str,
    out_dir: str | Path,
    fps: int = 4,
    size: int = 880,
) -> tuple[Path, int]:
    """JSOC-style frames + MP4 for one study window. Returns (mp4 path, n)."""
    import cv2
    import sunpy.map

    from solarflare.detect.bootstrap import boxes_to_pixels, fetch_harp_boxes
    from solarflare.viz.video import Mp4Writer

    window = next(w for w in cfg.study.windows if w.name == window_name)
    fulldisk_dir = Path(cfg.paths.data_root) / "raw" / "fulldisk" / window_name
    fits_files = sorted(fulldisk_dir.glob("*.fits"))
    if not fits_files:
        raise FileNotFoundError(
            f"no full-disk frames under {fulldisk_dir} - run build-detect-dataset"
        )
    boxes = fetch_harp_boxes(
        window.start, window.end, cfg.detect.bootstrap_cadence_hours, cfg.paths.cache_dir
    )

    out_dir = Path(out_dir)
    frames_dir = out_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    mp4_path = out_dir / f"harpmap_{window_name}.mp4"
    tol = pd.Timedelta(hours=cfg.detect.bootstrap_cadence_hours / 2)
    writer = None
    n = 0
    for fits_path in fits_files:
        smap = sunpy.map.Map(str(fits_path))
        t = pd.Timestamp(smap.date.datetime)
        near = boxes[(pd.to_datetime(boxes["time"]) - t).abs() <= tol]
        px = boxes_to_pixels(near, smap)
        frame = compose_harpmap_frame(smap, px, t, size=size)
        cv2.imwrite(
            str(frames_dir / f"{window_name}_{t:%Y%m%dT%H%M}.png"),
            cv2.cvtColor(frame, cv2.COLOR_RGB2BGR),
        )
        if writer is None:
            writer = Mp4Writer(mp4_path, (frame.shape[1], frame.shape[0]), fps)
        writer.write_rgb(frame)
        n += 1
    if writer is not None:
        writer.close()
    log.info("harpmap: %s (%d frames) + %s", mp4_path, n, frames_dir)
    return mp4_path, n
