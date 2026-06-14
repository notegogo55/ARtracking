"""Operational "Solar Region Summary" dashboard: frames + MP4 (NOAA SWPC style).

A cleaner, more operational alternative to the DeepFlareNet per-box view: each
frame pairs a grayscale HMI full-disk magnetogram (the classic SWPC/SolarMonitor
look) with tracked AR boxes color-coded by a four-level flare *risk*, and a
right-hand **region-summary table** listing every region's NOAA/HARP id, its
Stonyhurst location (e.g. N15W20), heliographic extent, and the model's
P(>=M, 24h) as a risk bar -- mirroring NOAA SWPC's daily Solar Region Summary.

The single calibrated >=M (24 h) probability per AR is mapped to a Low / Moderate
/ High / Very-High risk band (no fabricated per-class C/M/X numbers). The footer
always names the model + dataset so a clip is never mistaken for an operational
forecast. Probabilities come from the same pluggable fitted model and lookup as
the legacy dashboard (`solarflare.viz.dashboard.build_probability_lookup`).
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# Four-level >=M/24h risk palette (RGB; frames stay RGB end-to-end). Thresholds
# are ordered high->low so the first match wins.
RISK_BANDS: list[tuple[float, str, tuple[int, int, int]]] = [
    (0.50, "Very High", (231, 76, 60)),  # red
    (0.25, "High", (230, 126, 34)),  # orange
    (0.10, "Moderate", (241, 196, 15)),  # amber
    (0.00, "Low", (39, 174, 96)),  # green
]
NO_DATA_COLOR = (120, 124, 132)


def risk_level(p_m: float) -> tuple[str, tuple[int, int, int]]:
    """Map a >=M probability to a (label, RGB color) risk band."""
    for threshold, label, color in RISK_BANDS:
        if p_m >= threshold:
            return label, color
    return RISK_BANDS[-1][1], RISK_BANDS[-1][2]


def format_stonyhurst(lon: float | None, lat: float | None) -> str:
    """Stonyhurst (lon, lat) deg -> SWPC location string, e.g. N15W20 (W = +lon)."""
    if lon is None or lat is None or not (np.isfinite(lon) and np.isfinite(lat)):
        return "--"
    ns = "N" if lat >= 0 else "S"
    ew = "W" if lon >= 0 else "E"
    return f"{ns}{abs(lat):02.0f}{ew}{abs(lon):02.0f}"


def region_rows(boxes_px: pd.DataFrame, probs: dict[int, float]) -> list[dict]:
    """One summary row per AR box, sorted by descending risk (no-data last)."""
    rows = []
    for _, b in boxes_px.iterrows():
        harp = int(b["harpnum"])
        lon = b.get("lon_fwt")
        lat = b.get("lat_fwt")
        if lon is None or not np.isfinite(lon):
            lon = 0.5 * (float(b["lon_min"]) + float(b["lon_max"]))
        if lat is None or not np.isfinite(lat):
            lat = 0.5 * (float(b["lat_min"]) + float(b["lat_max"]))
        extent = (float(b["lon_max"]) - float(b["lon_min"])) * (
            float(b["lat_max"]) - float(b["lat_min"])
        )
        p = probs.get(harp)
        label, color = (risk_level(p)) if p is not None else ("No data", NO_DATA_COLOR)
        rows.append(
            {
                "harp": harp,
                "noaa": int(b.get("noaa_ar", 0) or 0),
                "loc": format_stonyhurst(lon, lat),
                "extent_deg2": extent,
                "p": p,
                "risk": label,
                "color": color,
            }
        )
    rows.sort(key=lambda r: -1.0 if r["p"] is None else r["p"], reverse=True)
    return rows


def render_magnetogram_disk(fulldisk_map, size: int = 820, clip_gauss: float = 120.0) -> np.ndarray:
    """Grayscale HMI full-disk magnetogram (signed B -> gray), off-disk near-black."""
    import cv2

    data = np.asarray(fulldisk_map.data, dtype=np.float32)
    h, w = data.shape
    yy, xx = np.mgrid[0:h, 0:w]
    cx = float(fulldisk_map.wcs.wcs.crpix[0]) - 1
    cy = float(fulldisk_map.wcs.wcs.crpix[1]) - 1
    r_pix = float(fulldisk_map.rsun_obs.to_value("arcsec") / abs(fulldisk_map.scale[0].value))
    on_disk = np.hypot(xx - cx, yy - cy) <= r_pix

    signed = np.clip(np.nan_to_num(data, nan=0.0) / clip_gauss, -1.0, 1.0)
    gray = np.where(on_disk, 0.30 + 0.60 * (0.5 + 0.5 * signed), 0.0)  # mid-gray disk
    rgb = (np.repeat(gray[..., None], 3, axis=2) * 255).astype(np.uint8)
    rgb[~on_disk] = (10, 12, 18)
    rgb = np.flipud(rgb)  # FITS bottom-left -> image top-left
    return cv2.resize(rgb, (size, size), interpolation=cv2.INTER_AREA)


def _draw_summary_panel(rows: list[dict], height: int, width: int) -> np.ndarray:
    """Right-hand region-summary table panel (dark theme)."""
    import cv2

    font = cv2.FONT_HERSHEY_SIMPLEX
    panel = np.full((height, width, 3), 22, dtype=np.uint8)
    cv2.putText(panel, "ACTIVE REGIONS", (14, 28), font, 0.6, (235, 235, 235), 2, cv2.LINE_AA)
    # column headers
    cols = {"noaa": 14, "harp": 104, "loc": 186, "bar": 262}
    y = 56
    for name, x in (("NOAA", cols["noaa"]), ("HARP", cols["harp"]), ("LOC", cols["loc"])):
        cv2.putText(panel, name, (x, y), font, 0.42, (150, 154, 160), 1, cv2.LINE_AA)
    cv2.putText(panel, "P(>=M, 24h)", (cols["bar"], y), font, 0.42, (150, 154, 160), 1, cv2.LINE_AA)
    cv2.line(panel, (10, y + 8), (width - 10, y + 8), (60, 60, 66), 1)

    y = y + 30
    for r in rows:
        if y > height - 86:
            cv2.putText(panel, "...", (cols["noaa"], y), font, 0.5, (160, 160, 160), 1, cv2.LINE_AA)
            break
        color = r["color"]
        cv2.putText(
            panel,
            str(r["noaa"]) if r["noaa"] else "-",
            (cols["noaa"], y),
            font,
            0.46,
            (225, 225, 225),
            1,
            cv2.LINE_AA,
        )
        grey = (200, 200, 205)
        cv2.putText(panel, str(r["harp"]), (cols["harp"], y), font, 0.46, grey, 1, cv2.LINE_AA)
        cv2.putText(panel, r["loc"], (cols["loc"], y), font, 0.46, grey, 1, cv2.LINE_AA)
        # risk bar + percentage
        bx0, bx1, bw = cols["bar"], width - 16, width - 16 - cols["bar"]
        cv2.rectangle(panel, (bx0, y - 11), (bx1, y + 1), (55, 55, 60), -1)
        if r["p"] is not None:
            fill = int(bx0 + bw * float(np.clip(r["p"], 0.0, 1.0)))
            cv2.rectangle(panel, (bx0, y - 11), (fill, y + 1), color, -1)
            pct = f"{r['p'] * 100:.0f}%"
            cv2.putText(panel, pct, (bx0 + 4, y - 1), font, 0.4, (15, 15, 15), 1, cv2.LINE_AA)
        else:
            cv2.putText(panel, "--", (bx0 + 4, y - 1), font, 0.4, (170, 170, 170), 1, cv2.LINE_AA)
        cv2.circle(panel, (width - 8, y - 5), 4, color, -1)
        y += 26

    # risk legend along the bottom
    ly = height - 58
    cv2.line(panel, (10, ly - 14), (width - 10, ly - 14), (60, 60, 66), 1)
    cv2.putText(panel, "RISK (>=M / 24h)", (14, ly), font, 0.42, (150, 154, 160), 1, cv2.LINE_AA)
    x = 14
    for _thr, label, color in RISK_BANDS:
        cv2.rectangle(panel, (x, ly + 10), (x + 12, ly + 22), color, -1)
        cv2.putText(panel, label, (x + 16, ly + 21), font, 0.36, (210, 210, 210), 1, cv2.LINE_AA)
        x += 92 if label != "Moderate" else 96
        if x > width - 90:
            x = 14
            ly += 26
    return panel


def compose_region_summary_frame(
    fulldisk_map,
    boxes_px: pd.DataFrame,
    probs: dict[int, float],
    time_utc: pd.Timestamp,
    model_note: str,
    size: int = 820,
    panel_width: int = 380,
) -> np.ndarray:
    """One Solar-Region-Summary frame: gray disk + risk boxes + summary table."""
    import cv2

    font = cv2.FONT_HERSHEY_SIMPLEX
    disk = render_magnetogram_disk(fulldisk_map, size)
    scale = size / fulldisk_map.data.shape[0]
    rows = region_rows(boxes_px, probs)
    by_harp = {r["harp"]: r for r in rows}

    for _, b in boxes_px.iterrows():
        r = by_harp[int(b["harpnum"])]
        color = r["color"]
        x0, x1 = int(b["x_min"] * scale), int(b["x_max"] * scale)
        y0 = int(size - b["y_max"] * scale)  # vertical flip: FITS y -> image row
        y1 = int(size - b["y_min"] * scale)
        cv2.rectangle(disk, (x0, y0), (x1, y1), color, 2)
        tag = f"AR {r['noaa']}" if r["noaa"] else f"HARP {r['harp']}"
        cv2.putText(disk, tag, (x0, max(14, y0 - 6)), font, 0.46, color, 1, cv2.LINE_AA)

    banner = np.full((52, size, 3), 18, dtype=np.uint8)
    cv2.putText(banner, "SOLAR REGION SUMMARY", (12, 26), font, 0.72, (255, 255, 255), 2)
    cv2.putText(
        banner,
        f"{time_utc:%Y-%m-%d %H:%M} UT   SDO/HMI   >=M-class 24 h flare risk",
        (12, 46),
        font,
        0.44,
        (190, 195, 205),
        1,
        cv2.LINE_AA,
    )
    footer = np.full((30, size, 3), 18, dtype=np.uint8)
    cv2.putText(footer, model_note, (12, 20), font, 0.4, (150, 150, 158), 1, cv2.LINE_AA)

    left = np.concatenate([banner, disk, footer], axis=0)
    panel = _draw_summary_panel(rows, left.shape[0], panel_width)
    return np.concatenate([left, panel], axis=1)


def render_region_summary(
    cfg,
    window_name: str,
    out_dir: str | Path,
    model,
    model_note: str,
    fps: int = 4,
    size: int = 820,
    max_prob_age_hours: float = 3.0,
) -> tuple[Path, int]:
    """Frames + MP4 of the Solar Region Summary for one window. Returns (mp4, n)."""
    import cv2
    import sunpy.map

    from solarflare.detect.bootstrap import boxes_to_pixels, fetch_harp_boxes
    from solarflare.viz.dashboard import build_probability_lookup
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

    samples_root = Path(cfg.paths.cache_dir) / "samples"
    sample_dirs = [p for p in samples_root.glob("*") if (p / "meta.json").exists()]
    lookback = int(cfg.forecast.lookback_hours * 60 / cfg.data.sample_cadence_minutes)
    lookup = build_probability_lookup(cfg, sample_dirs, model, lookback)

    out_dir = Path(out_dir)
    frames_dir = out_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    mp4_path = out_dir / f"region_summary_{window_name}.mp4"
    tol_boxes = pd.Timedelta(hours=cfg.detect.bootstrap_cadence_hours / 2)
    tol_prob = pd.Timedelta(hours=max_prob_age_hours)
    writer = None
    n = 0
    for fits_path in fits_files:
        smap = sunpy.map.Map(str(fits_path))
        t = pd.Timestamp(smap.date.datetime)
        near = boxes[(pd.to_datetime(boxes["time"]) - t).abs() <= tol_boxes]
        px = boxes_to_pixels(near, smap)
        probs: dict[int, float] = {}
        if len(lookup):
            recent = lookup[(t - pd.to_datetime(lookup["time"])).abs() <= tol_prob]
            for harp, grp in recent.groupby("harpnum"):
                nearest = grp.iloc[(pd.to_datetime(grp["time"]) - t).abs().argmin()]
                probs[int(harp)] = float(nearest["p"])
        frame = compose_region_summary_frame(smap, px, probs, t, model_note, size=size)
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
    log.info("region summary: %s (%d frames) + %s", mp4_path, n, frames_dir)
    return mp4_path, n
