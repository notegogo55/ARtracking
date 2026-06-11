"""Full-disk flare-probability dashboard: clip + frames (DeFN-style view).

Each frame: the real (rebinned) HMI full-disk magnetogram rendered as a solar
disk, every HARP box drawn and labeled, and — for ARs that have a cached
sample — the model's >=M probability colored by band (C/M/X palette). Frames
are saved as PNGs and encoded to MP4 (OpenCV mp4v).

Probabilities come from a pluggable fitted model (predict_proba over the AR's
feature sequence at the frame time). With only the MVP dataset behind it the
numbers are anecdotal — the footer always names the model + dataset so a
rendered clip can never be mistaken for an operational forecast.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# legend palette (BGR for cv2): C / M / X bands
BAND_COLORS = {"C": (246, 130, 59), "M": (106, 168, 79), "X": (0, 165, 255)}
NO_DATA_COLOR = (140, 140, 140)


def probability_band(p_m: float) -> str:
    """Color band for a >=M probability (X band when very likely)."""
    if p_m >= 0.6:
        return "X"
    if p_m >= 0.25:
        return "M"
    return "C"


def render_disk(fulldisk_map, size: int = 880) -> np.ndarray:
    """Solar-disk RGB: orange limb-darkened base, |B| regions darkened on top."""
    import cv2

    data = np.asarray(fulldisk_map.data, dtype=np.float32)
    h, w = data.shape
    yy, xx = np.mgrid[0:h, 0:w]
    cx = float(fulldisk_map.wcs.wcs.crpix[0]) - 1
    cy = float(fulldisk_map.wcs.wcs.crpix[1]) - 1
    r_pix = float(fulldisk_map.rsun_obs.to_value("arcsec")
                  / abs(fulldisk_map.scale[0].value))
    rr = np.hypot(xx - cx, yy - cy) / r_pix
    on_disk = rr <= 1.0

    # limb-darkened orange base + |B| darkening (real magnetic structures)
    base = np.clip(1.0 - 0.45 * rr**2, 0.0, 1.0)
    babs = np.abs(np.nan_to_num(data, nan=0.0))
    base *= np.clip(1.0 - babs / 600.0, 0.15, 1.0)
    shade = np.where(on_disk, 0.25 + 0.75 * base, 0.0)
    import matplotlib

    rgb = (matplotlib.colormaps["afmhot"](shade)[..., :3] * 255).astype(np.uint8)
    rgb[~on_disk] = (8, 8, 12)
    rgb = np.flipud(rgb)  # FITS bottom-left -> image top-left
    return cv2.resize(rgb, (size, size), interpolation=cv2.INTER_AREA)


def compose_dashboard_frame(
    fulldisk_map, boxes_px: pd.DataFrame, probs: dict[int, float],
    time_utc: pd.Timestamp, model_note: str, size: int = 880,
) -> np.ndarray:
    """One dashboard frame. boxes_px needs harpnum/noaa_ar + x/y_min/max (FITS px)."""
    import cv2

    disk = render_disk(fulldisk_map, size)
    h_img = disk.shape[0]
    scale = size / fulldisk_map.data.shape[0]
    rows = []
    for _, b in boxes_px.iterrows():
        harp = int(b["harpnum"])
        p = probs.get(harp)
        band = probability_band(p) if p is not None else None
        color = BAND_COLORS[band] if band else NO_DATA_COLOR
        x0, x1 = int(b["x_min"] * scale), int(b["x_max"] * scale)
        # vertical flip: FITS y -> image row
        y0 = int(h_img - b["y_max"] * scale)
        y1 = int(h_img - b["y_min"] * scale)
        cv2.rectangle(disk, (x0, y0), (x1, y1), color, 2)
        noaa = int(b.get("noaa_ar", 0) or 0)
        tag = f"AR {noaa}" if noaa else f"HARP {harp}"
        label = f"{tag}  {p * 100:.0f}%" if p is not None else tag
        cv2.putText(disk, label, (x0, max(14, y0 - 6)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, color, 1, cv2.LINE_AA)
        rows.append((tag, p, band))

    header = np.zeros((46, size, 3), dtype=np.uint8)
    cv2.putText(header, "Solar Flare Probability Dashboard", (12, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(header, f"{time_utc:%Y-%m-%d %H:%M} UT   P(>=M, 24h)", (12, 42),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)

    footer = np.zeros((58, size, 3), dtype=np.uint8)
    x = 12
    for band, color in BAND_COLORS.items():
        cv2.rectangle(footer, (x, 10), (x + 14, 24), color, -1)
        cv2.putText(footer, f"{band}-band", (x + 20, 22), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (220, 220, 220), 1, cv2.LINE_AA)
        x += 110
    cv2.rectangle(footer, (x, 10), (x + 14, 24), NO_DATA_COLOR, -1)
    cv2.putText(footer, "no sample", (x + 20, 22), cv2.FONT_HERSHEY_SIMPLEX,
                0.45, (220, 220, 220), 1, cv2.LINE_AA)
    cv2.putText(footer, model_note, (12, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                (150, 150, 150), 1, cv2.LINE_AA)
    return np.concatenate([header, disk, footer], axis=0)


def build_probability_lookup(
    cfg, sample_dirs: list[Path], model, lookback_steps: int,
) -> pd.DataFrame:
    """[time, harpnum, p] for every issuance hour of every cached sample."""
    from solarflare.data.cache import load_sample
    from solarflare.features.extract import build_frame_pipeline

    frames = []
    for sdir in sample_dirs:
        sample = load_sample(sdir)
        frame_csv = Path(sdir) / "features_frame.csv"
        if not frame_csv.exists():
            continue
        features = build_frame_pipeline(
            pd.read_csv(frame_csv, parse_dates=["time"]),
            cfg.data.sample_cadence_minutes)
        cols = [c for c in features.columns if c != "time"]
        values = features[cols].to_numpy(dtype=np.float32)
        times = pd.to_datetime(features["time"])
        windows, t0s = [], []
        for end in range(lookback_steps - 1, len(features)):
            windows.append(values[end - lookback_steps + 1: end + 1])
            t0s.append(times.iloc[end])
        if not windows:
            continue
        p = model.predict_proba(np.stack(windows))
        frames.append(pd.DataFrame({
            "time": t0s, "harpnum": int(sample.meta["harp"]), "p": p}))
    if not frames:
        return pd.DataFrame(columns=["time", "harpnum", "p"])
    return pd.concat(frames, ignore_index=True)


def render_dashboard(
    cfg, window_name: str, out_dir: str | Path, model, model_note: str,
    fps: int = 4, max_prob_age_hours: float = 3.0,
) -> tuple[Path, int]:
    """Frames + MP4 for one study window. Returns (mp4 path, n frames)."""
    import cv2
    import sunpy.map

    from solarflare.detect.bootstrap import boxes_to_pixels, fetch_harp_boxes

    window = next(w for w in cfg.study.windows if w.name == window_name)
    fulldisk_dir = Path(cfg.paths.data_root) / "raw" / "fulldisk" / window_name
    fits_files = sorted(fulldisk_dir.glob("*.fits"))
    if not fits_files:
        raise FileNotFoundError(
            f"no full-disk frames under {fulldisk_dir} - run build-detect-dataset")
    boxes = fetch_harp_boxes(window.start, window.end,
                             cfg.detect.bootstrap_cadence_hours, cfg.paths.cache_dir)

    samples_root = Path(cfg.paths.cache_dir) / "samples"
    sample_dirs = [p for p in samples_root.glob("*") if (p / "meta.json").exists()]
    lookback = int(cfg.forecast.lookback_hours * 60 / cfg.data.sample_cadence_minutes)
    lookup = build_probability_lookup(cfg, sample_dirs, model, lookback)

    out_dir = Path(out_dir)
    frames_dir = out_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    writer = None
    mp4_path = out_dir / f"dashboard_{window_name}.mp4"
    tol_boxes = pd.Timedelta(hours=cfg.detect.bootstrap_cadence_hours / 2)
    tol_prob = pd.Timedelta(hours=max_prob_age_hours)
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
        frame = compose_dashboard_frame(smap, px, probs, t, model_note)
        cv2.imwrite(str(frames_dir / f"{window_name}_{t:%Y%m%dT%H%M}.png"),
                    cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        if writer is None:
            writer = cv2.VideoWriter(str(mp4_path), cv2.VideoWriter_fourcc(*"mp4v"),
                                     fps, (frame.shape[1], frame.shape[0]))
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        n += 1
    if writer is not None:
        writer.release()
    log.info("dashboard: %s (%d frames) + %s", mp4_path, n, frames_dir)
    return mp4_path, n
