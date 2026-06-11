"""End-to-end pipeline orchestrator: stages A->E on one study window.

One config-driven entry point (`solarflare run-all`) that reuses every stage's
cache: raw FITS and the sample cache (A), AR masks (B), frame features (C) are
only rebuilt when missing or forced; the labeled dataset (D) and evaluation (E)
are cheap and deterministic (seeded), so they are rebuilt every run and their
outputs double as the reproducibility check. A manifest.json records per-stage
status, key statistics, config hash and versions.
"""

from __future__ import annotations

import json
import logging
import time
from datetime import UTC, datetime
from pathlib import Path

from solarflare import __version__
from solarflare.config import Config

log = logging.getLogger(__name__)

STAGES = ("fetch", "segment", "features", "dataset", "evaluate")


def plan_stages(sample_dir: Path, force: frozenset[str] = frozenset()) -> dict[str, str]:
    """Decide build vs cached per stage from the files present on disk."""
    plan = {}
    plan["fetch"] = ("build" if "fetch" in force or not (sample_dir / "meta.json").exists()
                     else "cached")
    plan["segment"] = ("build" if "segment" in force or plan["fetch"] == "build"
                       or not (sample_dir / "ar_masks.npy").exists() else "cached")
    plan["features"] = ("build" if "features" in force or plan["segment"] == "build"
                        or not (sample_dir / "features_frame.csv").exists() else "cached")
    # dataset + evaluation are cheap, deterministic, and always rebuilt
    plan["dataset"] = "build"
    plan["evaluate"] = "build"
    return plan


def run_all(
    cfg: Config,
    window_name: str,
    noaa: int | None = None,
    email: str = "",
    force: frozenset[str] = frozenset(),
    eval_models: str = "climatology,holt_winters",
    eval_folds: int = 2,
    eval_embargo_hours: float = 2.0,
    out_root: Path | None = None,
) -> dict:
    """Run stages A->E for one window; returns (and writes) the manifest."""
    import numpy as np
    import pandas as pd

    from solarflare.data.cache import load_sample
    from solarflare.data.harps import fetch_harp_noaa_mapping, resolve_harp
    from solarflare.data.sample import build_sample, resolve_window_target, sample_dir_name
    from solarflare.detect.bootstrap import fetch_harp_boxes
    from solarflare.detect.segment import segment_sample
    from solarflare.features.dataset import build_sequences, write_dataset
    from solarflare.features.extract import build_frame_pipeline, extract_sample_features
    from solarflare.forecast.validate import aggregate_table, crossval_table

    t_start = time.monotonic()
    window, target = resolve_window_target(cfg, window_name, noaa)
    harp = target.harp
    if harp is None:
        harp = resolve_harp(fetch_harp_noaa_mapping(cfg.paths.cache_dir), target.noaa)
    sample_dir = Path(cfg.paths.cache_dir) / "samples" / sample_dir_name(harp, window_name)
    plan = plan_stages(sample_dir, force)
    log.info("run-all %s (HARP %d): plan=%s", window_name, harp, plan)
    manifest: dict = {
        "window": window_name, "noaa": target.noaa, "harp": harp,
        "config_hash": cfg.short_hash(), "package_version": __version__,
        "started_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "plan": plan, "stages": {},
    }

    def record(stage: str, t0: float, **info) -> None:
        manifest["stages"][stage] = {
            "status": plan[stage], "seconds": round(time.monotonic() - t0, 2), **info}

    # --- A: fetch ----------------------------------------------------------
    t0 = time.monotonic()
    if plan["fetch"] == "build":
        if not email:
            raise ValueError("stage A needs a JSOC-registered email "
                             "(set JSOC_EMAIL) or an existing sample cache")
        build_sample(cfg, window_name, noaa=noaa, email=email)
    sample = load_sample(sample_dir)
    record("fetch", t0, n_frames=sample.n_frames, arrays=sorted(sample.arrays))

    # --- B: segment --------------------------------------------------------
    t0 = time.monotonic()
    masks_path = sample_dir / "ar_masks.npy"
    if plan["segment"] == "build":
        segment_sample(sample, cfg.segment)
    masks = np.load(masks_path)
    record("segment", t0, median_ar_pixels=int(np.median(masks.sum(axis=(1, 2)))))

    # --- C: features -------------------------------------------------------
    t0 = time.monotonic()
    frame_csv = sample_dir / "features_frame.csv"
    if plan["features"] == "build":
        extract_sample_features(sample, masks).to_csv(frame_csv, index=False)
    frame_df = pd.read_csv(frame_csv, parse_dates=["time"])
    record("features", t0, n_rows=len(frame_df), n_features=frame_df.shape[1] - 1)

    # --- D: labeled dataset -------------------------------------------------
    t0 = time.monotonic()
    features = build_frame_pipeline(frame_df, cfg.data.sample_cadence_minutes)
    boxes = fetch_harp_boxes(window.start, window.end,
                             cfg.detect.bootstrap_cadence_hours, cfg.paths.cache_dir)
    mine = boxes[boxes["harpnum"] == harp]
    lon_series = mine.set_index("time")["lon_fwt"].dropna() if len(mine) else None
    lookback_steps = int(cfg.forecast.lookback_hours * 60 / cfg.data.sample_cadence_minutes)
    X, rows, feature_names = build_sequences(
        features, sample.labels, noaa=target.noaa, harp=harp, window_name=window_name,
        lookback_steps=lookback_steps, lead_hours=cfg.forecast.lead_hours,
        min_class=cfg.forecast.flare_class_threshold, lon_series=lon_series,
        max_lon_deg=cfg.geometry.max_cm_longitude_deg,
        min_valid_fraction=cfg.features.min_valid_fraction,
    )
    out_root = out_root or Path(cfg.paths.outputs_dir) / "runs" / (
        f"{window_name}_{cfg.short_hash()}")
    dataset_dir = out_root / "dataset"
    stats = write_dataset(dataset_dir, X, rows, feature_names,
                          {"config_hash": cfg.short_hash(), "window": window_name})
    record("dataset", t0, **{k: stats[k] for k in
                             ("n_sequences", "n_positive", "positive_rate",
                              "missing_fraction_overall")})

    # --- E: evaluation -----------------------------------------------------
    t0 = time.monotonic()
    per_fold, _ = crossval_table(
        X, rows["label"].to_numpy(dtype=int), rows["t0"], feature_names,
        horizon_steps=lookback_steps, n_folds=eval_folds,
        embargo_hours=eval_embargo_hours, seed=cfg.project.seed,
        model_names=tuple(m.strip() for m in eval_models.split(",")),
    )
    table = aggregate_table(per_fold)
    table.to_csv(out_root / "metrics.csv", index=False)
    record("evaluate", t0, models=list(table["model"]),
           tss_mean={r["model"]: round(float(r["tss_mean"]), 4)
                     for _, r in table.iterrows()})

    manifest["total_seconds"] = round(time.monotonic() - t_start, 2)
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    log.info("run-all finished in %.1fs -> %s", manifest["total_seconds"], out_root)
    return manifest


def reproducibility_keys(manifest: dict) -> dict:
    """The deterministic subset of a manifest used to compare two runs."""
    return {
        "config_hash": manifest["config_hash"],
        "dataset": {k: v for k, v in manifest["stages"]["dataset"].items()
                    if k not in ("seconds", "status")},
        "evaluate": {k: v for k, v in manifest["stages"]["evaluate"].items()
                     if k not in ("seconds", "status")},
    }
