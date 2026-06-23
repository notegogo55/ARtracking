"""solarflare command-line interface.

Commands:
  show-config        Validate and print the resolved config.
  check-credentials  Verify JSOC/drms and SunPy/HEK connectivity with setup hints.
  base-rate          Compute the climatological >=M base rate (baseline).
  resolve-harps      Cross-check config HARP numbers against the official JSOC mapping.
  fetch              Fetch + co-align + normalize + cache one AR sample (Gate G1).
  qa-overlay         Re-render the QA overlay from a cached sample.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Annotated

import typer
import yaml

from solarflare.config import Config, load_config
from solarflare.utils.explog import append_experiment_row
from solarflare.utils.logging import setup_logging
from solarflare.utils.seed import set_global_seed

app = typer.Typer(add_completion=False, help=__doc__)
log = logging.getLogger("solarflare.cli")

ConfigOpt = Annotated[
    Path, typer.Option("--config", "-c", help="Path to YAML config", exists=True, readable=True)
]
DEFAULT_CONFIG = Path("configs/default.yaml")

JSOC_REGISTER_URL = "http://jsoc.stanford.edu/ajax/register_email.html"


def _load(config_path: Path) -> Config:
    setup_logging()
    cfg = load_config(config_path)
    set_global_seed(cfg.project.seed)
    return cfg


@app.command("show-config")
def show_config(config: ConfigOpt = DEFAULT_CONFIG) -> None:
    """Validate the YAML config and print the fully resolved values."""
    cfg = _load(config)
    typer.echo(yaml.safe_dump(cfg.model_dump(mode="json"), sort_keys=False))
    typer.echo(f"config hash: {cfg.short_hash()}")


@app.command("check-credentials")
def check_credentials(config: ConfigOpt = DEFAULT_CONFIG) -> None:
    """Verify JSOC/drms access and SunPy Fido (HEK) connectivity.

    Exits non-zero if a required check fails, with instructions to fix it.
    """
    cfg = _load(config)
    failures: list[str] = []

    # 1. JSOC export email: must be set AND registered for cutout exports
    email = os.environ.get("JSOC_EMAIL", "")
    if email:
        try:
            import drms

            registered = bool(drms.Client().check_email(email))
        except Exception as err:  # noqa: BLE001
            typer.secho(f"[warn] could not verify JSOC email registration: {err}", fg="yellow")
            registered = True  # connectivity check below will catch real outages
        if registered:
            typer.secho(f"[ok]   JSOC_EMAIL is set and registered ({email})", fg="green")
        else:
            failures.append(
                f"JSOC_EMAIL '{email}' is NOT registered with JSOC.\n"
                f"       Register it at {JSOC_REGISTER_URL} (reply to their "
                "confirmation email), then re-run this check."
            )
    else:
        typer.secho(
            "[warn] JSOC_EMAIL not set. Exports in Phase A will fail.\n"
            f"       Register your email at {JSOC_REGISTER_URL}\n"
            '       then:  $env:JSOC_EMAIL = "you@example.com"   (PowerShell)',
            fg="yellow",
        )

    # 2. drms / JSOC metadata access (no registration required for queries)
    series = cfg.data.hmi_sharp_series
    try:
        import drms

        client = drms.Client()
        keys = client.query(f"{series}[377]", key="T_REC, NOAA_AR", n=1)
        if keys is None or len(keys) == 0:
            raise RuntimeError(f"empty result querying {series}[377]")
        noaa = keys["NOAA_AR"][0]
        typer.secho(f"[ok]   drms: queried {series} (HARP 377 -> NOAA {noaa})", fg="green")
    except Exception as err:  # noqa: BLE001 - report every failure mode with remedy
        failures.append(
            f"drms/JSOC query failed: {err}\n"
            "       Check network access to jsoc.stanford.edu and that `drms` is installed\n"
            "       (uv sync). JSOC has occasional maintenance windows — retry later."
        )

    # 3. SunPy Fido -> HEK (flare catalog used for labels)
    try:
        from sunpy.net import Fido
        from sunpy.net import attrs as a

        # 2017-09-06: AR 12673's X9.3 day — guaranteed to contain >=M events.
        result = Fido.search(
            a.Time("2017-09-06 00:00", "2017-09-07 00:00"),
            a.hek.EventType("FL"),
            a.hek.FL.GOESCls >= "M1.0",
            a.hek.OBS.Observatory == "GOES",
        )
        n = len(result["hek"])
        if n == 0:
            raise RuntimeError("HEK returned 0 flares for 2017-09-06 (expected several)")
        typer.secho(f"[ok]   SunPy/HEK: {n} >=M flares found on 2017-09-06", fg="green")
    except Exception as err:  # noqa: BLE001
        failures.append(
            f"SunPy/HEK query failed: {err}\n"
            "       Check network access and that sunpy[net] is installed (uv sync)."
        )

    if failures:
        for f in failures:
            typer.secho(f"[FAIL] {f}", fg="red")
        raise typer.Exit(code=1)
    typer.secho("All connectivity checks passed.", fg="green", bold=True)


@app.command("base-rate")
def base_rate(
    config: ConfigOpt = DEFAULT_CONFIG,
    events_csv: Annotated[
        Path | None,
        typer.Option(help="Offline mode: read events from CSV instead of querying HEK"),
    ] = None,
    start: Annotated[datetime | None, typer.Option(help="Override climatology start (UTC)")] = None,
    end: Annotated[datetime | None, typer.Option(help="Override climatology end (UTC)")] = None,
    no_log: Annotated[bool, typer.Option("--no-log", help="Skip the experiment-log entry")] = False,
) -> None:
    """Compute the climatological >=M-class base rate over the configured period."""
    from solarflare.data.goes_events import fetch_goes_events, load_events_csv
    from solarflare.forecast.climatology import compute_base_rate

    cfg = _load(config)
    t0 = start or cfg.climatology.start
    t1 = end or cfg.climatology.end
    threshold = cfg.forecast.flare_class_threshold

    if events_csv is not None:
        events = load_events_csv(events_csv)
        source = str(events_csv)
    else:
        events = fetch_goes_events(t0, t1, min_class=threshold, cache_dir=cfg.paths.cache_dir)
        source = "HEK (GOES observatory)"

    result = compute_base_rate(
        events, windows=[(t0, t1)], bin_hours=cfg.forecast.bin_hours, min_class=threshold
    )
    typer.echo(f"Period: {t0:%Y-%m-%d %H:%M} .. {t1:%Y-%m-%d %H:%M} UTC   (events: {source})")
    typer.echo(result.summary())

    if not no_log:
        append_experiment_row(
            cfg.paths.experiment_log,
            {
                "phase": "P0",
                "experiment": "climatology_base_rate",
                "config_hash": cfg.short_hash(),
                "seed": cfg.project.seed,
                "period_start": t0.isoformat(),
                "period_end": t1.isoformat(),
                "min_class": threshold,
                "bin_hours": cfg.forecast.bin_hours,
                "n_bins": result.n_bins,
                "n_positive_bins": result.n_positive_bins,
                "n_events": result.n_events,
                "base_rate": round(result.base_rate, 6),
                "brier_climatology": round(result.brier_climatology, 6),
                "tss": 0.0,
            },
        )


@app.command("resolve-harps")
def resolve_harps(
    config: ConfigOpt = DEFAULT_CONFIG,
    mapping_file: Annotated[
        Path | None,
        typer.Option(help="Offline mode: parse a local copy of the JSOC mapping file"),
    ] = None,
    strict: Annotated[
        bool, typer.Option("--strict", help="Exit non-zero on any mismatch/unresolved AR")
    ] = False,
) -> None:
    """Resolve/verify HARP numbers for every configured NOAA AR via the JSOC mapping."""
    from solarflare.data.harps import (
        fetch_harp_noaa_mapping,
        harps_for_noaa,
        parse_harp_noaa_mapping,
    )

    cfg = _load(config)
    if mapping_file is not None:
        mapping = parse_harp_noaa_mapping(mapping_file.read_text(encoding="utf-8"))
    else:
        mapping = fetch_harp_noaa_mapping(cfg.paths.cache_dir)

    problems = 0
    for window in cfg.study.windows:
        for target in window.targets:
            hits = harps_for_noaa(mapping, target.noaa)
            resolved = hits[0] if len(hits) == 1 else None
            if resolved is None:
                typer.secho(
                    f"[??]   {window.name}: NOAA {target.noaa} -> ambiguous/missing: {hits}",
                    fg="red",
                )
                problems += 1
            elif target.harp is None:
                typer.secho(
                    f"[new]  {window.name}: NOAA {target.noaa} -> HARP {resolved} "
                    f"(set `harp: {resolved}` in the config)",
                    fg="yellow",
                )
            elif target.harp != resolved:
                typer.secho(
                    f"[BAD]  {window.name}: NOAA {target.noaa} config says HARP "
                    f"{target.harp} but JSOC mapping says {resolved}",
                    fg="red",
                )
                problems += 1
            else:
                typer.secho(
                    f"[ok]   {window.name}: NOAA {target.noaa} -> HARP {resolved}", fg="green"
                )
    if strict and problems:
        raise typer.Exit(code=1)


@app.command("fetch")
def fetch(
    window: Annotated[str, typer.Option("--window", "-w", help="Study window name")],
    config: ConfigOpt = DEFAULT_CONFIG,
    noaa: Annotated[int | None, typer.Option(help="Target NOAA AR (default: first)")] = None,
    start: Annotated[datetime | None, typer.Option(help="Override window start (UTC)")] = None,
    end: Annotated[datetime | None, typer.Option(help="Override window end (UTC)")] = None,
    channel: Annotated[
        list[int] | None, typer.Option("--channel", help="AIA channel(s); default: config list")
    ] = None,
    skip_aia: Annotated[bool, typer.Option("--skip-aia", help="HMI + labels only")] = False,
    overwrite: Annotated[bool, typer.Option("--overwrite", help="Re-download raw FITS")] = False,
    email: Annotated[str, typer.Option(help="JSOC notify email (default: $JSOC_EMAIL)")] = "",
    overlay_channel: Annotated[int, typer.Option(help="Channel for the QA overlay")] = 171,
) -> None:
    """Gate G1: fetch, co-align, normalize and cache one AR sample + QA overlay."""
    from solarflare.data.cache import load_sample
    from solarflare.data.sample import build_sample
    from solarflare.viz.overlay import build_overlay

    cfg = _load(config)
    email = email or os.environ.get("JSOC_EMAIL", "")
    if not email:
        typer.secho(
            "No JSOC email. Register at "
            f'{JSOC_REGISTER_URL}\nthen:  $env:JSOC_EMAIL = "you@example.com"',
            fg="red",
        )
        raise typer.Exit(code=1)

    sample_dir = build_sample(
        cfg,
        window,
        noaa=noaa,
        start=start,
        end=end,
        channels=channel,
        email=email,
        skip_aia=skip_aia,
        overwrite=overwrite,
    )
    sample = load_sample(sample_dir)
    typer.echo(f"cached:  {sample_dir}")
    typer.echo(f"frames:  {sample.n_frames}   arrays: {sorted(sample.arrays)}")
    if not sample.qa.empty and "flagged" in sample.qa:
        flagged = sample.qa[sample.qa["flagged"].astype(bool)]
        typer.echo(f"QA:      {len(flagged)}/{len(sample.qa)} frame-channel entries flagged")
    typer.echo(f"labels:  {len(sample.labels)} GOES events for NOAA {sample.meta.get('noaa')}")
    if not skip_aia:
        overlay = build_overlay(sample, channel=overlay_channel)
        typer.echo(f"overlay: {overlay}")


@app.command("bootstrap-boxes")
def bootstrap_boxes(
    window: Annotated[str, typer.Option("--window", "-w")],
    config: ConfigOpt = DEFAULT_CONFIG,
) -> None:
    """Fetch HARP-derived AR boxes for a window (keyword query only, no images)."""
    from solarflare.data.sample import resolve_window_target  # noqa: F401  (validates name)
    from solarflare.detect.bootstrap import fetch_harp_boxes

    cfg = _load(config)
    win = next((w for w in cfg.study.windows if w.name == window), None)
    if win is None:
        typer.secho(f"unknown window {window!r}", fg="red")
        raise typer.Exit(code=1)
    boxes = fetch_harp_boxes(
        win.start, win.end, cfg.detect.bootstrap_cadence_hours, cfg.paths.cache_dir
    )
    typer.echo(
        f"{len(boxes)} boxes, {boxes['harpnum'].nunique() if len(boxes) else 0} HARPs, "
        f"{boxes['time'].nunique() if len(boxes) else 0} timesteps"
    )


@app.command("segment-sample")
def segment_sample_cmd(
    sample_dir: Annotated[Path, typer.Option("--sample-dir", exists=True, file_okay=False)],
    config: ConfigOpt = DEFAULT_CONFIG,
    model: Annotated[
        str | None,
        typer.Option(
            "--model",
            "--method",  # backward-compatible alias
            help="threshold | unet | surya | sam2 (default: config segment.model)",
        ),
    ] = None,
    weights: Annotated[
        Path | None, typer.Option(help="U-Net weights (default: config segment.unet_weights)")
    ] = None,
    qa_frame: Annotated[int | None, typer.Option(help="Frame for the QA plot")] = None,
) -> None:
    """AR masks for a cached sample via the selected Segmenter; writes masks + QA plot."""
    from solarflare.data.cache import load_sample
    from solarflare.detect.segment import segment_sample_auto, segmentation_qa_plot
    from solarflare.detect.segmenter import available_segmenters
    from solarflare.viz.overlay import pick_overlay_frame

    cfg = _load(config)
    if model is not None and model not in available_segmenters():
        typer.secho(f"unknown --model {model!r} ({' | '.join(available_segmenters())})", fg="red")
        raise typer.Exit(code=1)
    seg_cfg = cfg.segment.model_copy(
        update={
            **({"model": model} if model else {}),
            **({"unet_weights": weights} if weights else {}),
        }
    )
    sample = load_sample(sample_dir)
    masks_path, areas = segment_sample_auto(sample, seg_cfg)
    import numpy as np

    frame = qa_frame if qa_frame is not None else pick_overlay_frame(sample)
    qa_png = segmentation_qa_plot(sample, np.load(masks_path), frame)
    typer.echo(f"masks:   {masks_path}")
    typer.echo(
        f"areas:   median AR {int(areas['ar_pixels'].median())} px, "
        f"max {int(areas['ar_pixels'].max())} px over {len(areas)} frames"
    )
    typer.echo(f"qa plot: {qa_png}")


@app.command("train-unet")
def train_unet_cmd(
    config: ConfigOpt = DEFAULT_CONFIG,
    samples_root: Annotated[
        Path | None, typer.Option(help="Dir of cached samples (default: <cache_dir>/samples)")
    ] = None,
    epochs: Annotated[int | None, typer.Option(help="Override config unet_epochs")] = None,
    device: Annotated[str, typer.Option(help="cpu / cuda")] = "cpu",
    weights_out: Annotated[
        Path | None, typer.Option(help="Output weights (default: config segment.unet_weights)")
    ] = None,
) -> None:
    """Train the U-Net on threshold pseudo-labels from every cached sample.

    Time-blocked split: the tail `unet_val_fraction` of each sample's frames is
    validation-only. Saves the best-val-IoU checkpoint + training_log.csv.
    """
    from solarflare.data.cache import load_sample
    from solarflare.detect.unet import train_unet

    cfg = _load(config)
    root = samples_root or Path(cfg.paths.cache_dir) / "samples"
    sample_dirs = sorted(p for p in root.glob("*") if (p / "meta.json").exists())
    samples = []
    for sdir in sample_dirs:
        sample = load_sample(sdir)
        if {"hmi_continuum", "hmi_magnetogram"} <= set(sample.arrays):
            samples.append(sample)
        else:
            typer.secho(f"[skip] {sdir.name}: missing HMI continuum/magnetogram", fg="yellow")
    if not samples:
        typer.secho(f"no usable cached samples under {root}", fg="red")
        raise typer.Exit(code=1)
    typer.echo(f"training on {len(samples)} cached sample(s) from {root}")
    weights, history = train_unet(
        samples,
        cfg.segment,
        weights_path=weights_out,
        epochs=epochs,
        device=device,
        seed=cfg.project.seed,
    )
    best = history["val_iou"].max()
    typer.echo(history.round(4).to_string(index=False))
    typer.echo(f"weights: {weights}  (best val IoU {best:.4f})")
    typer.echo("use it with:  segment.model: unet  (or `segment-sample --model unet`)")
    append_experiment_row(
        cfg.paths.experiment_log,
        {
            "phase": "P2",
            "experiment": "unet_train",
            "config_hash": cfg.short_hash(),
            "seed": cfg.project.seed,
            "n_samples": len(samples),
            "epochs": len(history),
            "encoder": cfg.segment.unet_encoder,
            "best_val_iou": round(float(best), 4),
            "weights": str(weights),
        },
    )


@app.command("track-window")
def track_window(
    window: Annotated[str, typer.Option("--window", "-w")],
    config: ConfigOpt = DEFAULT_CONFIG,
    boxes_csv: Annotated[
        Path | None,
        typer.Option(help="Track these boxes instead of HARP bootstrap ones"),
    ] = None,
) -> None:
    """Run the temporal-IoU tracker over a window's boxes; writes the track table."""
    import pandas as pd

    from solarflare.detect.bootstrap import fetch_harp_boxes
    from solarflare.track.iou import track_boxes, track_report

    cfg = _load(config)
    win = next((w for w in cfg.study.windows if w.name == window), None)
    if win is None:
        typer.secho(f"unknown window {window!r}", fg="red")
        raise typer.Exit(code=1)
    if boxes_csv is not None:
        boxes = pd.read_csv(boxes_csv, parse_dates=["time"])
    else:
        boxes = fetch_harp_boxes(
            win.start, win.end, cfg.detect.bootstrap_cadence_hours, cfg.paths.cache_dir
        )
    tracked = track_boxes(boxes, cfg.track.iou_threshold, cfg.track.max_gap_frames)
    report = track_report(tracked)
    out_dir = Path(cfg.paths.outputs_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tracks_path = out_dir / f"tracks_{window}.csv"
    tracked.to_csv(tracks_path, index=False)
    typer.echo(report.to_string(index=False))
    typer.echo(
        f"tracks:  {tracks_path}  ({tracked['track_id'].nunique()} tracks, "
        f"{len(tracked)} observations)"
    )


@app.command("fetch-fulldisk")
def fetch_fulldisk(
    window: Annotated[str, typer.Option("--window", "-w", help="Study window name")],
    config: ConfigOpt = DEFAULT_CONFIG,
    email: Annotated[str, typer.Option(help="JSOC notify email (default: $JSOC_EMAIL)")] = "",
) -> None:
    """Download bounded rebinned full-disk magnetograms for a window (for the full-disk views).

    The frames land in data/raw/fulldisk/<window>/ and feed render-region-summary /
    render-harpmap / render-dashboard, which overlay the tracked HARP boxes on each frame.
    """
    from solarflare.detect.fulldisk import fetch_fulldisk_frames

    cfg = _load(config)
    email = email or os.environ.get("JSOC_EMAIL", "")
    if not email:
        typer.secho("JSOC email required (set JSOC_EMAIL or --email)", fg="red")
        raise typer.Exit(code=1)
    win = next((w for w in cfg.study.windows if w.name == window), None)
    if win is None:
        typer.secho(f"unknown window {window!r}", fg="red")
        raise typer.Exit(code=1)
    out_dir = Path(cfg.paths.data_root) / "raw" / "fulldisk" / window
    frames = fetch_fulldisk_frames(cfg, win, email, out_dir)
    typer.echo(f"frames:  {len(frames)} full-disk magnetograms in {out_dir}")


@app.command("build-features")
def build_features(
    sample_dir: Annotated[Path, typer.Option("--sample-dir", exists=True, file_okay=False)],
    config: ConfigOpt = DEFAULT_CONFIG,
) -> None:
    """Extract per-frame AR features for a cached sample (requires ar_masks.npy)."""
    import numpy as np

    from solarflare.data.cache import load_sample
    from solarflare.features.extract import extract_sample_features

    _load(config)  # logging + seed; extraction itself is config-free
    sample = load_sample(sample_dir)
    masks_path = sample_dir / "ar_masks.npy"
    if not masks_path.exists():
        typer.secho(
            f"no ar_masks.npy in {sample_dir} - run `solarflare segment-sample` first", fg="red"
        )
        raise typer.Exit(code=1)
    frame_df = extract_sample_features(sample, np.load(masks_path))
    out = sample_dir / "features_frame.csv"
    frame_df.to_csv(out, index=False)
    typer.echo(f"features: {out}  ({len(frame_df)} rows x {frame_df.shape[1] - 1} features)")


@app.command("build-dataset")
def build_dataset(
    config: ConfigOpt = DEFAULT_CONFIG,
    out: Annotated[
        Path | None, typer.Option(help="Output dir (default: data/datasets/seq_<ver>)")
    ] = None,
) -> None:
    """Gate G3: assemble the labeled, leakage-safe per-AR sequence dataset."""
    import numpy as np
    import pandas as pd

    from solarflare.data.cache import load_sample
    from solarflare.detect.bootstrap import fetch_harp_boxes
    from solarflare.features.dataset import build_sequences, write_dataset
    from solarflare.features.extract import build_frame_pipeline, extract_sample_features

    cfg = _load(config)
    samples_root = Path(cfg.paths.cache_dir) / "samples"
    sample_dirs = sorted(p for p in samples_root.glob("*") if (p / "meta.json").exists())
    if not sample_dirs:
        typer.secho(f"no cached samples under {samples_root}", fg="red")
        raise typer.Exit(code=1)

    lookback_steps = int(cfg.forecast.lookback_hours * 60 / cfg.data.sample_cadence_minutes)
    windows = {w.name: w for w in cfg.study.windows}
    all_X, all_rows, feature_names = [], [], None
    for sdir in sample_dirs:
        sample = load_sample(sdir)
        masks_path = sdir / "ar_masks.npy"
        if not masks_path.exists():
            typer.secho(f"[skip] {sdir.name}: no ar_masks.npy (run segment-sample)", fg="yellow")
            continue
        frame_csv = sdir / "features_frame.csv"
        if frame_csv.exists():
            frame_df = pd.read_csv(frame_csv, parse_dates=["time"])
        else:
            frame_df = extract_sample_features(sample, np.load(masks_path))
            frame_df.to_csv(frame_csv, index=False)
        features = build_frame_pipeline(frame_df, cfg.data.sample_cadence_minutes)

        meta = sample.meta
        window = windows.get(meta.get("window"))
        lon_series = None
        if window is not None:
            boxes = fetch_harp_boxes(
                window.start, window.end, cfg.detect.bootstrap_cadence_hours, cfg.paths.cache_dir
            )
            mine = boxes[boxes["harpnum"] == int(meta["harp"])]
            if len(mine):
                lon_series = mine.set_index("time")["lon_fwt"].dropna()
        X, rows, feature_names = build_sequences(
            features,
            sample.labels,
            noaa=int(meta["noaa"]),
            harp=int(meta["harp"]),
            window_name=str(meta.get("window")),
            lookback_steps=lookback_steps,
            lead_hours=cfg.forecast.lead_hours,
            min_class=cfg.forecast.flare_class_threshold,
            lon_series=lon_series,
            max_lon_deg=cfg.geometry.max_cm_longitude_deg,
            min_valid_fraction=cfg.features.min_valid_fraction,
            lead_grid=cfg.forecast.lead_grid,
            class_grid=cfg.forecast.class_grid,
        )
        typer.echo(
            f"{sdir.name}: {len(rows)} sequences "
            f"({int(rows['label'].sum()) if len(rows) else 0} positive)"
        )
        if len(rows):
            all_X.append(X)
            all_rows.append(rows)

    if not all_rows:
        typer.secho("no sequences built", fg="red")
        raise typer.Exit(code=1)
    X = np.concatenate(all_X)
    samples_df = pd.concat(all_rows, ignore_index=True)
    out_dir = out or Path("data/datasets") / f"seq_{cfg.features.dataset_version}"
    stats = write_dataset(
        out_dir,
        X,
        samples_df,
        feature_names,
        {
            "cadence_minutes": cfg.data.sample_cadence_minutes,
            "lookback_steps": lookback_steps,
            "lead_hours": cfg.forecast.lead_hours,
            "flare_class_threshold": cfg.forecast.flare_class_threshold,
            "lead_grid": cfg.forecast.lead_grid,
            "class_grid": cfg.forecast.class_grid,
            "max_cm_longitude_deg": cfg.geometry.max_cm_longitude_deg,
            "min_valid_fraction": cfg.features.min_valid_fraction,
            "config_hash": cfg.short_hash(),
        },
    )
    typer.echo(f"dataset: {out_dir}")
    typer.echo(
        f"class balance: {stats['n_positive']} pos / {stats['n_negative']} neg "
        f"(rate {stats['positive_rate']:.3f}); "
        f"missing cells: {stats['missing_fraction_overall']:.4f}"
    )
    append_experiment_row(
        cfg.paths.experiment_log,
        {
            "phase": "P3",
            "experiment": "build_dataset",
            "config_hash": cfg.short_hash(),
            "n_sequences": stats["n_sequences"],
            "n_positive": stats["n_positive"],
            "positive_rate": stats["positive_rate"],
            "missing_fraction": stats["missing_fraction_overall"],
        },
    )


@app.command("swansf-prepare")
def swansf_prepare(
    archive: Annotated[Path, typer.Option(exists=True, help="partitionN_instances.tar.gz")],
    out: Annotated[Path, typer.Option(help="Output dataset dir")],
    config: ConfigOpt = DEFAULT_CONFIG,
    max_instances: Annotated[int | None, typer.Option(help="Subsample cap")] = None,
) -> None:
    """Convert a SWAN-SF partition archive into the project dataset layout.

    Streams the tar.gz directly: member names contain ':' (illegal on NTFS),
    so the archive is never extracted to disk.
    """
    from solarflare.forecast.swansf import prepare_archive

    cfg = _load(config)
    stats = prepare_archive(archive, out, max_instances=max_instances, seed=cfg.project.seed)
    for key, value in stats.items():
        typer.echo(f"{key:>18}: {value}")


@app.command("forecast-benchmark")
def forecast_benchmark(
    dataset: Annotated[
        Path, typer.Option(exists=True, file_okay=False, help="Dir with X.npz + samples.parquet")
    ],
    config: ConfigOpt = DEFAULT_CONFIG,
    models: Annotated[
        str, typer.Option(help="Comma list: climatology,holt_winters,lstm,ensemble")
    ] = "climatology,holt_winters,lstm,ensemble",
    folds: Annotated[int, typer.Option()] = 5,
    lookback_steps: Annotated[
        int | None, typer.Option(help="Truncate sequences to the last N steps")
    ] = None,
    horizon_steps: Annotated[
        int, typer.Option(help="Holt-Winters extrapolation horizon (steps)")
    ] = 24,
    max_epochs: Annotated[int | None, typer.Option(help="Override LSTM epochs")] = None,
    embargo_hours: Annotated[
        float | None,
        typer.Option(help="Override split.embargo_hours (use small values for short datasets)"),
    ] = None,
    out: Annotated[Path | None, typer.Option(help="Output dir")] = None,
    tag: Annotated[str, typer.Option(help="Run tag for outputs/experiment log")] = "bench",
) -> None:
    """Gate G4: time-blocked CV metrics table (TSS primary) + reliability diagram."""
    import numpy as np
    import pandas as pd

    from solarflare.forecast.validate import (
        aggregate_table,
        crossval_table,
        reliability_plot,
    )

    cfg = _load(config)
    data = np.load(dataset / "X.npz", allow_pickle=False)
    X = data["X"]
    feature_names = [str(n) for n in data["feature_names"]]
    samples = pd.read_parquet(dataset / "samples.parquet")
    y = samples["label"].to_numpy(dtype=int)
    t0s = pd.to_datetime(samples["t0"])
    if lookback_steps:
        X = X[:, -lookback_steps:, :]
    out_dir = out or Path(cfg.paths.outputs_dir) / "forecast" / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    lstm_overrides = {"max_epochs": max_epochs} if max_epochs else None
    per_fold, oof = crossval_table(
        X,
        y,
        t0s,
        feature_names,
        horizon_steps=horizon_steps,
        n_folds=folds,
        embargo_hours=(embargo_hours if embargo_hours is not None else cfg.split.embargo_hours),
        seed=cfg.project.seed,
        model_names=tuple(m.strip() for m in models.split(",")),
        curves_dir=out_dir / "curves",
        lstm_overrides=lstm_overrides,
    )
    if per_fold.empty:
        typer.secho(
            "no usable folds (embargo wiped the training data?) - "
            "try --embargo-hours 0 or fewer folds",
            fg="red",
        )
        raise typer.Exit(code=1)
    per_fold.to_csv(out_dir / "metrics_per_fold.csv", index=False)
    table = aggregate_table(per_fold)
    table.to_csv(out_dir / "metrics_aggregate.csv", index=False)
    reliability_plot(oof, out_dir / "reliability.png")

    show_cols = ["model"] + [c for c in table.columns if c.startswith(("tss", "hss", "bss"))]
    typer.echo(table[show_cols].round(4).to_string(index=False))
    typer.echo(f"outputs: {out_dir}")
    for _, row in table.iterrows():
        append_experiment_row(
            cfg.paths.experiment_log,
            {
                "phase": "P4",
                "experiment": f"forecast_{tag}",
                "model": row["model"],
                "dataset": str(dataset),
                "folds": folds,
                "lookback_steps": lookback_steps or X.shape[1],
                "tss_mean": round(float(row["tss_mean"]), 4),
                "tss_std": round(float(row.get("tss_std", np.nan)), 4),
                "hss_mean": round(float(row["hss_mean"]), 4),
                "bss_mean": round(float(row["bss_mean"]), 4),
                "config_hash": cfg.short_hash(),
            },
        )


@app.command("forecast-grid")
def forecast_grid(
    dataset: Annotated[
        Path, typer.Option(exists=True, file_okay=False, help="Dir with X.npz + samples.parquet")
    ],
    config: ConfigOpt = DEFAULT_CONFIG,
    models: Annotated[
        str, typer.Option(help="Comma list: climatology,holt_winters,lstm,ensemble")
    ] = "climatology,holt_winters,lstm,ensemble",
    folds: Annotated[int, typer.Option()] = 5,
    max_epochs: Annotated[int | None, typer.Option(help="Override LSTM epochs")] = None,
    embargo_hours: Annotated[
        float | None, typer.Option(help="Override split.embargo_hours")
    ] = None,
    out: Annotated[Path | None, typer.Option(help="Output dir")] = None,
    tag: Annotated[str, typer.Option(help="Run tag for outputs/experiment log")] = "grid",
) -> None:
    """M3: TSS/HSS/BSS for every {horizon x class} cell (label_{H}h_{C}) + per-cell reliability."""
    import numpy as np
    import pandas as pd

    from solarflare.forecast.validate import crossval_grid, reliability_plot

    cfg = _load(config)
    data = np.load(dataset / "X.npz", allow_pickle=False)
    X = data["X"]
    feature_names = [str(n) for n in data["feature_names"]]
    samples = pd.read_parquet(dataset / "samples.parquet")
    out_dir = out or Path(cfg.paths.outputs_dir) / "forecast" / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    table, oof_by_cell = crossval_grid(
        X,
        samples,
        feature_names,
        horizon_steps=X.shape[1],
        n_folds=folds,
        embargo_hours=(embargo_hours if embargo_hours is not None else cfg.split.embargo_hours),
        seed=cfg.project.seed,
        model_names=tuple(m.strip() for m in models.split(",")),
        lstm_overrides={"max_epochs": max_epochs} if max_epochs else None,
    )
    if table.empty:
        typer.secho(
            "no usable grid cells (each had a single class, or embargo wiped the folds)", fg="red"
        )
        raise typer.Exit(code=1)
    table.to_csv(out_dir / "grid_metrics.csv", index=False)
    for col, oof in oof_by_cell.items():
        reliability_plot(oof, out_dir / f"reliability_{col}.png")

    show = ["label_col", "n_positive", "model", "tss_mean", "hss_mean", "bss_mean"]
    typer.echo(table[show].round(4).to_string(index=False))
    typer.echo(f"outputs: {out_dir}")
    for _, row in table.iterrows():
        append_experiment_row(
            cfg.paths.experiment_log,
            {
                "phase": "P4",
                "experiment": f"forecast_grid_{tag}",
                "label_col": row["label_col"],
                "horizon_h": row["horizon_h"],
                "class": row["class"],
                "n_positive": int(row["n_positive"]),
                "model": row["model"],
                "dataset": str(dataset),
                "tss_mean": round(float(row["tss_mean"]), 4),
                "hss_mean": round(float(row["hss_mean"]), 4),
                "bss_mean": round(float(row["bss_mean"]), 4),
                "config_hash": cfg.short_hash(),
            },
        )


@app.command("forecast-sweep")
def forecast_sweep(
    dataset: Annotated[Path, typer.Option(exists=True, file_okay=False)],
    config: ConfigOpt = DEFAULT_CONFIG,
    lookbacks: Annotated[str, typer.Option(help="Comma list of step counts")] = "30,60",
    folds: Annotated[int, typer.Option()] = 3,
    max_epochs: Annotated[int, typer.Option()] = 15,
    tag: Annotated[str, typer.Option()] = "sweep",
) -> None:
    """Look-back sweep: LSTM TSS (mean+/-std) per window length."""
    import numpy as np
    import pandas as pd

    from solarflare.forecast.validate import aggregate_table, crossval_table

    cfg = _load(config)
    data = np.load(dataset / "X.npz", allow_pickle=False)
    X_full = data["X"]
    feature_names = [str(n) for n in data["feature_names"]]
    samples = pd.read_parquet(dataset / "samples.parquet")
    y = samples["label"].to_numpy(dtype=int)
    t0s = pd.to_datetime(samples["t0"])
    out_dir = Path(cfg.paths.outputs_dir) / "forecast" / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for steps in [int(s) for s in lookbacks.split(",")]:
        per_fold, _ = crossval_table(
            X_full[:, -steps:, :],
            y,
            t0s,
            feature_names,
            horizon_steps=24,
            n_folds=folds,
            embargo_hours=cfg.split.embargo_hours,
            seed=cfg.project.seed,
            model_names=("lstm",),
            curves_dir=out_dir / f"curves_{steps}",
            lstm_overrides={"max_epochs": max_epochs},
        )
        agg = aggregate_table(per_fold).iloc[0]
        rows.append(
            {
                "lookback_steps": steps,
                "tss_mean": agg["tss_mean"],
                "tss_std": agg["tss_std"],
                "hss_mean": agg["hss_mean"],
            }
        )
        typer.echo(f"lookback {steps:>3} steps: TSS {agg['tss_mean']:.4f} +/- {agg['tss_std']:.4f}")
    pd.DataFrame(rows).to_csv(out_dir / "lookback_sweep.csv", index=False)
    typer.echo(f"sweep table: {out_dir / 'lookback_sweep.csv'}")


@app.command("run-all")
def run_all_cmd(
    window: Annotated[str, typer.Option("--window", "-w")],
    config: ConfigOpt = DEFAULT_CONFIG,
    noaa: Annotated[int | None, typer.Option()] = None,
    email: Annotated[
        str, typer.Option(help="JSOC email (only needed if fetch is not cached)")
    ] = "",
    force: Annotated[
        str, typer.Option(help="Comma list of stages to force-rebuild (fetch,segment,features)")
    ] = "",
    eval_models: Annotated[str, typer.Option()] = "climatology,holt_winters",
    eval_folds: Annotated[int, typer.Option()] = 2,
    eval_embargo_hours: Annotated[float, typer.Option()] = 2.0,
) -> None:
    """Gate G5: stages A->E end-to-end on one window, with per-stage caching."""
    import json

    from solarflare.pipeline import run_all

    cfg = _load(config)
    manifest = run_all(
        cfg,
        window,
        noaa=noaa,
        email=email or os.environ.get("JSOC_EMAIL", ""),
        force=frozenset(s.strip() for s in force.split(",") if s.strip()),
        eval_models=eval_models,
        eval_folds=eval_folds,
        eval_embargo_hours=eval_embargo_hours,
    )
    typer.echo(
        json.dumps(
            {
                "plan": manifest["plan"],
                "dataset": manifest["stages"]["dataset"],
                "evaluate": manifest["stages"]["evaluate"],
                "total_seconds": manifest["total_seconds"],
            },
            indent=2,
            default=str,
        )
    )


@app.command("forecast-holdout")
def forecast_holdout(
    train_dataset: Annotated[Path, typer.Option(exists=True, file_okay=False)],
    test_dataset: Annotated[Path, typer.Option(exists=True, file_okay=False)],
    config: ConfigOpt = DEFAULT_CONFIG,
    models: Annotated[str, typer.Option()] = "climatology,holt_winters,lstm,ensemble",
    max_epochs: Annotated[int | None, typer.Option()] = None,
    horizon_steps: Annotated[int, typer.Option()] = 24,
    tag: Annotated[str, typer.Option()] = "holdout",
) -> None:
    """Train on one time block, evaluate ONCE on a held-out block (+ROC/reliability)."""
    import numpy as np
    import pandas as pd

    from solarflare.forecast.validate import holdout_evaluate, reliability_plot, roc_plot

    cfg = _load(config)

    def _load_ds(d: Path):
        data = np.load(d / "X.npz", allow_pickle=False)
        samples = pd.read_parquet(d / "samples.parquet")
        return (
            data["X"],
            samples["label"].to_numpy(dtype=int),
            pd.to_datetime(samples["t0"]),
            [str(n) for n in data["feature_names"]],
        )

    X_tr, y_tr, t_tr, names_tr = _load_ds(train_dataset)
    X_te, y_te, _, names_te = _load_ds(test_dataset)
    if names_tr != names_te:
        typer.secho("train/test feature names differ", fg="red")
        raise typer.Exit(code=1)
    out_dir = Path(cfg.paths.outputs_dir) / "forecast" / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    table, predictions, _ = holdout_evaluate(
        X_tr,
        y_tr,
        t_tr,
        X_te,
        y_te,
        names_tr,
        horizon_steps=horizon_steps,
        seed=cfg.project.seed,
        model_names=tuple(m.strip() for m in models.split(",")),
        curves_dir=out_dir / "curves",
        lstm_overrides={"max_epochs": max_epochs} if max_epochs else None,
    )
    table.to_csv(out_dir / "holdout_metrics.csv", index=False)
    reliability_plot(predictions, out_dir / "reliability.png")
    roc_plot(predictions, out_dir / "roc.png")
    typer.echo(
        table[["model", "tss", "hss", "bss", "precision", "recall"]].round(4).to_string(index=False)
    )
    typer.echo(f"outputs: {out_dir}")
    for _, row in table.iterrows():
        append_experiment_row(
            cfg.paths.experiment_log,
            {
                "phase": "P5",
                "experiment": f"holdout_{tag}",
                "model": row["model"],
                "train": str(train_dataset),
                "test": str(test_dataset),
                "tss": round(float(row["tss"]), 4),
                "hss": round(float(row["hss"]), 4),
                "bss": round(float(row["bss"]), 4),
                "config_hash": cfg.short_hash(),
            },
        )


@app.command("ablate")
def ablate(
    train_dataset: Annotated[Path, typer.Option(exists=True, file_okay=False)],
    config: ConfigOpt = DEFAULT_CONFIG,
    test_dataset: Annotated[
        Path | None,
        typer.Option(help="Evaluate importance on this block (default: 20% tail of train)"),
    ] = None,
    max_epochs: Annotated[int, typer.Option()] = 15,
    n_repeats: Annotated[int, typer.Option()] = 3,
    drop_one: Annotated[
        str,
        typer.Option(
            help="Comma list of focus groups to also drop-one retrain "
            "(e.g. aia_0131,aia_0304,hmi_magnetogram... group names = base features)"
        ),
    ] = "",
    tag: Annotated[str, typer.Option()] = "ablation",
) -> None:
    """Gate G5 ablation: grouped permutation importance (+optional drop-one retrains)."""
    import numpy as np
    import pandas as pd

    from solarflare.eval.metrics import best_tss_threshold
    from solarflare.forecast.ablation import (
        ablation_bar_chart,
        drop_one_retrain,
        permutation_importance,
    )
    from solarflare.forecast.lstm import LSTMConfig, LSTMForecaster

    cfg = _load(config)
    data = np.load(train_dataset / "X.npz", allow_pickle=False)
    X = data["X"]
    names = [str(n) for n in data["feature_names"]]
    samples = pd.read_parquet(train_dataset / "samples.parquet")
    y = samples["label"].to_numpy(dtype=int)
    order = np.argsort(pd.to_datetime(samples["t0"]).to_numpy(), kind="stable")
    cut = max(int(0.8 * len(order)), 1)
    tr, va = order[:cut], order[cut:]

    lstm_cfg = LSTMConfig(seed=cfg.project.seed, max_epochs=max_epochs)
    model = LSTMForecaster(lstm_cfg, name="ablation_lstm").fit(X[tr], y[tr], X[va], y[va])
    thr, _ = best_tss_threshold(y[va], model.predict_proba(X[va]))

    if test_dataset is not None:
        te = np.load(test_dataset / "X.npz", allow_pickle=False)
        te_samples = pd.read_parquet(test_dataset / "samples.parquet")
        X_eval, y_eval = te["X"], te_samples["label"].to_numpy(dtype=int)
    else:
        X_eval, y_eval = X[va], y[va]

    out_dir = Path(cfg.paths.outputs_dir) / "forecast" / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    table = permutation_importance(
        model, X_eval, y_eval, names, thr, n_repeats=n_repeats, seed=cfg.project.seed
    )
    table.to_csv(out_dir / "permutation_importance.csv", index=False)
    ablation_bar_chart(
        table, out_dir / "permutation_importance.png", f"Permutation importance ({tag})"
    )
    typer.echo(table.round(4).to_string(index=False))

    if drop_one:
        focus = [g.strip() for g in drop_one.split(",") if g.strip()]

        def train_fn(X_tr, y_tr, X_va, y_va):
            return LSTMForecaster(lstm_cfg, name="dropone_lstm").fit(X_tr, y_tr, X_va, y_va)

        drop_table = drop_one_retrain(
            train_fn,
            X[tr],
            y[tr],
            X[va],
            y[va],
            X_eval,
            y_eval,
            names,
            focus,
            seed=cfg.project.seed,
        )
        drop_table.to_csv(out_dir / "drop_one.csv", index=False)
        typer.echo(drop_table.round(4).to_string(index=False))
    typer.echo(f"outputs: {out_dir}")


@app.command("ablate-layers")
def ablate_layers_cmd(
    dataset: Annotated[Path, typer.Option(exists=True, file_okay=False)],
    config: ConfigOpt = DEFAULT_CONFIG,
    models: Annotated[str, typer.Option(help="Comma list: holt_winters,lstm,ensemble")] = (
        "holt_winters,lstm"
    ),
    folds: Annotated[int, typer.Option()] = 3,
    max_epochs: Annotated[int | None, typer.Option(help="Override LSTM epochs")] = None,
    embargo_hours: Annotated[
        float | None, typer.Option(help="Override split.embargo_hours")
    ] = None,
    out: Annotated[Path | None, typer.Option()] = None,
    tag: Annotated[str, typer.Option()] = "layer_ablation",
) -> None:
    """M6: the 6-case atmospheric-layer ablation matrix (TSS/HSS/BSS per case x cell)."""
    import numpy as np
    import pandas as pd

    from solarflare.forecast.ablation import ablate_layers, layer_case_bar_chart

    cfg = _load(config)
    data = np.load(dataset / "X.npz", allow_pickle=False)
    X = data["X"]
    feature_names = [str(n) for n in data["feature_names"]]
    samples = pd.read_parquet(dataset / "samples.parquet")
    out_dir = out or Path(cfg.paths.outputs_dir) / "forecast" / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    table, _ = ablate_layers(
        X,
        samples,
        feature_names,
        n_folds=folds,
        embargo_hours=(embargo_hours if embargo_hours is not None else cfg.split.embargo_hours),
        seed=cfg.project.seed,
        model_names=tuple(m.strip() for m in models.split(",")),
        lstm_overrides={"max_epochs": max_epochs} if max_epochs else None,
    )
    if table.empty:
        typer.secho("no usable layer cases (single-class cells / embargo wiped folds)", fg="red")
        raise typer.Exit(code=1)
    table.to_csv(out_dir / "layer_ablation.csv", index=False)
    for model in table["model"].unique():
        for label_col in table["label_col"].unique():
            layer_case_bar_chart(
                table, out_dir / f"tss_by_case_{model}_{label_col}.png", model, label_col
            )

    show = ["case", "n_aia_channels", "label_col", "model", "tss_mean", "hss_mean", "bss_mean"]
    typer.echo(table[show].round(4).to_string(index=False))
    typer.echo(f"outputs: {out_dir}")
    for _, row in table.iterrows():
        append_experiment_row(
            cfg.paths.experiment_log,
            {
                "phase": "P5",
                "experiment": f"layer_ablation_{tag}",
                "case": row["case"],
                "n_aia_channels": int(row["n_aia_channels"]),
                "label_col": row["label_col"],
                "model": row["model"],
                "tss_mean": round(float(row["tss_mean"]), 4),
                "hss_mean": round(float(row["hss_mean"]), 4),
                "bss_mean": round(float(row["bss_mean"]), 4),
                "config_hash": cfg.short_hash(),
            },
        )


@app.command("render-video")
def render_video(
    sample_dir: Annotated[Path, typer.Option("--sample-dir", exists=True, file_okay=False)],
    config: ConfigOpt = DEFAULT_CONFIG,
    channel: Annotated[int, typer.Option(help="AIA channel for the third panel")] = 171,
    start: Annotated[datetime | None, typer.Option(help="Clip start (UTC)")] = None,
    end: Annotated[datetime | None, typer.Option(help="Clip end (UTC)")] = None,
    fps: Annotated[int, typer.Option()] = 12,
    out: Annotated[Path | None, typer.Option(help="Output .mp4")] = None,
) -> None:
    """MP4 of a cached sample: continuum | B_los | AIA with AR-mask contours."""
    import numpy as np

    from solarflare.data.cache import load_sample
    from solarflare.viz.video import render_sample_video

    _load(config)
    sample = load_sample(sample_dir)
    masks_path = sample_dir / "ar_masks.npy"
    if not masks_path.exists():
        typer.secho("no ar_masks.npy - run `solarflare segment-sample` first", fg="red")
        raise typer.Exit(code=1)
    out = out or sample_dir / f"video_{channel:04d}.mp4"
    path = render_sample_video(
        sample, np.load(masks_path), out, channel=channel, start=start, end=end, fps=fps
    )
    typer.echo(f"video: {path}")


@app.command("render-dashboard")
def render_dashboard_cmd(
    window: Annotated[str, typer.Option("--window", "-w")],
    config: ConfigOpt = DEFAULT_CONFIG,
    dataset: Annotated[
        Path, typer.Option(help="Sequence dataset used to fit the probability model")
    ] = Path("data/datasets/seq_v1"),
    model: Annotated[str, typer.Option(help="holt_winters | lstm")] = "holt_winters",
    fps: Annotated[int, typer.Option()] = 4,
    out: Annotated[Path | None, typer.Option()] = None,
) -> None:
    """Full-disk per-AR flare-probability dashboard: PNG frames + MP4 clip."""
    import numpy as np
    import pandas as pd

    from solarflare.forecast.validate import fit_model, make_models
    from solarflare.viz.dashboard import render_dashboard

    cfg = _load(config)
    data = np.load(dataset / "X.npz", allow_pickle=False)
    X = data["X"]
    names = [str(n) for n in data["feature_names"]]
    samples = pd.read_parquet(dataset / "samples.parquet")
    y = samples["label"].to_numpy(dtype=int)
    order = np.argsort(pd.to_datetime(samples["t0"]).to_numpy(), kind="stable")
    cut = max(int(0.8 * len(order)), 1)
    factories = make_models(names, horizon_steps=X.shape[1], seed=cfg.project.seed)
    fitted = fit_model(
        factories[model], X[order[:cut]], y[order[:cut]], X[order[cut:]], y[order[cut:]]
    )
    note = f"model: {model} fitted on {dataset.name} (n={len(samples)}, pos={int(y.sum())})" + (
        "  [MVP - anecdotal sample size]" if len(samples) < 200 else ""
    )
    out_dir = out or Path(cfg.paths.outputs_dir) / "dashboard" / window
    mp4, n = render_dashboard(cfg, window, out_dir, fitted, note, fps=fps)
    typer.echo(f"clip:   {mp4}  ({n} frames)")
    typer.echo(f"frames: {out_dir / 'frames'}")


@app.command("render-region-summary")
def render_region_summary_cmd(
    window: Annotated[str, typer.Option("--window", "-w")],
    config: ConfigOpt = DEFAULT_CONFIG,
    dataset: Annotated[
        Path, typer.Option(help="Sequence dataset used to fit the probability model")
    ] = Path("data/datasets/seq_v1"),
    model: Annotated[str, typer.Option(help="holt_winters | lstm")] = "holt_winters",
    fps: Annotated[int, typer.Option()] = 4,
    size: Annotated[int, typer.Option(help="Disk panel size in px")] = 820,
    out: Annotated[Path | None, typer.Option()] = None,
) -> None:
    """Operational NOAA SWPC-style Solar Region Summary: PNG frames + MP4 clip."""
    import numpy as np
    import pandas as pd

    from solarflare.forecast.validate import fit_model, make_models
    from solarflare.viz.regionsummary import render_region_summary

    cfg = _load(config)
    data = np.load(dataset / "X.npz", allow_pickle=False)
    X = data["X"]
    names = [str(n) for n in data["feature_names"]]
    samples = pd.read_parquet(dataset / "samples.parquet")
    y = samples["label"].to_numpy(dtype=int)
    order = np.argsort(pd.to_datetime(samples["t0"]).to_numpy(), kind="stable")
    cut = max(int(0.8 * len(order)), 1)
    factories = make_models(names, horizon_steps=X.shape[1], seed=cfg.project.seed)
    fitted = fit_model(
        factories[model], X[order[:cut]], y[order[:cut]], X[order[cut:]], y[order[cut:]]
    )
    note = f"model: {model} fitted on {dataset.name} (n={len(samples)}, pos={int(y.sum())})" + (
        "  [MVP - anecdotal sample size]" if len(samples) < 200 else ""
    )
    out_dir = out or Path(cfg.paths.outputs_dir) / "region_summary" / window
    mp4, n = render_region_summary(cfg, window, out_dir, fitted, note, fps=fps, size=size)
    typer.echo(f"clip:   {mp4}  ({n} frames)")
    typer.echo(f"frames: {out_dir / 'frames'}")


@app.command("render-harpmap")
def render_harpmap_cmd(
    window: Annotated[str, typer.Option("--window", "-w")],
    config: ConfigOpt = DEFAULT_CONFIG,
    fps: Annotated[int, typer.Option()] = 4,
    size: Annotated[int, typer.Option(help="Disk panel size in px")] = 880,
    out: Annotated[Path | None, typer.Option()] = None,
) -> None:
    """JSOC-style Tracked AR (HARP) full-disk map: PNG frames + MP4 clip."""
    from solarflare.viz.harpmap import render_harpmap

    cfg = _load(config)
    out_dir = out or Path(cfg.paths.outputs_dir) / "harpmap" / window
    mp4, n = render_harpmap(cfg, window, out_dir, fps=fps, size=size)
    typer.echo(f"clip:   {mp4}  ({n} frames)")
    typer.echo(f"frames: {out_dir / 'frames'}")


@app.command("qa-overlay")
def qa_overlay(
    sample_dir: Annotated[Path, typer.Option("--sample-dir", exists=True, file_okay=False)],
    channel: Annotated[int, typer.Option(help="AIA channel to show")] = 171,
    frame: Annotated[int | None, typer.Option(help="Frame index (default: flare peak)")] = None,
    out: Annotated[Path | None, typer.Option(help="Output PNG path")] = None,
) -> None:
    """Re-render the QA overlay from a cached sample directory."""
    from solarflare.data.cache import load_sample
    from solarflare.viz.overlay import build_overlay

    setup_logging()
    sample = load_sample(sample_dir)
    path = build_overlay(sample, channel=channel, frame_idx=frame, out_path=out)
    typer.echo(f"overlay: {path}")


if __name__ == "__main__":
    app()
