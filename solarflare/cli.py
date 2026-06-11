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
            f"{JSOC_REGISTER_URL}\nthen:  $env:JSOC_EMAIL = \"you@example.com\"",
            fg="red",
        )
        raise typer.Exit(code=1)

    sample_dir = build_sample(
        cfg, window, noaa=noaa, start=start, end=end, channels=channel,
        email=email, skip_aia=skip_aia, overwrite=overwrite,
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
    boxes = fetch_harp_boxes(win.start, win.end, cfg.detect.bootstrap_cadence_hours,
                             cfg.paths.cache_dir)
    typer.echo(f"{len(boxes)} boxes, {boxes['harpnum'].nunique() if len(boxes) else 0} HARPs, "
               f"{boxes['time'].nunique() if len(boxes) else 0} timesteps")


@app.command("segment-sample")
def segment_sample_cmd(
    sample_dir: Annotated[Path, typer.Option("--sample-dir", exists=True, file_okay=False)],
    config: ConfigOpt = DEFAULT_CONFIG,
    qa_frame: Annotated[int | None, typer.Option(help="Frame for the QA plot")] = None,
) -> None:
    """Threshold+morphology AR masks for a cached sample; writes masks + QA plot."""
    from solarflare.data.cache import load_sample
    from solarflare.detect.segment import segment_sample, segmentation_qa_plot
    from solarflare.viz.overlay import pick_overlay_frame

    cfg = _load(config)
    sample = load_sample(sample_dir)
    masks_path, areas = segment_sample(sample, cfg.segment)
    import numpy as np

    frame = qa_frame if qa_frame is not None else pick_overlay_frame(sample)
    qa_png = segmentation_qa_plot(sample, np.load(masks_path), frame)
    typer.echo(f"masks:   {masks_path}")
    typer.echo(f"areas:   median AR {int(areas['ar_pixels'].median())} px, "
               f"max {int(areas['ar_pixels'].max())} px over {len(areas)} frames")
    typer.echo(f"qa plot: {qa_png}")


@app.command("track-window")
def track_window(
    window: Annotated[str, typer.Option("--window", "-w")],
    config: ConfigOpt = DEFAULT_CONFIG,
    boxes_csv: Annotated[
        Path | None, typer.Option(help="Track these boxes instead of HARP bootstrap ones"),
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
        boxes = fetch_harp_boxes(win.start, win.end, cfg.detect.bootstrap_cadence_hours,
                                 cfg.paths.cache_dir)
    tracked = track_boxes(boxes, cfg.track.iou_threshold, cfg.track.max_gap_frames)
    report = track_report(tracked)
    out_dir = Path(cfg.paths.outputs_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tracks_path = out_dir / f"tracks_{window}.csv"
    tracked.to_csv(tracks_path, index=False)
    typer.echo(report.to_string(index=False))
    typer.echo(f"tracks:  {tracks_path}  ({tracked['track_id'].nunique()} tracks, "
               f"{len(tracked)} observations)")


@app.command("build-detect-dataset")
def build_detect_dataset(
    config: ConfigOpt = DEFAULT_CONFIG,
    email: Annotated[str, typer.Option(help="JSOC notify email (default: $JSOC_EMAIL)")] = "",
    out: Annotated[Path, typer.Option(help="Dataset root")] = Path("data/detect_dataset"),
) -> None:
    """Download bounded full-disk frames + write the YOLO dataset (window-blocked splits)."""
    from solarflare.detect.dataset import build_yolo_dataset

    cfg = _load(config)
    email = email or os.environ.get("JSOC_EMAIL", "")
    if not email:
        typer.secho("JSOC email required (set JSOC_EMAIL or --email)", fg="red")
        raise typer.Exit(code=1)
    summary = build_yolo_dataset(cfg, email, out)
    typer.echo(summary.to_string(index=False))
    typer.echo(f"dataset: {out / 'dataset.yaml'}")


@app.command("train-detect")
def train_detect(
    config: ConfigOpt = DEFAULT_CONFIG,
    dataset: Annotated[Path, typer.Option(help="dataset.yaml path")] = Path(
        "data/detect_dataset/dataset.yaml"),
    epochs: Annotated[int | None, typer.Option(help="Override config epochs")] = None,
    device: Annotated[str | None, typer.Option(help="cpu / 0 / mps (default: auto)")] = None,
) -> None:
    """Fine-tune pretrained YOLO on the bootstrapped dataset."""
    from solarflare.detect.yolo import train_detector

    cfg = _load(config)
    weights = train_detector(cfg, dataset, Path(cfg.paths.outputs_dir) / "detect",
                             epochs=epochs, device=device)
    typer.echo(f"weights: {weights}")


@app.command("eval-detect")
def eval_detect(
    weights: Annotated[Path, typer.Option(exists=True)],
    config: ConfigOpt = DEFAULT_CONFIG,
    dataset_root: Annotated[Path, typer.Option()] = Path("data/detect_dataset"),
    split: Annotated[str, typer.Option(help="val | test")] = "test",
    conf: Annotated[float, typer.Option()] = 0.25,
    iou_match: Annotated[float, typer.Option(help="IoU threshold for a match")] = 0.5,
) -> None:
    """Gate G2 report: detector IoU/recall/precision vs SHARP-derived boxes."""
    from solarflare.detect.dataset import load_yolo_labels
    from solarflare.detect.yolo import evaluate_vs_truth, predict_boxes

    cfg = _load(config)
    images = sorted((dataset_root / "images" / split).glob("*.png"))
    if not images:
        typer.secho(f"no images under {dataset_root}/images/{split}", fg="red")
        raise typer.Exit(code=1)
    truth = load_yolo_labels(dataset_root, split)
    preds = predict_boxes(weights, images, conf=conf)
    metrics = evaluate_vs_truth(preds, truth, iou_match=iou_match)
    for key, value in metrics.items():
        typer.echo(f"{key:>22}: {value:.4f}" if isinstance(value, float)
                   else f"{key:>22}: {value}")
    append_experiment_row(
        cfg.paths.experiment_log,
        {"phase": "P2", "experiment": f"detect_eval_{split}", "weights": str(weights),
         "config_hash": cfg.short_hash(), **metrics},
    )


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
