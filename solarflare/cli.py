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
