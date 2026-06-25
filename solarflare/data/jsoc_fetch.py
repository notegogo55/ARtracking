"""JSOC fetchers: SHARP CEA cutouts via drms exports, AIA cutouts via Fido + im_patch.

Both paths require a JSOC-registered notify email (env JSOC_EMAIL). The AIA
cutout request uses `a.jsoc.Cutout(..., tracking=True)` so the box follows the
AR with solar rotation, mirroring how the HARP patch tracks it on the HMI side.
Downloads are idempotent: existing non-empty target directories are reused.
"""

from __future__ import annotations

import logging
import re
import time
from datetime import datetime
from pathlib import Path

log = logging.getLogger(__name__)

#: T_REC stamp in a SHARP filename, e.g. ...11149.20240510_000000_TAI.magnetogram.fits
_TREC_RE = re.compile(r"(\d{8})_(\d{6})_TAI")


def _trec_from_name(name: str) -> datetime | None:
    """Parse the nominal T_REC time from a SHARP filename (None for recnum-style)."""
    m = _TREC_RE.search(name)
    if m is None:
        return None
    try:
        return datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S")
    except ValueError:
        return None


EUV_CHANNELS = {94, 131, 171, 193, 211, 304, 335}
UV_CHANNELS = {1600, 1700}


def tai_str(t: datetime) -> str:
    """Format a datetime as a JSOC TAI record-set time, e.g. 2011.02.14_00:00:00_TAI."""
    return f"{t:%Y.%m.%d_%H:%M:%S}_TAI"


def build_sharp_ds(
    series: str,
    harp: int,
    start: datetime,
    end: datetime,
    segments: list[str],
    cadence_seconds: int | None = None,
) -> str:
    """Record-set query for SHARP segments, e.g.
    hmi.sharp_cea_720s[377][2011.02.14_00:00:00_TAI-2011.02.15_12:00:00_TAI@3600s]{magnetogram,continuum}

    cadence_seconds subsamples the native 720 s series (prime-key @step slice,
    same syntax the AIA query uses); None keeps every record. Hourly is plenty
    because the forecast features are resampled to 60 min downstream.
    """
    step = f"@{cadence_seconds}s" if cadence_seconds else ""
    return f"{series}[{harp}][{tai_str(start)}-{tai_str(end)}{step}]{{{','.join(segments)}}}"


def aia_series_for_channel(channel: int, euv_series: str, uv_series: str) -> str:
    if channel in EUV_CHANNELS:
        return euv_series
    if channel in UV_CHANNELS:
        return uv_series
    raise ValueError(
        f"AIA channel {channel} is neither EUV {sorted(EUV_CHANNELS)} nor UV {sorted(UV_CHANNELS)}"
    )


def _existing_fits(out_dir: Path) -> list[Path]:
    return sorted(p for p in out_dir.glob("*.fits") if p.stat().st_size > 0)


def _subsample_sharp_files(
    files: list[Path],
    start: datetime,
    end: datetime,
    cadence_seconds: int | None,
    segments: list[str],
    native_seconds: int = 720,
) -> list[Path]:
    """Thin a reused SHARP dir down to ~cadence_seconds spacing (no re-download).

    A raw dir fetched before the hourly-cadence switch holds every 720 s record;
    loading all of them blows up memory for a large patch (e.g. AR 12192 is
    800x1472 px -> 3.7 GiB per segment at 12 min). The step is derived from the
    actual file count vs the hours in the window, so it is a no-op when the dir
    is already hourly, and it works for both JSOC filename styles (recnum and
    T_REC) by grouping on the per-timestamp prefix (segment suffix stripped).
    """
    if not files:
        return files
    # Restrict a reused (possibly whole-window) dir to the requested [start, end]:
    # a sub-range fetch (e.g. one day of an already fully-staged window) must not
    # load the entire staged span. Best-effort -- only T_REC-style names carry a
    # parseable stamp; recnum-style names are kept. If nothing parses in range,
    # fall back to all files (never strand the loader with an empty list).
    within = [f for f in files if (t := _trec_from_name(Path(f).name)) is None or start <= t <= end]
    if within and len(within) != len(files):
        log.info(
            "restricting reused SHARP to %s..%s: %d -> %d files",
            start,
            end,
            len(files),
            len(within),
        )
        files = within
    if not cadence_seconds or cadence_seconds <= native_seconds:
        return files
    groups: dict[str, list[Path]] = {}
    for f in files:
        key = Path(f).name
        for seg in segments:
            key = key.replace(f".{seg}.fits", "")
        groups.setdefault(key, []).append(f)
    keys = sorted(groups)
    expected = max(1, round((end - start).total_seconds() / cadence_seconds))
    step = max(1, round(len(keys) / expected))
    if step == 1:
        return files
    kept = keys[::step]
    log.info(
        "subsampling reused SHARP: %d -> %d timestamps (~%ds cadence)",
        len(keys),
        len(kept),
        cadence_seconds,
    )
    return sorted(f for k in kept for f in groups[k])


def _export_with_retry(
    client, ds: str, export_kwargs: dict, max_retries: int = 6, pending_wait_s: int = 60
):
    """client.export(ds) + wait(), retried when JSOC reports its pending-export limit.

    Free JSOC accounts allow only one pending export at a time, so the back-to-back
    SHARP + per-channel AIA submits within one window (and across windows) transiently
    collide with 'has N pending export requests'. That is a queue race, not a real
    failure: wait for the slot and resubmit rather than aborting the whole window.
    """
    for attempt in range(max_retries + 1):
        try:
            request = client.export(ds, **export_kwargs)
            request.wait()
            return request
        except Exception as err:  # noqa: BLE001 - drms raises several types for this
            if "pending export" in str(err).lower() and attempt < max_retries:
                log.warning(
                    "JSOC export queue busy; waiting %ds then retry %d/%d (%s)",
                    pending_wait_s,
                    attempt + 1,
                    max_retries,
                    ds,
                )
                time.sleep(pending_wait_s)
                continue
            raise


#: Substrings that mark a retriable JSOC/network failure (not a permanent one).
_TRANSIENT_MARKERS = (
    "timed out",
    "timeout",
    "10060",
    "urlopen",
    "connection",
    "reset",
    "temporarily",
    "no aia",
    "produced no files",
    "returned no",
)


def fetch_with_retry(fn, *, attempts: int = 4, base_wait_s: int = 20, label: str = "fetch"):
    """Call fn(), retrying transient JSOC/network failures with linear backoff.

    JSOC cutout exports time out intermittently (WinError 10060) and a stressed
    search occasionally returns zero records; both are transient. Retry a few
    times (waiting base_wait_s, 2x, 3x, ...) before letting the caller fall back
    to an all-NaN channel. Permanent errors (anything not in _TRANSIENT_MARKERS)
    propagate immediately.
    """
    last: Exception | None = None
    for k in range(attempts):
        try:
            return fn()
        except Exception as err:  # noqa: BLE001 - many network/drms/Fido error types
            last = err
            if k < attempts - 1 and any(s in str(err).lower() for s in _TRANSIENT_MARKERS):
                wait = base_wait_s * (k + 1)
                log.warning(
                    "%s transient failure (%s); retry %d/%d in %ds",
                    label,
                    err,
                    k + 1,
                    attempts - 1,
                    wait,
                )
                time.sleep(wait)
                continue
            raise
    raise last  # pragma: no cover - loop either returns or raises


def fetch_sharp_cutouts(
    series: str,
    harp: int,
    start: datetime,
    end: datetime,
    segments: list[str],
    email: str,
    out_dir: str | Path,
    overwrite: bool = False,
    cadence_seconds: int | None = None,
) -> list[Path]:
    """Export + download SHARP CEA segment FITS files via drms. Returns sorted paths."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if not overwrite:
        existing = _existing_fits(out_dir)
        if existing:
            existing = _subsample_sharp_files(existing, start, end, cadence_seconds, segments)
            log.info("reusing %d existing SHARP files in %s", len(existing), out_dir)
            return existing

    import drms

    ds = build_sharp_ds(series, harp, start, end, segments, cadence_seconds)
    log.info("JSOC export: %s", ds)
    client = drms.Client()
    request = _export_with_retry(client, ds, {"method": "url", "protocol": "fits", "email": email})
    if not request.has_succeeded():
        raise RuntimeError(f"JSOC export failed for {ds!r}: status {request.status}")
    request.download(str(out_dir))
    files = _existing_fits(out_dir)
    log.info("downloaded %d SHARP files to %s", len(files), out_dir)
    if not files:
        raise RuntimeError(f"JSOC export for {ds!r} produced no files")
    return files


def cutout_corners_from_map(hmi_map, pad_arcsec: float = 40.0):
    """Helioprojective bottom-left/top-right corners that enclose a SHARP CEA patch.

    Transforms the four CEA patch corners to helioprojective coordinates as seen
    by the map's own observer (SDO) and pads the bounding box. Used as the
    `a.jsoc.Cutout` reference box at the map's time.
    """
    import astropy.units as u
    import numpy as np
    from astropy.coordinates import SkyCoord
    from sunpy.coordinates import Helioprojective

    ny, nx = hmi_map.data.shape
    corners_px = ([0, nx - 1, 0, nx - 1] * u.pix, [0, 0, ny - 1, ny - 1] * u.pix)
    corners_world = hmi_map.pixel_to_world(*corners_px)
    hpc = corners_world.transform_to(
        Helioprojective(observer=hmi_map.observer_coordinate, obstime=hmi_map.date)
    )
    tx = hpc.Tx.to_value(u.arcsec)
    ty = hpc.Ty.to_value(u.arcsec)
    pad = float(pad_arcsec)
    frame = Helioprojective(observer=hmi_map.observer_coordinate, obstime=hmi_map.date)
    bottom_left = SkyCoord(
        (np.min(tx) - pad) * u.arcsec, (np.min(ty) - pad) * u.arcsec, frame=frame
    )
    top_right = SkyCoord((np.max(tx) + pad) * u.arcsec, (np.max(ty) + pad) * u.arcsec, frame=frame)
    return bottom_left, top_right


def build_aia_uv_ds(
    series: str, channel: int, start: datetime, end: datetime, cadence_seconds: int
) -> str:
    """UV record set with WAVELNTH as a prime-key slice, e.g.
    aia.lev1_uv_24s[2012.03.04_00:00:00_TAI-2012.03.10_00:00:00_TAI@3600s][1700]{image}

    Needed because lev1_uv_24s interleaves 1600/1700 every 24 s: Fido's
    Sample+Wavelength filters the sampled T_REC slots and can return ZERO
    records when the cadence is a multiple of 48 s (phase-locked to the other
    wavelength). The native [time@step][wavelength] syntax samples WITHIN the
    wavelength slice (verified live 2026-06-12).
    """
    return f"{series}[{tai_str(start)}-{tai_str(end)}@{cadence_seconds}s][{channel}]{{image}}"


def _fetch_uv_via_drms(
    series: str,
    channel: int,
    start: datetime,
    end: datetime,
    cadence_seconds: int,
    bottom_left,
    top_right,
    email: str,
    out_dir: Path,
) -> list[Path]:
    """UV cutouts via a drms export with the im_patch payload sunpy would build."""
    import drms
    from sunpy.net.jsoc import attrs as ja

    cutout = ja.Cutout(bottom_left, top_right=top_right, tracking=True)
    ds = build_aia_uv_ds(series, channel, start, end, cadence_seconds)
    log.info("JSOC UV export: %s", ds)
    request = _export_with_retry(
        drms.Client(),
        ds,
        {
            "method": "url",
            "protocol": "fits",
            "email": email,
            "process": {"im_patch": cutout.value},
        },
    )
    if not request.has_succeeded():
        raise RuntimeError(f"JSOC UV export failed for {ds!r}")
    request.download(str(out_dir))
    return _existing_fits(out_dir)


def fetch_aia_cutouts(
    channel: int,
    start: datetime,
    end: datetime,
    cadence_seconds: int,
    bottom_left,
    top_right,
    email: str,
    out_dir: str | Path,
    euv_series: str = "aia.lev1_euv_12s",
    uv_series: str = "aia.lev1_uv_24s",
    overwrite: bool = False,
) -> list[Path]:
    """Request rotation-tracked AIA cutouts from JSOC. Returns sorted paths.

    EUV goes through Fido; UV (1600/1700) goes through a direct drms export
    (see build_aia_uv_ds for why).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if not overwrite:
        existing = _existing_fits(out_dir)
        if existing:
            log.info("reusing %d existing AIA %d files in %s", len(existing), channel, out_dir)
            return existing

    series = aia_series_for_channel(channel, euv_series, uv_series)
    if channel in UV_CHANNELS:
        paths = _fetch_uv_via_drms(
            series, channel, start, end, cadence_seconds, bottom_left, top_right, email, out_dir
        )
        log.info("downloaded %d AIA %d A files to %s", len(paths), channel, out_dir)
        if not paths:
            raise RuntimeError(f"AIA {channel} A UV export produced no files")
        return paths

    import astropy.units as u
    from sunpy.net import Fido
    from sunpy.net import attrs as a

    log.info(
        "JSOC AIA cutout request: %s, %d A, %s .. %s @ %ds",
        series,
        channel,
        start,
        end,
        cadence_seconds,
    )
    query = Fido.search(
        a.Time(start, end),
        a.Wavelength(channel * u.angstrom),
        a.Sample(cadence_seconds * u.s),
        a.jsoc.Series(series),
        a.jsoc.Segment("image"),
        a.jsoc.Notify(email),
        a.jsoc.Cutout(bottom_left, top_right=top_right, tracking=True),
    )
    n_records = sum(len(block) for block in query)
    if n_records == 0:
        raise RuntimeError(f"JSOC search returned no AIA {channel} A records in window")
    files = Fido.fetch(query, path=str(out_dir / "{file}"), progress=False)
    if files.errors:
        log.warning("AIA %d: %d fetch errors, retrying once", channel, len(files.errors))
        files = Fido.fetch(files, path=str(out_dir / "{file}"), progress=False)
    paths = _existing_fits(out_dir)
    log.info("downloaded %d AIA %d A files to %s", len(paths), channel, out_dir)
    if not paths:
        raise RuntimeError(f"AIA {channel} A fetch produced no files")
    return paths
