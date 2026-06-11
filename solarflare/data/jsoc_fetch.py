"""JSOC fetchers: SHARP CEA cutouts via drms exports, AIA cutouts via Fido + im_patch.

Both paths require a JSOC-registered notify email (env JSOC_EMAIL). The AIA
cutout request uses `a.jsoc.Cutout(..., tracking=True)` so the box follows the
AR with solar rotation, mirroring how the HARP patch tracks it on the HMI side.
Downloads are idempotent: existing non-empty target directories are reused.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

log = logging.getLogger(__name__)

EUV_CHANNELS = {94, 131, 171, 193, 211, 304, 335}
UV_CHANNELS = {1600, 1700}


def tai_str(t: datetime) -> str:
    """Format a datetime as a JSOC TAI record-set time, e.g. 2011.02.14_00:00:00_TAI."""
    return f"{t:%Y.%m.%d_%H:%M:%S}_TAI"


def build_sharp_ds(
    series: str, harp: int, start: datetime, end: datetime, segments: list[str]
) -> str:
    """Record-set query for SHARP segments, e.g.
    hmi.sharp_cea_720s[377][2011.02.14_00:00:00_TAI-2011.02.15_12:00:00_TAI]{magnetogram,continuum}
    """
    return f"{series}[{harp}][{tai_str(start)}-{tai_str(end)}]{{{','.join(segments)}}}"


def aia_series_for_channel(channel: int, euv_series: str, uv_series: str) -> str:
    if channel in EUV_CHANNELS:
        return euv_series
    if channel in UV_CHANNELS:
        return uv_series
    raise ValueError(f"AIA channel {channel} is neither EUV {sorted(EUV_CHANNELS)} "
                     f"nor UV {sorted(UV_CHANNELS)}")


def _existing_fits(out_dir: Path) -> list[Path]:
    return sorted(p for p in out_dir.glob("*.fits") if p.stat().st_size > 0)


def fetch_sharp_cutouts(
    series: str,
    harp: int,
    start: datetime,
    end: datetime,
    segments: list[str],
    email: str,
    out_dir: str | Path,
    overwrite: bool = False,
) -> list[Path]:
    """Export + download SHARP CEA segment FITS files via drms. Returns sorted paths."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if not overwrite:
        existing = _existing_fits(out_dir)
        if existing:
            log.info("reusing %d existing SHARP files in %s", len(existing), out_dir)
            return existing

    import drms

    ds = build_sharp_ds(series, harp, start, end, segments)
    log.info("JSOC export: %s", ds)
    client = drms.Client()
    request = client.export(ds, method="url", protocol="fits", email=email)
    request.wait()
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
    top_right = SkyCoord(
        (np.max(tx) + pad) * u.arcsec, (np.max(ty) + pad) * u.arcsec, frame=frame
    )
    return bottom_left, top_right


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
    """Request rotation-tracked AIA cutouts from JSOC via Fido. Returns sorted paths."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if not overwrite:
        existing = _existing_fits(out_dir)
        if existing:
            log.info("reusing %d existing AIA %d files in %s", len(existing), channel, out_dir)
            return existing

    import astropy.units as u
    from sunpy.net import Fido
    from sunpy.net import attrs as a

    series = aia_series_for_channel(channel, euv_series, uv_series)
    log.info("JSOC AIA cutout request: %s, %d A, %s .. %s @ %ds",
             series, channel, start, end, cadence_seconds)
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
