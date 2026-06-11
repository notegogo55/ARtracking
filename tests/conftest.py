"""Shared fixtures. All tests run offline — no network, no live downloads (CI-safe).

Synthetic sunpy maps (factories below) carry a deliberately unrealistic
obstime/geometry: they test WCS math, not solar physics.
"""

from pathlib import Path

import numpy as np
import pytest

FIXTURES_DIR = Path(__file__).parent / "fixtures"
REPO_ROOT = Path(__file__).parent.parent

SYNTH_OBSTIME = "2011-02-14T12:00:00"


def _earth_observer():
    from sunpy.coordinates import get_earth

    return get_earth(SYNTH_OBSTIME)


@pytest.fixture(scope="session")
def make_cea_map():
    """Factory for a synthetic SHARP-like CEA map in Heliographic Carrington."""

    def _make(shape=(40, 60), lon_deg=None, lat_deg=15.0, scale_deg=0.03, data=None):
        import astropy.units as u
        import sunpy.map
        from astropy.coordinates import SkyCoord
        from sunpy.coordinates import HeliographicCarrington
        from sunpy.coordinates.sun import L0

        observer = _earth_observer()
        if lon_deg is None:
            # near disk centre for the synthetic obstime, so the patch is front-side
            lon_deg = float(L0(SYNTH_OBSTIME).to_value(u.deg))
        ref = SkyCoord(
            lon_deg * u.deg, lat_deg * u.deg,
            frame=HeliographicCarrington(observer=observer, obstime=SYNTH_OBSTIME),
        )
        header = sunpy.map.make_fitswcs_header(
            np.zeros(shape, dtype=np.float32), ref,
            scale=[scale_deg, scale_deg] * u.deg / u.pix, projection_code="CEA",
        )
        if data is None:
            data = np.zeros(shape, dtype=np.float32)
        return sunpy.map.Map(data, header)

    return _make


@pytest.fixture(scope="session")
def make_hpc_map():
    """Factory for a synthetic AIA-like helioprojective map (TAN projection)."""

    def _make(center, shape=(80, 80), scale_arcsec=0.6, data=None, exptime_s=2.0):
        import astropy.units as u
        import sunpy.map

        header = sunpy.map.make_fitswcs_header(
            np.zeros(shape, dtype=np.float32), center,
            scale=[scale_arcsec, scale_arcsec] * u.arcsec / u.pix,
        )
        header["exptime"] = exptime_s
        if data is None:
            data = np.zeros(shape, dtype=np.float32)
        return sunpy.map.Map(data, header)

    return _make


@pytest.fixture()
def sample_events_csv() -> Path:
    """Synthetic GOES event list (year 2099 — deliberately not real data)."""
    return FIXTURES_DIR / "goes_events_sample.csv"


@pytest.fixture()
def default_config_path() -> Path:
    return REPO_ROOT / "configs" / "default.yaml"
