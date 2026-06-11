"""JSOC query construction and cutout-box geometry (offline; no exports)."""

from datetime import datetime

import astropy.units as u
import pytest

from solarflare.data.jsoc_fetch import (
    aia_series_for_channel,
    build_sharp_ds,
    cutout_corners_from_map,
    tai_str,
)


def test_tai_str():
    assert tai_str(datetime(2011, 2, 14, 0, 0, 0)) == "2011.02.14_00:00:00_TAI"


def test_build_sharp_ds():
    ds = build_sharp_ds(
        "hmi.sharp_cea_720s", 377,
        datetime(2011, 2, 14), datetime(2011, 2, 15, 12),
        ["magnetogram", "continuum"],
    )
    assert ds == (
        "hmi.sharp_cea_720s[377]"
        "[2011.02.14_00:00:00_TAI-2011.02.15_12:00:00_TAI]"
        "{magnetogram,continuum}"
    )


def test_aia_series_routing():
    assert aia_series_for_channel(171, "euv", "uv") == "euv"
    assert aia_series_for_channel(94, "euv", "uv") == "euv"
    assert aia_series_for_channel(1600, "euv", "uv") == "uv"
    assert aia_series_for_channel(1700, "euv", "uv") == "uv"
    with pytest.raises(ValueError):
        aia_series_for_channel(9999, "euv", "uv")


def test_cutout_corners_enclose_patch(make_cea_map):
    cea = make_cea_map()
    bl, tr = cutout_corners_from_map(cea, pad_arcsec=0.0)
    assert bl.Tx < tr.Tx and bl.Ty < tr.Ty

    bl_pad, tr_pad = cutout_corners_from_map(cea, pad_arcsec=50.0)
    # padding must widen the box by 2*pad on each axis
    width = (tr.Tx - bl.Tx).to_value(u.arcsec)
    width_pad = (tr_pad.Tx - bl_pad.Tx).to_value(u.arcsec)
    assert width_pad == pytest.approx(width + 100.0, abs=1.0)
    height = (tr.Ty - bl.Ty).to_value(u.arcsec)
    height_pad = (tr_pad.Ty - bl_pad.Ty).to_value(u.arcsec)
    assert height_pad == pytest.approx(height + 100.0, abs=1.0)
