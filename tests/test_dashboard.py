"""Dashboard renderer: band mapping and frame composition on a synthetic disk."""

import numpy as np
import pandas as pd
import pytest

from solarflare.viz.dashboard import (
    BAND_COLORS,
    compose_dashboard_frame,
    probability_band,
    render_disk,
)


def test_probability_band_edges():
    assert probability_band(0.0) == "C"
    assert probability_band(0.24) == "C"
    assert probability_band(0.25) == "M"
    assert probability_band(0.59) == "M"
    assert probability_band(0.6) == "X"
    assert probability_band(1.0) == "X"
    assert set(BAND_COLORS) == {"C", "M", "X"}


@pytest.fixture()
def fulldisk_map():
    import astropy.units as u
    import sunpy.map
    from astropy.coordinates import SkyCoord
    from sunpy.coordinates import Helioprojective, get_earth

    obstime = "2011-02-14T12:00:00"
    center = SkyCoord(
        0 * u.arcsec,
        0 * u.arcsec,
        frame=Helioprojective(observer=get_earth(obstime), obstime=obstime),
    )
    data = np.zeros((128, 128), dtype=np.float32)
    data[70:80, 60:75] = 800.0  # a strong-field region
    header = sunpy.map.make_fitswcs_header(data, center, scale=[16, 16] * u.arcsec / u.pix)
    return sunpy.map.Map(data, header)


def test_render_disk_shape_and_offdisk(fulldisk_map):
    rgb = render_disk(fulldisk_map, size=256)
    assert rgb.shape == (256, 256, 3)
    assert rgb.dtype == np.uint8
    # corners are off-disk: near-black
    assert rgb[2, 2].max() <= 20
    # disk centre is bright
    assert rgb[128, 128].max() > 120


def test_compose_dashboard_frame(fulldisk_map):
    boxes = pd.DataFrame(
        [
            {"harpnum": 11, "noaa_ar": 90001, "x_min": 55, "x_max": 80, "y_min": 65, "y_max": 85},
            {"harpnum": 22, "noaa_ar": 0, "x_min": 30, "x_max": 45, "y_min": 30, "y_max": 45},
        ]
    )
    frame = compose_dashboard_frame(
        fulldisk_map,
        boxes,
        probs={11: 0.72},
        time_utc=pd.Timestamp("2011-02-14 12:00"),
        model_note="model: test",
        size=256,
    )
    assert frame.dtype == np.uint8
    assert frame.shape[1] == 256
    assert frame.shape[0] > 256  # header + footer attached
