"""HARP-map renderer: color stability, disk masking, frame composition."""

import numpy as np
import pandas as pd
import pytest

from solarflare.viz.harpmap import (
    PALETTE,
    compose_harpmap_frame,
    harp_color,
    render_harp_disk,
)


def test_harp_color_deterministic():
    assert harp_color(13617) == harp_color(13617)
    assert harp_color(0) == PALETTE[0]
    assert all(len(harp_color(h)) == 3 for h in range(50))


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
    data[70:80, 60:75] = 900.0  # one strong-field region
    header = sunpy.map.make_fitswcs_header(data, center, scale=[16, 16] * u.arcsec / u.pix)
    return sunpy.map.Map(data, header)


def test_render_harp_disk_colors(fulldisk_map):
    boxes = pd.DataFrame(
        [{"harpnum": 7, "noaa_ar": 90001, "x_min": 58, "x_max": 77, "y_min": 68, "y_max": 82}]
    )
    rgb = render_harp_disk(fulldisk_map, boxes, size=256, spot_threshold=2000.0)
    assert rgb.shape == (256, 256, 3)
    # corners off-disk: white background
    assert (rgb[2, 2] == 255).all()
    # the AR pixels carry the HARP color (image y is flipped vs FITS y)
    assert (rgb == np.array(harp_color(7), dtype=np.uint8)).all(axis=-1).any()


def test_compose_harpmap_frame_layout(fulldisk_map):
    boxes = pd.DataFrame(
        [
            {"harpnum": 7, "noaa_ar": 90001, "x_min": 58, "x_max": 77, "y_min": 68, "y_max": 82},
            {"harpnum": 8, "noaa_ar": 0, "x_min": 30, "x_max": 45, "y_min": 30, "y_max": 45},
        ]
    )
    frame = compose_harpmap_frame(
        fulldisk_map, boxes, pd.Timestamp("2011-02-14 12:00"), size=256, legend_width=120
    )
    assert frame.dtype == np.uint8
    assert frame.shape == (256, 256 + 120, 3)  # disk + side legend
