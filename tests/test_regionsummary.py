"""Solar Region Summary renderer: risk mapping, location formatting, frame compose."""

import numpy as np
import pandas as pd
import pytest

from solarflare.viz.regionsummary import (
    NO_DATA_COLOR,
    compose_region_summary_frame,
    format_stonyhurst,
    region_rows,
    render_magnetogram_disk,
    risk_level,
)


def test_risk_level_bands():
    assert risk_level(0.0)[0] == "Low"
    assert risk_level(0.09)[0] == "Low"
    assert risk_level(0.10)[0] == "Moderate"
    assert risk_level(0.24)[0] == "Moderate"
    assert risk_level(0.25)[0] == "High"
    assert risk_level(0.49)[0] == "High"
    assert risk_level(0.50)[0] == "Very High"
    assert risk_level(1.0)[0] == "Very High"


def test_format_stonyhurst():
    assert format_stonyhurst(20.0, 15.0) == "N15W20"
    assert format_stonyhurst(-12.0, -8.0) == "S08E12"
    assert format_stonyhurst(np.nan, 5.0) == "--"
    assert format_stonyhurst(None, None) == "--"


def test_region_rows_sorted_by_risk_no_data_last():
    boxes = pd.DataFrame(
        [
            {
                "harpnum": 11,
                "noaa_ar": 90001,
                "lon_min": 10,
                "lon_max": 30,
                "lat_min": 5,
                "lat_max": 20,
                "lon_fwt": 20.0,
                "lat_fwt": 12.0,
            },
            {
                "harpnum": 22,
                "noaa_ar": 0,
                "lon_min": -50,
                "lon_max": -35,
                "lat_min": -20,
                "lat_max": -8,
                "lon_fwt": -42.0,
                "lat_fwt": -14.0,
            },
        ]
    )
    rows = region_rows(boxes, probs={11: 0.7})
    assert [r["harp"] for r in rows] == [11, 22]  # 22 has no prob -> last
    assert rows[0]["risk"] == "Very High" and rows[0]["loc"] == "N12W20"
    assert rows[1]["p"] is None and rows[1]["color"] == NO_DATA_COLOR


@pytest.fixture()
def fulldisk_map():
    import astropy.units as u
    import sunpy.map
    from astropy.coordinates import SkyCoord
    from sunpy.coordinates import Helioprojective, get_earth

    obstime = "2024-05-10T12:00:00"
    center = SkyCoord(
        0 * u.arcsec,
        0 * u.arcsec,
        frame=Helioprojective(observer=get_earth(obstime), obstime=obstime),
    )
    data = np.zeros((128, 128), dtype=np.float32)
    data[70:80, 60:75] = 800.0  # a strong-field region
    header = sunpy.map.make_fitswcs_header(data, center, scale=[16, 16] * u.arcsec / u.pix)
    return sunpy.map.Map(data, header)


def test_render_magnetogram_disk_shape_and_offdisk(fulldisk_map):
    rgb = render_magnetogram_disk(fulldisk_map, size=256)
    assert rgb.shape == (256, 256, 3) and rgb.dtype == np.uint8
    assert rgb[2, 2].max() <= 25  # off-disk corner is near-black
    assert rgb[128, 128].max() > 100  # disk centre is mid-gray


def test_compose_region_summary_frame(fulldisk_map):
    boxes = pd.DataFrame(
        [
            {
                "harpnum": 11,
                "noaa_ar": 90001,
                "x_min": 55,
                "x_max": 80,
                "y_min": 65,
                "y_max": 85,
                "lon_min": 10,
                "lon_max": 30,
                "lat_min": 5,
                "lat_max": 20,
                "lon_fwt": 20.0,
                "lat_fwt": 12.0,
            },
        ]
    )
    frame = compose_region_summary_frame(
        fulldisk_map,
        boxes,
        probs={11: 0.72},
        time_utc=pd.Timestamp("2024-05-10 12:00"),
        model_note="model: test",
        size=256,
        panel_width=380,
    )
    assert frame.dtype == np.uint8
    assert frame.shape[1] == 256 + 380  # disk + summary panel
    assert frame.shape[0] == 256 + 52 + 30  # banner + disk + footer
