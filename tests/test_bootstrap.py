"""HARP keyword tidying and heliographic->pixel box projection (offline)."""

import numpy as np
import pandas as pd
import pytest

from solarflare.detect.bootstrap import boxes_to_pixels, parse_tai, tidy_harp_keywords


def test_parse_tai():
    s = pd.Series(["2011.02.15_02:00:00_TAI", "2024.05.14_16:48:00_TAI"])
    out = parse_tai(s)
    assert out.iloc[0] == pd.Timestamp("2011-02-15T02:00:00")
    assert out.iloc[1] == pd.Timestamp("2024-05-14T16:48:00")


def _raw_keywords() -> pd.DataFrame:
    return pd.DataFrame({
        "HARPNUM": [377, 378, 379],
        "T_REC": ["2011.02.15_02:00:00_TAI"] * 3,
        "NOAA_AR": [11158, 0, np.nan],
        "NOAA_ARS": ["11158", "", ""],
        "LON_MIN": [3.87, np.nan, 10.0],
        "LON_MAX": [22.32, 5.0, 9.0],     # 379: degenerate (max < min)
        "LAT_MIN": [-25.24, -10.0, 0.0],
        "LAT_MAX": [-15.63, -5.0, 5.0],
        "LON_FWT": [12.78, np.nan, np.nan],
        "LAT_FWT": [-20.20, np.nan, np.nan],
        "OMEGA_DT": [13.34, np.nan, np.nan],
    })


def test_tidy_drops_bad_rows():
    out = tidy_harp_keywords(_raw_keywords())
    assert list(out["harpnum"]) == [377]          # NaN lon and degenerate box dropped
    assert out.iloc[0]["noaa_ar"] == 11158
    assert out.iloc[0]["lon_max"] == pytest.approx(22.32)


def test_boxes_to_pixels_round_trip(make_cea_map):
    """Project a small Stonyhurst box onto a synthetic full-disk-like HPC map and
    verify the box center maps back to the same heliographic point."""
    import astropy.units as u
    import sunpy.map
    from astropy.coordinates import SkyCoord
    from sunpy.coordinates import HeliographicStonyhurst, Helioprojective, get_earth

    obstime = "2011-02-14T12:00:00"
    observer = get_earth(obstime)
    center = SkyCoord(0 * u.arcsec, 0 * u.arcsec,
                      frame=Helioprojective(observer=observer, obstime=obstime))
    header = sunpy.map.make_fitswcs_header(
        np.zeros((256, 256), dtype=np.float32), center,
        scale=[8, 8] * u.arcsec / u.pix,
    )
    fulldisk = sunpy.map.Map(np.zeros((256, 256), dtype=np.float32), header)

    boxes = pd.DataFrame({
        "time": [pd.Timestamp(obstime)],
        "harpnum": [377],
        "lon_min": [5.0], "lon_max": [20.0],
        "lat_min": [-25.0], "lat_max": [-15.0],
    })
    px = boxes_to_pixels(boxes, fulldisk)
    row = px.iloc[0]
    assert row["x_max"] > row["x_min"] and row["y_max"] > row["y_min"]
    # box center pixel -> world -> HGS should be inside the heliographic box
    cx, cy = 0.5 * (row["x_min"] + row["x_max"]), 0.5 * (row["y_min"] + row["y_max"])
    world = fulldisk.pixel_to_world(cx * u.pix, cy * u.pix)
    hgs = world.transform_to(HeliographicStonyhurst(obstime=obstime))
    assert 5.0 < hgs.lon.deg < 20.0
    assert -25.0 < hgs.lat.deg < -15.0
    # west (positive lon) must map to larger x on a non-rotated map
    boxes_east = boxes.assign(lon_min=-20.0, lon_max=-5.0)
    px_east = boxes_to_pixels(boxes_east, fulldisk)
    assert px_east.iloc[0]["x_max"] < row["x_min"]


def test_boxes_to_pixels_empty(make_cea_map):
    import astropy.units as u
    import sunpy.map
    from astropy.coordinates import SkyCoord
    from sunpy.coordinates import Helioprojective, get_earth

    obstime = "2011-02-14T12:00:00"
    center = SkyCoord(0 * u.arcsec, 0 * u.arcsec,
                      frame=Helioprojective(observer=get_earth(obstime), obstime=obstime))
    header = sunpy.map.make_fitswcs_header(np.zeros((64, 64), dtype=np.float32), center,
                                           scale=[32, 32] * u.arcsec / u.pix)
    fulldisk = sunpy.map.Map(np.zeros((64, 64), dtype=np.float32), header)
    out = boxes_to_pixels(pd.DataFrame(columns=["lon_min", "lon_max",
                                                "lat_min", "lat_max"]), fulldisk)
    assert out.empty and "x_min" in out.columns
