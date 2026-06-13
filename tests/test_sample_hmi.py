"""_load_hmi_stacks aligns SHARP segments by T_REC and drops incomplete timestamps."""

import astropy.units as u
import numpy as np
import pytest
import sunpy.map
from astropy.coordinates import SkyCoord
from sunpy.coordinates import HeliographicStonyhurst

from solarflare.data.sample import _load_hmi_stacks


def _save_sharp(tmp_path, obstime: str, seg: str, value: float):
    """Write a tiny CEA SHARP-like FITS for one segment at a given obstime."""
    ref = SkyCoord(0 * u.deg, 0 * u.deg, frame=HeliographicStonyhurst(obstime=obstime))
    header = sunpy.map.make_fitswcs_header(
        np.zeros((4, 5), dtype=np.float32),
        ref,
        scale=[0.03, 0.03] * u.deg / u.pix,
        projection_code="CEA",
    )
    m = sunpy.map.Map(np.full((4, 5), value, dtype=np.float32), header)
    stamp = obstime.replace("-", "").replace(":", "").replace("T", "_")
    path = tmp_path / f"hmi.sharp_cea_720s.7115.{stamp}_TAI.{seg}.fits"
    m.save(str(path), overwrite=True)
    return path


def test_load_hmi_stacks_aligns_and_drops_gap(tmp_path):
    times = ["2017-09-03T00:00:00", "2017-09-03T01:00:00", "2017-09-03T02:00:00"]
    files = []
    for i, t in enumerate(times):
        files.append(_save_sharp(tmp_path, t, "magnetogram", float(i)))
        if i != 1:  # middle timestamp is missing its continuum segment
            files.append(_save_sharp(tmp_path, t, "continuum", float(i)))

    stacks, anchor_maps, frame_times = _load_hmi_stacks(files, ["magnetogram", "continuum"])

    # only the two complete timestamps survive, and both stacks agree in length
    assert stacks["hmi_magnetogram"].shape[0] == 2
    assert stacks["hmi_continuum"].shape[0] == 2
    assert len(anchor_maps) == 2
    assert list(frame_times["time_utc"].dt.hour) == [0, 2]
    # per-frame segments stay paired (value encodes the original timestamp index)
    np.testing.assert_array_equal(
        stacks["hmi_magnetogram"][:, 0, 0], stacks["hmi_continuum"][:, 0, 0]
    )


def test_load_hmi_stacks_no_common_raises(tmp_path):
    files = [
        _save_sharp(tmp_path, "2017-09-03T00:00:00", "magnetogram", 0.0),
        _save_sharp(tmp_path, "2017-09-03T01:00:00", "continuum", 1.0),
    ]
    with pytest.raises(RuntimeError, match="common to all HMI segments"):
        _load_hmi_stacks(files, ["magnetogram", "continuum"])
