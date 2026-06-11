"""Preprocessing units: time matching, normalization, QA, and the key
synthetic-map test that HPC->CEA reprojection lands a feature where WCS says."""

import numpy as np
import pandas as pd
import pytest

from solarflare.data.preprocess import (
    exposure_normalize,
    frame_qa,
    match_nearest,
    reproject_to_target,
    robust_stats,
)


class TestMatchNearest:
    def test_basic(self):
        targets = pd.Series(pd.to_datetime(
            ["2099-01-01T00:00", "2099-01-01T00:12", "2099-01-01T00:24"]))
        cands = pd.Series(pd.to_datetime(
            ["2099-01-01T00:01", "2099-01-01T00:11", "2099-01-01T00:44"]))
        out = match_nearest(targets, cands, tolerance_seconds=360)
        assert list(out) == [0, 1, -1]  # third candidate is 20 min away

    def test_empty_candidates(self):
        targets = pd.Series(pd.to_datetime(["2099-01-01T00:00"]))
        out = match_nearest(targets, pd.Series([], dtype="datetime64[ns]"), 360)
        assert list(out) == [-1]

    def test_unsorted_candidates(self):
        targets = pd.Series(pd.to_datetime(["2099-01-01T01:00"]))
        cands = pd.Series(pd.to_datetime(["2099-01-01T02:00", "2099-01-01T00:59"]))
        assert list(match_nearest(targets, cands, 360)) == [1]


class TestExposureNormalize:
    def test_divides_by_exptime(self, make_hpc_map, make_cea_map):
        import astropy.units as u
        from sunpy.coordinates import Helioprojective

        cea = make_cea_map()
        center = cea.pixel_to_world(30 * u.pix, 20 * u.pix).transform_to(
            Helioprojective(observer=cea.observer_coordinate, obstime=cea.date))
        m = make_hpc_map(center, data=np.full((80, 80), 6.0, dtype=np.float32),
                         exptime_s=2.0)
        out = exposure_normalize(m)
        assert out.dtype == np.float32
        assert np.allclose(out, 3.0)

    def test_invalid_exptime_masks_frame(self, make_hpc_map, make_cea_map):
        """EXPTIME=0 (eclipse-season frames) must yield NaN, never raw DN
        (raw DN would silently mix units with DN/s in the same stack)."""
        import astropy.units as u
        from sunpy.coordinates import Helioprojective

        cea = make_cea_map()
        center = cea.pixel_to_world(30 * u.pix, 20 * u.pix).transform_to(
            Helioprojective(observer=cea.observer_coordinate, obstime=cea.date))
        m = make_hpc_map(center, data=np.full((80, 80), 6.0, dtype=np.float32),
                         exptime_s=0.0)
        out = exposure_normalize(m)
        assert np.isnan(out).all()
        assert out.dtype == np.float32


class TestFrameQA:
    def test_clean_frame(self):
        flags = frame_qa(np.ones((10, 10)), quality=0, max_nan_fraction=0.5)
        assert not flags["flagged"]

    def test_high_nan(self):
        data = np.full((10, 10), np.nan)
        data[:3] = 1.0
        flags = frame_qa(data, quality=0, max_nan_fraction=0.5)
        assert flags["high_nan"] and flags["flagged"]

    def test_bad_quality(self):
        flags = frame_qa(np.ones((4, 4)), quality=0x40, max_nan_fraction=0.5)
        assert flags["bad_quality"] and flags["flagged"]

    def test_low_coverage(self):
        flags = frame_qa(np.ones((4, 4)), quality=0, max_nan_fraction=0.5,
                         coverage=0.3, min_coverage=0.6)
        assert flags["low_coverage"] and flags["flagged"]


def test_robust_stats_ordering():
    rng = np.random.default_rng(0)
    stats = robust_stats(rng.normal(100.0, 10.0, size=(4, 50, 50)).astype(np.float32))
    assert stats["p01"] < stats["median"] < stats["p99"] <= stats["p999"]
    assert stats["finite_frac"] == 1.0


def test_reprojection_lands_blob_at_wcs_position(make_cea_map, make_hpc_map):
    """A Gaussian blob placed at a known world coordinate in a synthetic AIA-like
    HPC map must reproject onto the CEA grid at the pixel the CEA WCS predicts."""
    import astropy.units as u
    from sunpy.coordinates import Helioprojective

    cea = make_cea_map(shape=(40, 60))
    # world point slightly off the CEA patch centre
    blob_world = cea.pixel_to_world(36 * u.pix, 25 * u.pix)
    hpc_frame = Helioprojective(observer=cea.observer_coordinate, obstime=cea.date)
    blob_hpc = blob_world.transform_to(hpc_frame)

    # AIA-like map centred on the patch centre, blob injected at its WCS pixel.
    # Realistic 0.6"/pix keeps source and target pixel scales comparable, so the
    # bilinear-resampled peak cannot snap to a coarse source-grid node.
    center_hpc = cea.pixel_to_world(30 * u.pix, 20 * u.pix).transform_to(hpc_frame)
    aia = make_hpc_map(center_hpc, shape=(80, 80))
    x0, y0 = aia.wcs.world_to_pixel(blob_hpc)
    yy, xx = np.mgrid[0:80, 0:80]
    data = np.exp(-(((xx - x0) ** 2 + (yy - y0) ** 2) / (2 * 3.0 ** 2))).astype(np.float32)
    aia = make_hpc_map(center_hpc, shape=(80, 80), data=data)

    out, coverage = reproject_to_target(aia, cea)
    assert out.shape == cea.data.shape
    assert coverage > 0.5

    expected_x, expected_y = cea.wcs.world_to_pixel(blob_world)
    got_y, got_x = np.unravel_index(np.nanargmax(out), out.shape)
    assert got_x == pytest.approx(float(expected_x), abs=1.5)
    assert got_y == pytest.approx(float(expected_y), abs=1.5)
