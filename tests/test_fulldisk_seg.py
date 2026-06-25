"""Full-disk segmentation helpers and pipeline tests (offline)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Shared helper
# ---------------------------------------------------------------------------


def _write_fake_fits(path: Path, date_obs: str) -> None:
    """Minimal FITS file with a DATE-OBS header (no valid solar WCS needed here)."""
    from astropy.io import fits

    hdu = fits.PrimaryHDU(np.zeros((8, 8), dtype=np.float32))
    hdu.header["DATE-OBS"] = date_obs
    path.parent.mkdir(parents=True, exist_ok=True)
    hdu.writeto(path, overwrite=True)


def _synthetic_sample(tmp_path: Path):
    """Sample placed at the correct data_root layout for FullDiskSegmenter."""
    from solarflare.data.cache import load_sample, write_sample

    n, h, w = 2, 16, 16
    blos = np.zeros((n, h, w), dtype=np.float32)
    blos[:, 5:10, 5:10] = 300.0
    times = pd.DataFrame(
        {
            "frame_idx": range(n),
            "time_utc": pd.date_range("2099-01-01", periods=n, freq="1h"),
        }
    )
    data_root = tmp_path / "data"
    sample_dir = write_sample(
        data_root / "cache" / "samples" / "harp00001_synthetic",
        {"hmi_continuum": np.full((n, h, w), 1000.0, dtype=np.float32), "hmi_magnetogram": blos},
        times,
        pd.DataFrame(),
        pd.DataFrame(),
        {"noaa": 99999, "harp": 1, "window": "synthetic"},
    )
    return load_sample(sample_dir)


# ---------------------------------------------------------------------------
# Unit tests — helpers
# ---------------------------------------------------------------------------


def test_ts_from_sharp_name_parses_trec():
    from solarflare.detect.fulldisk_seg import _ts_from_sharp_name

    p = Path("hmi.sharp_cea_720s.6670.20160715_000000_TAI.magnetogram.fits")
    assert _ts_from_sharp_name(p) == pd.Timestamp("2016-07-15 00:00:00")


def test_ts_from_sharp_name_returns_none_for_unknown():
    from solarflare.detect.fulldisk_seg import _ts_from_sharp_name

    assert _ts_from_sharp_name(Path("other_format.fits")) is None


def test_nearest_returns_closest():
    from solarflare.detect.fulldisk_seg import _nearest

    index = [
        (pd.Timestamp("2020-01-01 00:00"), Path("a.fits")),
        (pd.Timestamp("2020-01-01 06:00"), Path("b.fits")),
        (pd.Timestamp("2020-01-01 12:00"), Path("c.fits")),
    ]
    _, p = _nearest(index, pd.Timestamp("2020-01-01 07:00"))
    assert p.name == "b.fits"

    _, p2 = _nearest(index, pd.Timestamp("2020-01-01 10:00"))
    assert p2.name == "c.fits"


def test_build_fulldisk_index_raises_on_empty_dir(tmp_path):
    from solarflare.detect.fulldisk_seg import _build_fulldisk_index

    empty = tmp_path / "fulldisk"
    empty.mkdir()
    with pytest.raises(RuntimeError, match="fetch-fulldisk"):
        _build_fulldisk_index(empty)


def test_build_fulldisk_index_sorted(tmp_path):
    from solarflare.detect.fulldisk_seg import _build_fulldisk_index

    fd_dir = tmp_path / "fulldisk"
    _write_fake_fits(fd_dir / "frame_b.fits", "2020-01-02T00:00:00")
    _write_fake_fits(fd_dir / "frame_a.fits", "2020-01-01T00:00:00")
    index = _build_fulldisk_index(fd_dir)
    assert index[0][0] < index[1][0]
    assert index[0][0] == pd.Timestamp("2020-01-01")


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def test_fulldisk_segmenter_registered():
    from solarflare.config import SegmentConfig
    from solarflare.detect.segmenter import FullDiskSegmenter, available_segmenters, get_segmenter

    assert "fulldisk" in available_segmenters()
    seg = get_segmenter(SegmentConfig(model="fulldisk"))
    assert isinstance(seg, FullDiskSegmenter)
    assert seg.name == "fulldisk"


def test_segment_raises_without_fulldisk_dir(tmp_path):
    from solarflare.config import SegmentConfig
    from solarflare.detect.segmenter import get_segmenter

    sample = _synthetic_sample(tmp_path)
    seg = get_segmenter(SegmentConfig(model="fulldisk"))
    with pytest.raises(RuntimeError, match="fetch-fulldisk"):
        seg.segment_sample(sample)


# ---------------------------------------------------------------------------
# End-to-end with monkeypatched WCS reproject
# ---------------------------------------------------------------------------


def test_segment_fulldisk_end_to_end(tmp_path, monkeypatch):
    """Full pipeline with sunpy.map.Map and _reproject_mask monkeypatched."""
    import sunpy.map

    import solarflare.detect.fulldisk_seg as fdseg
    from solarflare.config import SegmentConfig
    from solarflare.detect.segmenter import get_segmenter

    sample = _synthetic_sample(tmp_path)
    data_root = tmp_path / "data"

    # Fake full-disk FITS
    fd_dir = data_root / "raw" / "fulldisk" / "synthetic"
    _write_fake_fits(
        fd_dir / "hmi.m_720s.20990101_000000_TAI.magnetogram.fits",
        "2099-01-01T00:00:00",
    )

    # Fake SHARP magnetogram FITS (name matches the T_REC parser)
    sharp_dir = data_root / "raw" / "synthetic" / "sharp_1"
    for hhmm in ("000000", "010000"):
        _write_fake_fits(
            sharp_dir / f"hmi.sharp_cea_720s.1.20990101_{hhmm}_TAI.magnetogram.fits",
            f"2099-01-01T{hhmm[:2]}:{hhmm[2:4]}:00",
        )

    # Monkeypatch sunpy.map.Map to avoid real solar-WCS loading
    class _FakeMap:
        def __init__(self, *args, **kwargs):
            self.data = np.zeros((8, 8), dtype=np.float32)
            self.meta = {"DATE-OBS": "2099-01-01T00:00:00"}
            self.wcs = None

    monkeypatch.setattr(sunpy.map, "Map", _FakeMap)

    # Monkeypatch _reproject_mask to return a known synthetic mask
    reproject_calls: list[int] = []

    def _fake_reproject(full_mask, fd_map, harp_map, target_shape):
        reproject_calls.append(1)
        out = np.zeros(target_shape, dtype=np.uint8)
        out[2:5, 2:5] = 1  # 9 AR pixels
        return out

    monkeypatch.setattr(fdseg, "_reproject_mask", _fake_reproject)

    seg = get_segmenter(SegmentConfig(model="fulldisk"))
    masks_path, areas = seg.segment_sample(sample)

    masks = np.load(masks_path)
    assert masks.shape == (2, 16, 16), "wrong mask shape"
    assert masks.dtype == np.uint8
    assert masks[0, 3, 3] == 1, "expected AR pixel from fake reproject"
    assert areas["ar_pixels"].eq(9).all(), "expected 9 AR pixels per frame"
    assert len(reproject_calls) == 2, "_reproject_mask must be called once per frame"
