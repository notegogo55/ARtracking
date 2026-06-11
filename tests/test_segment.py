"""Threshold+morphology segmentation on synthetic patches (offline)."""

import numpy as np

from solarflare.config import SegmentConfig
from solarflare.detect.segment import active_mask, ar_mask, sunspot_mask


def _disk_indices(shape, cy, cx, r):
    yy, xx = np.ogrid[: shape[0], : shape[1]]
    return (yy - cy) ** 2 + (xx - cx) ** 2 <= r**2


def test_sunspot_mask_finds_dark_spot():
    cont = np.full((100, 120), 1000.0, dtype=np.float32)
    spot = _disk_indices(cont.shape, 50, 60, 8)
    cont[spot] = 500.0  # 50% of quiet level, well below 0.85 threshold
    mask = sunspot_mask(cont, spot_threshold=0.85, min_pixels=20, morph_radius=2)
    assert mask[50, 60]
    # mask should be confined to the spot's neighbourhood
    assert not mask[10, 10]
    assert 100 < mask.sum() < 400  # pi*8^2 ~ 200 px


def test_sunspot_mask_removes_specks():
    cont = np.full((100, 120), 1000.0, dtype=np.float32)
    cont[5, 5] = 100.0  # single dark pixel: must be cleaned away
    mask = sunspot_mask(cont, spot_threshold=0.85, min_pixels=20, morph_radius=2)
    assert mask.sum() == 0


def test_sunspot_mask_handles_all_nan():
    cont = np.full((50, 50), np.nan, dtype=np.float32)
    assert sunspot_mask(cont).sum() == 0


def test_active_mask_polarity_blind():
    blos = np.zeros((100, 120), dtype=np.float32)
    blos[_disk_indices(blos.shape, 30, 40, 7)] = 800.0
    blos[_disk_indices(blos.shape, 70, 80, 7)] = -800.0
    mask = active_mask(blos, bfield_threshold=100, min_pixels=20, morph_radius=2)
    assert mask[30, 40] and mask[70, 80]
    assert not mask[5, 110]


def test_ar_mask_is_union():
    cfg = SegmentConfig(min_region_pixels=20)
    cont = np.full((100, 120), 1000.0, dtype=np.float32)
    cont[_disk_indices(cont.shape, 20, 20, 7)] = 400.0       # spot only
    blos = np.zeros_like(cont)
    blos[_disk_indices(blos.shape, 80, 100, 7)] = 500.0      # field only
    mask = ar_mask(cont, blos, cfg)
    assert mask[20, 20] and mask[80, 100]
