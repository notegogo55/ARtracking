"""Pluggable Segmenter registry: resolution, stubs, and the threshold path (offline)."""

import numpy as np
import pandas as pd
import pytest

from solarflare.config import SegmentConfig
from solarflare.detect.segmenter import (
    SAM2Segmenter,
    SuryaSegmenter,
    ThresholdSegmenter,
    UNetSegmenter,
    available_segmenters,
    get_segmenter,
)


def test_registry_lists_all_four_models():
    assert available_segmenters() == ["sam2", "surya", "threshold", "unet"]


@pytest.mark.parametrize(
    ("model", "cls"),
    [
        ("threshold", ThresholdSegmenter),
        ("unet", UNetSegmenter),
        ("surya", SuryaSegmenter),
        ("sam2", SAM2Segmenter),
    ],
)
def test_get_segmenter_resolves_each_name(model, cls):
    seg = get_segmenter(SegmentConfig(model=model))
    assert isinstance(seg, cls)
    assert seg.name == model
    assert seg.cfg.model == model


def test_unknown_model_rejected_by_schema():
    # Literal field: an unknown model never reaches the registry.
    with pytest.raises(ValueError, match="model"):
        SegmentConfig(model="nope")


@pytest.mark.parametrize("model", ["surya", "sam2"])
def test_stub_segmenters_raise_with_guidance(model):
    seg = get_segmenter(SegmentConfig(model=model))
    with pytest.raises(NotImplementedError, match="segment.model"):
        seg.segment_sample(sample=None)


def _synthetic_sample(tmp_path):
    from solarflare.data.cache import load_sample, write_sample

    n, h, w = 4, 48, 48
    cont = np.full((n, h, w), 1000.0, dtype=np.float32)
    blos = np.zeros((n, h, w), dtype=np.float32)
    yy, xx = np.ogrid[:h, :w]
    spot = (yy - 24) ** 2 + (xx - 24) ** 2 <= 7**2
    for i in range(n):
        cont[i][spot] = 400.0
        blos[i][spot] = 600.0
    times = pd.DataFrame(
        {"frame_idx": range(n), "time_utc": pd.date_range("2099-01-01", periods=n, freq="1h")}
    )
    sample_dir = write_sample(
        tmp_path / "sample",
        {"hmi_continuum": cont, "hmi_magnetogram": blos},
        times,
        qa=pd.DataFrame(),
        labels=pd.DataFrame(),
        meta={"noaa": 99999, "harp": 1, "window": "synthetic"},
    )
    return load_sample(sample_dir)


def test_threshold_segmenter_writes_masks(tmp_path):
    sample = _synthetic_sample(tmp_path)
    seg = get_segmenter(SegmentConfig(model="threshold", min_region_pixels=20))
    masks_path, areas = seg.segment_sample(sample)
    masks = np.load(masks_path)
    assert masks.shape == (sample.n_frames, 48, 48) and masks.dtype == np.uint8
    assert areas["ar_pixels"].gt(0).all()  # the synthetic blob is always present
