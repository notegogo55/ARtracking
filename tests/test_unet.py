"""U-Net segmentation upgrade: training + inference, fully offline (CPU, no downloads).

Uses a tiny synthetic sample (dark spot + strong-field blob on 64x64 frames),
2 epochs from scratch (unet_pretrained=False so no ImageNet download). Tests
assert mechanics (files, shapes, interface parity with the threshold baseline),
not segmentation quality.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from solarflare.config import SegmentConfig
from solarflare.detect.unet import normalize_inputs, pad_to_stride

N_FRAMES, H, W = 8, 64, 64


def _tiny_seg_cfg(weights: Path) -> SegmentConfig:
    return SegmentConfig(
        min_region_pixels=20,
        unet_pretrained=False,  # offline: no ImageNet download
        unet_epochs=2,
        unet_tile_px=64,
        unet_crops_per_epoch=8,
        unet_batch_size=4,
        unet_val_fraction=0.25,
        unet_weights=weights,
    )


@pytest.fixture()
def tiny_sample(tmp_path):
    """Cached synthetic sample: quiet Sun 1000 DN with a dark spot + 500 G blob."""
    from solarflare.data.cache import load_sample, write_sample

    rng = np.random.default_rng(0)
    cont = np.full((N_FRAMES, H, W), 1000.0, dtype=np.float32)
    blos = rng.normal(0.0, 10.0, (N_FRAMES, H, W)).astype(np.float32)
    yy, xx = np.ogrid[:H, :W]
    spot = (yy - 30) ** 2 + (xx - 30) ** 2 <= 8**2
    for i in range(N_FRAMES):
        cont[i][spot] = 400.0
        blos[i][spot] = 600.0
    times = pd.DataFrame(
        {
            "frame_idx": range(N_FRAMES),
            "time_utc": pd.date_range("2099-01-01", periods=N_FRAMES, freq="1h"),
        }
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


def test_normalize_inputs_ranges_and_nans():
    cont = np.full((40, 50), 1000.0, dtype=np.float32)
    cont[0, 0] = np.nan
    cont[1, 1] = 5000.0  # clipped
    blos = np.full((40, 50), -900.0, dtype=np.float32)
    blos[2, 2] = np.nan
    x = normalize_inputs(cont, blos)
    assert x.shape == (2, 40, 50) and x.dtype == np.float32
    assert np.isfinite(x).all()
    assert x[0, 0, 0] == pytest.approx(1.0)  # NaN continuum -> quiet level
    assert x[0, 1, 1] == pytest.approx(1.5)  # clip high
    assert x[1, 2, 2] == 0.0  # NaN B_los -> 0
    assert x[1].min() == pytest.approx(-1.0)  # -900 G clipped at -300 G


def test_pad_to_stride_pads_and_remembers_shape():
    x = np.ones((2, 50, 70), dtype=np.float32)
    padded, (h, w) = pad_to_stride(x)
    assert (h, w) == (50, 70)
    assert padded.shape == (2, 64, 96)
    # already-aligned input is returned unchanged
    aligned, shape = pad_to_stride(np.ones((2, 64, 64), dtype=np.float32))
    assert aligned.shape == (2, 64, 64) and shape == (64, 64)


def test_train_then_segment_roundtrip(tiny_sample, tmp_path):
    from solarflare.detect.unet import segment_sample_unet, train_unet

    cfg = _tiny_seg_cfg(tmp_path / "unet" / "best.pt")
    weights, history = train_unet([tiny_sample], cfg, seed=7)
    assert weights.exists()
    assert (weights.parent / "training_log.csv").exists()
    assert list(history["epoch"]) == [0, 1]
    assert np.isfinite(history["train_loss"]).all()

    masks_path, areas = segment_sample_unet(tiny_sample, cfg)
    masks = np.load(masks_path)
    assert masks.shape == (N_FRAMES, H, W) and masks.dtype == np.uint8
    assert set(np.unique(masks)) <= {0, 1}
    assert list(areas.columns) == ["frame_idx", "ar_pixels"]
    assert len(areas) == N_FRAMES
    assert (Path(tiny_sample.sample_dir) / "ar_mask_areas.csv").exists()


def test_dispatcher_threshold_and_missing_weights(tiny_sample, tmp_path):
    from solarflare.detect.segment import segment_sample_auto

    # threshold path: unchanged baseline behaviour
    cfg = SegmentConfig(min_region_pixels=20)
    masks_path, areas = segment_sample_auto(tiny_sample, cfg)
    assert np.load(masks_path).shape == (N_FRAMES, H, W)
    assert areas["ar_pixels"].gt(0).all()  # the synthetic blob is always present

    # unet path without trained weights: actionable error
    cfg_unet = _tiny_seg_cfg(tmp_path / "missing.pt").model_copy(update={"method": "unet"})
    with pytest.raises(FileNotFoundError, match="train-unet"):
        segment_sample_auto(tiny_sample, cfg_unet)


def test_tile_px_must_be_multiple_of_32():
    with pytest.raises(ValueError, match="multiple of 32"):
        SegmentConfig(unet_tile_px=100)
