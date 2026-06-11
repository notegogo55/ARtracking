"""Video renderer: frame composition and MP4 encoding on a tiny synthetic sample."""

import numpy as np
import pandas as pd
import pytest

from solarflare.data.cache import load_sample, write_sample
from solarflare.viz.video import (
    SampleScaling,
    compose_frame,
    flare_annotations,
    render_sample_video,
)


@pytest.fixture()
def synth_sample(tmp_path):
    rng = np.random.default_rng(0)
    n, h, w = 6, 24, 30
    arrays = {
        "hmi_continuum": rng.normal(1000, 30, (n, h, w)).astype(np.float32),
        "hmi_magnetogram": rng.normal(0, 120, (n, h, w)).astype(np.float32),
        "aia_0171": np.abs(rng.normal(200, 60, (n, h, w))).astype(np.float32),
    }
    times = pd.DataFrame({
        "frame_idx": range(n),
        "time_utc": pd.date_range("2099-01-01", periods=n, freq="12min"),
    })
    labels = pd.DataFrame([{
        "start_time": "2099-01-01 00:20", "peak_time": "2099-01-01 00:24",
        "end_time": "2099-01-01 00:40", "goes_class": "M5.0", "noaa_ar": 90001,
    }])
    sdir = write_sample(tmp_path / "s", arrays, times, pd.DataFrame(), labels,
                        {"noaa": 90001, "harp": 11, "window": "w1"})
    masks = np.zeros((n, h, w), dtype=np.uint8)
    masks[:, 8:16, 10:20] = 1
    np.save(sdir / "ar_masks.npy", masks)
    return load_sample(sdir), masks


def test_compose_frame_shape_and_annotation(synth_sample):
    sample, masks = synth_sample
    scaling = SampleScaling(sample, "aia_0171")
    rgb = compose_frame(sample, masks, 2, "aia_0171", scaling, "M5.0 flare!")
    assert rgb.dtype == np.uint8
    assert rgb.shape[2] == 3
    assert rgb.shape[1] >= 3 * 30  # three panels wide


def test_flare_annotations_window(synth_sample):
    sample, _ = synth_sample
    notes = flare_annotations(sample, window_minutes=15)
    # peak 00:24: frames at 00:12, 00:24, 00:36 are within 15 min
    assert set(notes.values()) == {"M5.0 flare!"}
    assert 2 in notes
    assert 0 not in notes


def test_render_video_mp4(synth_sample, tmp_path):
    sample, masks = synth_sample
    out = render_sample_video(sample, masks, tmp_path / "clip.mp4",
                              channel=171, fps=6)
    assert out.exists()
    assert out.stat().st_size > 5_000  # a real encoded file, not a stub


def test_render_video_time_range(synth_sample, tmp_path):
    sample, masks = synth_sample
    out = render_sample_video(
        sample, masks, tmp_path / "clip2.mp4", channel=171,
        start=pd.Timestamp("2099-01-01 00:24"), end=pd.Timestamp("2099-01-01 00:48"))
    assert out.exists()
    with pytest.raises(ValueError, match="no frames"):
        render_sample_video(sample, masks, tmp_path / "x.mp4", channel=171,
                            start=pd.Timestamp("2099-02-01"))
