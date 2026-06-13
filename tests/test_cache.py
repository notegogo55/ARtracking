"""Sample-cache round trip: arrays, tables, and metadata survive write/load."""

import numpy as np
import pandas as pd
import pytest

from solarflare.data.cache import channel_key, load_sample, write_sample


def _toy_sample(tmp_path):
    rng = np.random.default_rng(7)
    arrays = {
        "hmi_magnetogram": rng.normal(0, 100, (3, 4, 5)).astype(np.float32),
        channel_key(171): rng.random((3, 4, 5)).astype(np.float32),
    }
    times = pd.DataFrame(
        {
            "frame_idx": [0, 1, 2],
            "time_utc": pd.to_datetime(
                ["2099-01-01T00:00", "2099-01-01T00:12", "2099-01-01T00:24"]
            ),
        }
    )
    qa = pd.DataFrame(
        {
            "frame_idx": [0, 0],
            "channel": ["hmi_magnetogram", "aia_0171"],
            "flagged": [False, True],
        }
    )
    labels = pd.DataFrame(
        {
            "start_time": ["2099-01-01T00:05"],
            "peak_time": ["2099-01-01T00:10"],
            "end_time": ["2099-01-01T00:20"],
            "goes_class": ["M1.5"],
            "noaa_ar": [90001],
        }
    )
    meta = {"noaa": 90001, "harp": 11, "window": "w1"}
    return write_sample(tmp_path / "s", arrays, times, qa, labels, meta), arrays


def test_round_trip(tmp_path):
    sample_dir, arrays = _toy_sample(tmp_path)
    sample = load_sample(sample_dir)
    assert sample.n_frames == 3
    assert set(sample.arrays) == set(arrays)
    for key in arrays:
        np.testing.assert_allclose(np.asarray(sample.arrays[key]), arrays[key])
    assert sample.meta["noaa"] == 90001
    assert sample.meta["array_shapes"]["hmi_magnetogram"] == [3, 4, 5]
    assert len(sample.labels) == 1
    assert bool(sample.qa["flagged"].iloc[1]) is True


def test_derived_arrays_excluded_and_overwritable(tmp_path):
    """ar_masks.npy is a product, not a channel: never mmap'd by load_sample.

    Regression: on Windows a live read-only mmap blocks overwriting the file,
    so re-running segment-sample on an already-segmented sample crashed.
    """
    sample_dir, _ = _toy_sample(tmp_path)
    np.save(sample_dir / "ar_masks.npy", np.zeros((3, 4, 5), dtype=np.uint8))
    sample = load_sample(sample_dir)
    assert "ar_masks" not in sample.arrays
    # overwrite must succeed while the sample (mmaps) is still alive
    np.save(sample_dir / "ar_masks.npy", np.ones((3, 4, 5), dtype=np.uint8))
    assert np.load(sample_dir / "ar_masks.npy").max() == 1


def test_channel_key():
    assert channel_key(94) == "aia_0094"
    assert channel_key(1700) == "aia_1700"


def test_load_missing_dir_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_sample(tmp_path / "nope")


def test_empty_labels_ok(tmp_path):
    write_sample(
        tmp_path / "s2",
        {"hmi_magnetogram": np.zeros((1, 2, 2), dtype=np.float32)},
        pd.DataFrame({"frame_idx": [0], "time_utc": pd.to_datetime(["2099-01-01"])}),
        pd.DataFrame(),
        pd.DataFrame(),
        {},
    )
    sample = load_sample(tmp_path / "s2")
    assert sample.labels.empty and sample.qa.empty
