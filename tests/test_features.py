"""Feature extraction: max-in-mask policy, resampling edges, gradients (offline)."""

import numpy as np
import pandas as pd
import pytest

from solarflare.features.extract import (
    add_gradients,
    build_frame_pipeline,
    frame_features,
    resample_features,
)


class TestFrameFeatures:
    def _channels(self):
        aia = np.zeros((10, 12), dtype=np.float32)
        aia[2, 3] = 50.0      # inside mask
        aia[8, 10] = 999.0    # OUTSIDE mask: must not contaminate the max
        blos = np.zeros((10, 12), dtype=np.float32)
        blos[2, 3] = -800.0
        blos[2, 4] = 300.0
        return {"aia_0171": aia, "hmi_magnetogram": blos}

    def _mask(self):
        mask = np.zeros((10, 12), dtype=bool)
        mask[1:4, 2:6] = True
        return mask

    def test_max_inside_mask_only(self):
        out = frame_features(self._channels(), self._mask())
        assert out["aia_0171_max"] == 50.0  # NOT 999 from outside the boundary

    def test_flux_and_area(self):
        out = frame_features(self._channels(), self._mask())
        assert out["flux_total"] == pytest.approx(1100.0)   # |−800| + |300|
        assert out["signed_flux"] == pytest.approx(-500.0)
        assert out["b_peak"] == pytest.approx(800.0)
        assert out["area_px"] == 12.0

    def test_empty_mask(self):
        out = frame_features(self._channels(), np.zeros((10, 12), dtype=bool))
        assert out["area_px"] == 0.0
        assert np.isnan(out["aia_0171_max"]) and np.isnan(out["flux_total"])

    def test_nan_pixels_ignored(self):
        ch = self._channels()
        ch["aia_0171"][2, 3] = np.nan
        out = frame_features(ch, self._mask())
        assert out["aia_0171_max"] == 0.0  # remaining in-mask pixels are zeros


class TestResample:
    def _frame_df(self):
        times = pd.date_range("2099-01-01 00:12", periods=10, freq="12min")
        return pd.DataFrame({
            "time": times,
            "aia_0171_max": np.arange(10, dtype=float),
            "area_px": np.full(10, 7.0),
        })

    def test_right_edge_labeling(self):
        out = resample_features(self._frame_df(), 60)
        # rows 00:12..01:00 (values 0..4) belong to the bin LABELED 01:00
        first = out.iloc[0]
        assert first["time"] == pd.Timestamp("2099-01-01 01:00")
        assert first["aia_0171_max"] == 4.0  # max of 0..4; nothing after 01:00
        second = out.iloc[1]
        assert second["time"] == pd.Timestamp("2099-01-01 02:00")
        assert second["aia_0171_max"] == 9.0

    def test_median_for_slow_features(self):
        out = resample_features(self._frame_df(), 60)
        assert out.iloc[0]["area_px"] == 7.0


def test_gradients_are_backward_only():
    df = pd.DataFrame({
        "time": pd.date_range("2099-01-01", periods=4, freq="1h"),
        "flux_total": [1.0, 3.0, 6.0, 6.0],
    })
    out = add_gradients(df)
    assert np.isnan(out["flux_total_d1"].iloc[0])
    assert list(out["flux_total_d1"].iloc[1:]) == [2.0, 3.0, 0.0]


def test_pipeline_shapes():
    times = pd.date_range("2099-01-01 00:12", periods=20, freq="12min")
    frame_df = pd.DataFrame({
        "time": times,
        "aia_0171_max": np.random.default_rng(0).random(20),
        "flux_total": np.linspace(1, 2, 20),
    })
    out = build_frame_pipeline(frame_df, 60)
    assert {"aia_0171_max", "flux_total", "aia_0171_max_d1", "flux_total_d1"} <= set(out.columns)
    assert (out["time"].dt.minute == 0).all()
