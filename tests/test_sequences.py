"""Sequence building & labeling: strict boundaries and the explicit NO-LEAKAGE check."""

import numpy as np
import pandas as pd

from solarflare.features.dataset import (
    build_sequences,
    label_for_issuance,
    write_dataset,
)
from solarflare.features.extract import build_frame_pipeline

NOAA = 90001


def _events(peak: str, goes_class: str = "M5.0", noaa: int = NOAA) -> pd.DataFrame:
    return pd.DataFrame([{
        "start_time": peak, "peak_time": peak, "end_time": peak,
        "goes_class": goes_class, "noaa_ar": noaa,
    }])


class TestLabelBoundaries:
    T0 = pd.Timestamp("2099-01-02 12:00")

    def _label(self, events):
        label, _ = label_for_issuance(events, NOAA, self.T0, lead_hours=24.0)
        return label

    def test_event_just_after_t0_is_positive(self):
        assert self._label(_events("2099-01-02 12:01")) == 1

    def test_event_at_t0_already_happened(self):
        assert self._label(_events("2099-01-02 12:00")) == 0

    def test_event_at_lead_edge_inclusive(self):
        assert self._label(_events("2099-01-03 12:00")) == 1

    def test_event_past_lead_window(self):
        assert self._label(_events("2099-01-03 12:01")) == 0

    def test_sub_threshold_event_negative_but_recorded(self):
        label, biggest = label_for_issuance(
            _events("2099-01-02 13:00", goes_class="C9.9"), NOAA, self.T0, 24.0)
        assert label == 0 and biggest == "C9.9"

    def test_other_ar_event_ignored(self):
        assert self._label(_events("2099-01-02 13:00", noaa=99999)) == 0


def _frame_features(n_hours: int = 60) -> pd.DataFrame:
    """Synthetic 12-min frame features over n_hours."""
    times = pd.date_range("2099-01-01 00:12", periods=n_hours * 5, freq="12min")
    rng = np.random.default_rng(1)
    return pd.DataFrame({
        "time": times,
        "aia_0171_max": 100 + rng.random(len(times)),
        "flux_total": np.linspace(1e3, 2e3, len(times)),
    })


def _build(frame_df, events=None, **kw):
    features = build_frame_pipeline(frame_df, 60)
    defaults = dict(
        events=events if events is not None else pd.DataFrame(),
        noaa=NOAA, harp=11, window_name="w1",
        lookback_steps=24, lead_hours=24.0, min_valid_fraction=0.5,
    )
    defaults.update(kw)
    return build_sequences(features, **defaults)


class TestNoLeakage:
    def test_window_times_end_at_t0(self):
        X, rows, names = _build(_frame_features())
        assert len(rows) > 0
        for _, row in rows.iterrows():
            assert row["t_first"] == row["t0"] - pd.Timedelta(hours=23)

    def test_poisoning_the_future_changes_nothing(self):
        """THE leakage check: corrupt every raw frame after a chosen t0; the
        sequence issued at t0 must be bit-identical."""
        frame_df = _frame_features()
        X_clean, rows, _ = _build(frame_df)
        t0 = rows["t0"].iloc[3]

        poisoned = frame_df.copy()
        future = pd.to_datetime(poisoned["time"]) > t0
        assert future.sum() > 0
        poisoned.loc[future, ["aia_0171_max", "flux_total"]] *= 1000.0
        X_poisoned, rows_p, _ = _build(poisoned)

        i = rows.index[rows["t0"] == t0][0]
        j = rows_p.index[rows_p["t0"] == t0][0]
        np.testing.assert_array_equal(X_clean[i], X_poisoned[j])

    def test_poisoning_the_past_does_change_it(self):
        """Sanity counterpart: corrupting the past MUST alter the sequence."""
        frame_df = _frame_features()
        X_clean, rows, _ = _build(frame_df)
        t0 = rows["t0"].iloc[3]
        poisoned = frame_df.copy()
        past = pd.to_datetime(poisoned["time"]) <= t0
        poisoned.loc[past, "aia_0171_max"] += 5000.0
        X_poisoned, rows_p, _ = _build(poisoned)
        i = rows.index[rows["t0"] == t0][0]
        j = rows_p.index[rows_p["t0"] == t0][0]
        assert not np.array_equal(X_clean[i], X_poisoned[j])


class TestGates:
    def test_longitude_gate(self):
        frame_df = _frame_features()
        features = build_frame_pipeline(frame_df, 60)
        t = pd.to_datetime(features["time"])
        # AR crosses 65 deg halfway through
        lon = pd.Series(
            np.linspace(40, 90, len(features)), index=t)
        X, rows, _ = _build(frame_df, lon_series=lon, max_lon_deg=65.0)
        assert len(rows) > 0
        assert (rows["lon_t0_deg"].abs() <= 65.0).all()
        # and some issuance times were actually rejected
        X_all, rows_all, _ = _build(frame_df)
        assert len(rows) < len(rows_all)

    def test_min_valid_fraction_drops_gappy_windows(self):
        frame_df = _frame_features()
        frame_df.loc[40:200, "aia_0171_max"] = np.nan   # big hole
        X_strict, rows_strict, _ = _build(frame_df, min_valid_fraction=0.95)
        X_loose, rows_loose, _ = _build(frame_df, min_valid_fraction=0.3)
        assert len(rows_strict) < len(rows_loose)

    def test_labels_attached(self):
        events = _events("2099-01-02 06:30")  # M5.0
        X, rows, _ = _build(_frame_features(), events=events)
        labeled = rows.set_index("t0")["label"]
        # positive iff t0 < peak <= t0+24h
        peak = pd.Timestamp("2099-01-02 06:30")
        for t0, label in labeled.items():
            expected = int(t0 < peak <= t0 + pd.Timedelta(hours=24))
            assert label == expected, t0


def test_write_dataset_round_trip(tmp_path):
    X, rows, names = _build(_frame_features(), events=_events("2099-01-02 06:30"))
    stats = write_dataset(tmp_path / "ds", X, rows, names, {"config_hash": "abc"})
    assert stats["n_sequences"] == len(rows)
    assert 0 <= stats["positive_rate"] <= 1
    loaded = np.load(tmp_path / "ds" / "X.npz", allow_pickle=False)
    np.testing.assert_array_equal(loaded["X"], X)
    assert list(loaded["feature_names"]) == names
    table = pd.read_parquet(tmp_path / "ds" / "samples.parquet")
    assert len(table) == len(rows)
    assert (tmp_path / "ds" / "data_dictionary.json").exists()
