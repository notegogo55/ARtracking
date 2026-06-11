"""Base-rate computation against hand-computed values on the synthetic fixture.

Fixture (year 2099, synthetic): within 2099-01-01 .. 2099-01-11 there are
3 events >= M1.0 (two on Jan 3, one on Jan 6) and 2 sub-threshold C-class
events; one M5.0 lies outside the window. With 24 h bins anchored at the
window start: 10 bins, 2 positive -> base rate 0.2.
"""

from datetime import datetime

import pandas as pd
import pytest

from solarflare.data.goes_events import load_events_csv
from solarflare.forecast.climatology import compute_base_rate

WIN = (datetime(2099, 1, 1), datetime(2099, 1, 11))


def test_base_rate_hand_computed(sample_events_csv):
    events = load_events_csv(sample_events_csv)
    r = compute_base_rate(events, [WIN], bin_hours=24, min_class="M1.0")
    assert r.n_bins == 10
    assert r.n_positive_bins == 2
    assert r.n_events == 3  # the out-of-window M5.0 must not count
    assert r.base_rate == pytest.approx(0.2)
    assert r.brier_climatology == pytest.approx(0.2 * 0.8)


def test_threshold_changes_rate(sample_events_csv):
    events = load_events_csv(sample_events_csv)
    # X1.0 threshold: only the Jan 6 event qualifies -> 1/10
    r = compute_base_rate(events, [WIN], bin_hours=24, min_class="X1.0")
    assert r.n_positive_bins == 1
    assert r.base_rate == pytest.approx(0.1)
    # C1.0 threshold: Jan 2, 3, 6, 9 are positive -> 4/10
    r = compute_base_rate(events, [WIN], bin_hours=24, min_class="C1.0")
    assert r.base_rate == pytest.approx(0.4)


def test_partial_trailing_bin_dropped(sample_events_csv):
    events = load_events_csv(sample_events_csv)
    # 10-day + 12-hour window: the trailing half-bin is dropped -> still 10 bins
    r = compute_base_rate(events, [(WIN[0], datetime(2099, 1, 11, 12))], bin_hours=24)
    assert r.n_bins == 10


def test_multiple_windows(sample_events_csv):
    events = load_events_csv(sample_events_csv)
    windows = [WIN, (datetime(2099, 2, 1), datetime(2099, 2, 3))]  # catches the M5.0
    r = compute_base_rate(events, windows, bin_hours=24)
    assert r.n_bins == 12
    assert r.n_positive_bins == 3
    assert r.n_events == 4


def test_empty_events_gives_zero_rate():
    empty = pd.DataFrame({"peak_time": pd.to_datetime([]), "goes_class": []})
    r = compute_base_rate(empty, [WIN], bin_hours=24)
    assert r.base_rate == 0.0
    assert r.brier_climatology == 0.0


def test_no_complete_bins_raises(sample_events_csv):
    events = load_events_csv(sample_events_csv)
    with pytest.raises(ValueError, match="no complete"):
        compute_base_rate(events, [(WIN[0], datetime(2099, 1, 1, 12))], bin_hours=24)
