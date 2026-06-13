"""GOES class parsing and offline event-CSV loading (no network)."""

import pytest

from solarflare.data.goes_events import goes_class_to_flux, load_events_csv


@pytest.mark.parametrize(
    ("cls", "flux"),
    [
        ("A1.0", 1e-8),
        ("B2.5", 2.5e-7),
        ("C5", 5e-6),
        ("M1.0", 1e-5),
        ("m1.0", 1e-5),  # case-insensitive
        ("M", 1e-5),  # bare letter -> multiplier 1.0
        ("X9.3", 9.3e-4),
        ("X17.2", 1.72e-3),  # above-X10 events keep scaling linearly
    ],
)
def test_goes_class_to_flux(cls, flux):
    assert goes_class_to_flux(cls) == pytest.approx(flux)


def test_class_ordering():
    assert goes_class_to_flux("C9.9") < goes_class_to_flux("M1.0") < goes_class_to_flux("X1.0")


@pytest.mark.parametrize("bad", ["", "Z1.0", "M-1", "Mfoo", "1.0M"])
def test_invalid_classes_raise(bad):
    with pytest.raises(ValueError):
        goes_class_to_flux(bad)


def test_load_events_csv(sample_events_csv):
    df = load_events_csv(sample_events_csv)
    assert len(df) == 6
    assert str(df["peak_time"].dtype).startswith("datetime64")


def test_select_ar_events(sample_events_csv):
    from solarflare.data.goes_events import select_ar_events

    events = load_events_csv(sample_events_csv)
    sel = select_ar_events(events, 90001, "2099-01-01", "2099-01-04")
    assert list(sel["goes_class"]) == ["C5.0", "M1.5", "M2.0"]
    # window end is exclusive and other ARs are excluded
    assert select_ar_events(events, 90001, "2099-01-01", "2099-01-03").shape[0] == 1
    assert select_ar_events(events, 90003, "2099-01-01", "2099-03-01").shape[0] == 1
    assert select_ar_events(events, 12345, "2099-01-01", "2099-03-01").empty


def test_empty_hek_result_is_not_cached(tmp_path, monkeypatch):
    """A 0-event HEK answer must not poison the cache (transient outages happen).

    Regression: 2026-06-13 a network blip cached 0 events for Mar 2012 (X5.4
    window), silently emptying the labels of every later fetch of that range.
    """
    from datetime import datetime

    from sunpy.net import Fido

    from solarflare.data.goes_events import fetch_goes_events

    monkeypatch.setattr(Fido, "search", lambda *a, **k: {"hek": []})
    df = fetch_goes_events(datetime(2099, 1, 1), datetime(2099, 1, 2), cache_dir=tmp_path)
    assert df.empty
    assert not list(tmp_path.glob("goes_flares_*.csv"))  # nothing cached


def test_load_events_csv_missing_columns(tmp_path):
    p = tmp_path / "bad.csv"
    p.write_text("peak_time,goes_class\n2099-01-01T00:00:00,M1.0\n")
    with pytest.raises(ValueError, match="missing columns"):
        load_events_csv(p)
