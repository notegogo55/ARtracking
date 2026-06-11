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
        ("m1.0", 1e-5),   # case-insensitive
        ("M", 1e-5),      # bare letter -> multiplier 1.0
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


def test_load_events_csv_missing_columns(tmp_path):
    p = tmp_path / "bad.csv"
    p.write_text("peak_time,goes_class\n2099-01-01T00:00:00,M1.0\n")
    with pytest.raises(ValueError, match="missing columns"):
        load_events_csv(p)
