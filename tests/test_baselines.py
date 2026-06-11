"""Climatology and Holt-Winters baselines on synthetic sequences."""

import numpy as np
import pytest

from solarflare.forecast.baselines import (
    ClimatologyForecaster,
    EnsembleForecaster,
    HoltWintersForecaster,
    _holt_forecast_max,
)


def _synthetic(n=80, t=20, seed=0):
    """Positives have a rising key feature; negatives are flat noise."""
    rng = np.random.default_rng(seed)
    y = (np.arange(n) % 2 == 0).astype(int)
    X = rng.normal(0, 0.1, (n, t, 2)).astype(np.float32)
    ramp = np.linspace(0, 3, t)
    X[y == 1, :, 0] += ramp
    return X, y


def test_climatology_constant():
    X, y = _synthetic()
    model = ClimatologyForecaster().fit(X, y)
    p = model.predict_proba(X)
    assert np.allclose(p, y.mean())


def test_holt_forecast_max_rising_series():
    rising = np.linspace(1, 10, 20)
    flat = np.full(20, 5.0)
    assert _holt_forecast_max(rising, 10) > 10.0   # trend extrapolated upward
    assert _holt_forecast_max(flat, 10) == pytest.approx(5.0, abs=0.1)


def test_holt_forecast_max_handles_nans():
    s = np.linspace(1, 10, 20)
    s[3:6] = np.nan
    assert np.isfinite(_holt_forecast_max(s, 5))


def test_hw_baseline_discriminates():
    X, y = _synthetic()
    model = HoltWintersForecaster(["key", "other"], key_feature="key",
                                  horizon_steps=5).fit(X, y)
    p = model.predict_proba(X)
    assert p.shape == (len(X),)
    assert np.all((p >= 0) & (p <= 1))
    assert p[y == 1].mean() > p[y == 0].mean() + 0.2


def test_ensemble_averages():
    X, y = _synthetic()
    a = ClimatologyForecaster()
    b = ClimatologyForecaster()
    ens = EnsembleForecaster([a, b]).fit(X, y)
    assert np.allclose(ens.predict_proba(X), y.mean())
