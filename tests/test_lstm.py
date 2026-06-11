"""LSTM forecaster: learns a separable synthetic task, deterministic, sane outputs."""

import numpy as np
import pytest

from solarflare.forecast.lstm import LSTMConfig, LSTMForecaster


def _separable(n=240, t=12, f=3, seed=0):
    """Positive sequences ramp up in feature 0 near the end; negatives don't."""
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.3).astype(int)   # imbalanced on purpose
    X = rng.normal(0, 0.3, (n, t, f)).astype(np.float32)
    ramp = np.concatenate([np.zeros(t - 4), np.linspace(0.5, 2.5, 4)])
    X[y == 1, :, 0] += ramp
    X[rng.random((n, t)) < 0.02] = np.nan   # sprinkle missing values
    return X, y


@pytest.fixture(scope="module")
def fitted():
    X, y = _separable()
    cut = int(0.7 * len(X))
    cfg = LSTMConfig(max_epochs=12, patience=12, batch_size=64, hidden_size=32,
                     num_layers=1, seed=7)
    model = LSTMForecaster(cfg).fit(X[:cut], y[:cut], X[cut:], y[cut:])
    return model, X, y, cut


def test_learns_separable_task(fitted):
    from solarflare.eval.metrics import best_tss_threshold

    model, X, y, cut = fitted
    p = model.predict_proba(X[cut:])
    _, val_tss = best_tss_threshold(y[cut:], p)
    assert val_tss > 0.6, f"LSTM failed to learn separable task (TSS={val_tss:.2f})"


def test_probabilities_valid(fitted):
    model, X, _, cut = fitted
    p = model.predict_proba(X[cut:])
    assert p.shape == (len(X) - cut,)
    assert np.all((p >= 0) & (p <= 1))
    assert np.all(np.isfinite(p))


def test_history_recorded(fitted):
    model, *_ = fitted
    assert len(model.history_) >= 1
    assert "train_loss" in model.history_[0]
    assert "val_tss" in model.history_[0]


def test_deterministic_given_seed():
    X, y = _separable(n=120)
    cut = 80
    cfg = LSTMConfig(max_epochs=3, patience=3, batch_size=64, hidden_size=16,
                     num_layers=1, seed=11)
    p1 = LSTMForecaster(cfg).fit(X[:cut], y[:cut], X[cut:], y[cut:]).predict_proba(X[cut:])
    p2 = LSTMForecaster(cfg).fit(X[:cut], y[:cut], X[cut:], y[cut:]).predict_proba(X[cut:])
    np.testing.assert_allclose(p1, p2, rtol=1e-5, atol=1e-6)
