"""Statistical baselines every learned model must beat (master prompt §7).

- ClimatologyForecaster: constant train base rate (TSS = 0 by construction).
- HoltWintersForecaster: per-sequence Holt linear-trend extrapolation of one
  key feature over the lead horizon, calibrated to a probability by logistic
  regression fitted on TRAIN only.
Both consume the same (X, y) sequence arrays as the LSTM.
"""

from __future__ import annotations

import logging

import numpy as np

log = logging.getLogger(__name__)


class ClimatologyForecaster:
    """Predicts the constant training-set base rate."""

    def __init__(self) -> None:
        self.rate_: float | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> ClimatologyForecaster:
        self.rate_ = float(np.mean(y))
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.rate_ is None:
            raise RuntimeError("fit first")
        return np.full(len(X), self.rate_, dtype=float)


def _holt_forecast_max(series: np.ndarray, horizon_steps: int) -> float:
    """Max of a Holt linear-trend forecast over the horizon (robust fallbacks)."""
    s = np.asarray(series, dtype=float)
    finite = np.isfinite(s)
    if finite.sum() < 4 or np.nanstd(s) == 0:
        return float(s[finite][-1]) if finite.any() else np.nan
    # forward-fill then back-fill the short gaps
    idx = np.where(finite)[0]
    s = np.interp(np.arange(len(s)), idx, s[idx])
    try:
        from statsmodels.tsa.holtwinters import Holt

        fit = Holt(s, initialization_method="estimated").fit(optimized=True)
        fc = fit.forecast(horizon_steps)
        return float(np.max(fc))
    except Exception as err:  # noqa: BLE001 - any fit failure degrades gracefully
        log.warning("Holt fit failed (%r); using last value", err)
        return float(s[-1])


class HoltWintersForecaster:
    """Holt trend extrapolation of one feature -> logistic probability."""

    def __init__(
        self,
        feature_names: list[str],
        key_feature: str | None = None,
        horizon_steps: int = 24,
        max_fit_samples: int = 20_000,
        seed: int = 0,
    ) -> None:
        if key_feature is None:
            preferred = [n for n in feature_names
                         if n in ("flux_total", "USFLUX", "TOTUSJH")]
            key_feature = preferred[0] if preferred else feature_names[0]
        self.feature_idx = feature_names.index(key_feature)
        self.key_feature = key_feature
        self.horizon_steps = horizon_steps
        self.max_fit_samples = max_fit_samples
        self.seed = seed
        self._calibrator = None
        self._scale: tuple[float, float] | None = None

    def _scores(self, X: np.ndarray) -> np.ndarray:
        series = X[:, :, self.feature_idx]
        return np.array([_holt_forecast_max(s, self.horizon_steps) for s in series])

    def fit(self, X: np.ndarray, y: np.ndarray) -> HoltWintersForecaster:
        from sklearn.linear_model import LogisticRegression

        rng = np.random.default_rng(self.seed)
        idx = np.arange(len(X))
        if len(idx) > self.max_fit_samples:
            idx = rng.choice(idx, self.max_fit_samples, replace=False)
        scores = self._scores(X[idx])
        ok = np.isfinite(scores)
        z = scores[ok]
        y_fit = np.asarray(y)[idx][ok]
        if len(np.unique(y_fit)) < 2 or len(z) == 0:
            # degenerate training block (one class): degrade to climatology
            self._fallback_rate = float(np.mean(y))
            self._calibrator = None
            self._scale = None
            log.warning("HW baseline: single-class training data; "
                        "falling back to constant rate %.3f", self._fallback_rate)
            return self
        mu, sigma = float(np.mean(z)), float(np.std(z) or 1.0)
        self._scale = (mu, sigma)
        self._calibrator = LogisticRegression().fit(
            ((z - mu) / sigma).reshape(-1, 1), y_fit
        )
        self._fallback_rate = None
        log.info("HW baseline calibrated on %d samples (key=%s)", ok.sum(),
                 self.key_feature)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self._calibrator is None:
            if getattr(self, "_fallback_rate", None) is None:
                raise RuntimeError("fit first")
            return np.full(len(X), self._fallback_rate, dtype=float)
        scores = self._scores(X)
        mu, sigma = self._scale
        z = np.where(np.isfinite(scores), (scores - mu) / sigma, 0.0)
        return self._calibrator.predict_proba(z.reshape(-1, 1))[:, 1]


class EnsembleForecaster:
    """Weighted average of member probabilities (default equal weights)."""

    def __init__(self, members: list, weights: list[float] | None = None) -> None:
        self.members = members
        self.weights = weights or [1.0 / len(members)] * len(members)

    def fit(self, X: np.ndarray, y: np.ndarray):
        for member in self.members:
            member.fit(X, y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        probs = [m.predict_proba(X) for m in self.members]
        return np.average(np.column_stack(probs), axis=1, weights=self.weights)
