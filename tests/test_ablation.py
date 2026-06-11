"""Ablation harness: grouped permutation importance with a known-informative feature."""

import numpy as np
import pandas as pd
import pytest

from solarflare.forecast.ablation import (
    drop_one_retrain,
    feature_groups,
    permutation_importance,
)

NAMES = ["aia_0131_max", "aia_0131_max_d1", "flux_total", "flux_total_d1"]


def test_feature_groups_bundle_gradients():
    groups = feature_groups(NAMES)
    assert groups == {"aia_0131": [0, 1], "flux_total": [2, 3]}


class _StubModel:
    """Probability driven ONLY by feature 0's mean over the last 4 steps."""

    def predict_proba(self, X):
        signal = np.nanmean(X[:, -4:, 0], axis=1)
        return 1.0 / (1.0 + np.exp(-4.0 * (signal - 0.5)))


def _data(n=400, t=10, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(0.0, 0.2, (n, t, len(NAMES))).astype(np.float32)
    y = (rng.random(n) < 0.4).astype(int)
    X[y == 1, :, 0] += 1.0      # informative channel
    X[:, :, 2] += rng.normal(0, 1, (n, t))  # loud but useless channel
    return X, y


def test_informative_group_dominates():
    X, y = _data()
    table = permutation_importance(_StubModel(), X, y, NAMES, threshold=0.5,
                                   n_repeats=3, seed=1)
    table = table.set_index("group")
    assert table.loc["aia_0131", "delta_tss_mean"] > 0.5
    assert abs(table.loc["flux_total", "delta_tss_mean"]) < 0.05
    assert table.index[0] == "aia_0131"  # sorted by importance


def test_permutation_is_deterministic_given_seed():
    X, y = _data()
    t1 = permutation_importance(_StubModel(), X, y, NAMES, 0.5, seed=42)
    t2 = permutation_importance(_StubModel(), X, y, NAMES, 0.5, seed=42)
    pd.testing.assert_frame_equal(t1, t2)


def test_drop_one_retrain_flags_informative_group():
    X, y = _data()
    halves = np.array_split(np.arange(len(X)), 4)

    class _MeanModel:
        """'Trains' by remembering which columns it received."""

        def __init__(self, n_cols):
            self.n_cols = n_cols

        def predict_proba(self, X):
            # without column 0 the first remaining column is noise
            signal = np.nanmean(X[:, -4:, 0], axis=1)
            return 1.0 / (1.0 + np.exp(-4.0 * (signal - 0.5)))

    def train_fn(X_tr, y_tr, X_va, y_va):
        return _MeanModel(X_tr.shape[-1])

    table = drop_one_retrain(
        train_fn, X[halves[0]], y[halves[0]], X[halves[1]], y[halves[1]],
        X[halves[2]], y[halves[2]], NAMES, focus_groups=["aia_0131", "flux_total"],
    )
    table = table.set_index("group")
    assert table.loc["<full model>", "delta_tss"] == 0.0
    assert table.loc["aia_0131", "delta_tss"] > 0.3
    assert table.loc["flux_total", "delta_tss"] < 0.15


def test_unknown_focus_group_raises():
    X, y = _data(n=40)
    with pytest.raises(ValueError, match="unknown focus groups"):
        drop_one_retrain(lambda *a: _StubModel(), X, y, X, y, X, y, NAMES,
                         focus_groups=["nope"])
