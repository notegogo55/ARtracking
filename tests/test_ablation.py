"""Ablation harness: grouped permutation importance with a known-informative feature."""

import numpy as np
import pandas as pd
import pytest

from solarflare.forecast.ablation import (
    LAYER_CASES,
    available_aia_channels,
    case_feature_indices,
    drop_one_retrain,
    feature_groups,
    permutation_importance,
    resolve_case_channels,
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
    X[y == 1, :, 0] += 1.0  # informative channel
    X[:, :, 2] += rng.normal(0, 1, (n, t))  # loud but useless channel
    return X, y


def test_informative_group_dominates():
    X, y = _data()
    table = permutation_importance(_StubModel(), X, y, NAMES, threshold=0.5, n_repeats=3, seed=1)
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
        train_fn,
        X[halves[0]],
        y[halves[0]],
        X[halves[1]],
        y[halves[1]],
        X[halves[2]],
        y[halves[2]],
        NAMES,
        focus_groups=["aia_0131", "flux_total"],
    )
    table = table.set_index("group")
    assert table.loc["<full model>", "delta_tss"] == 0.0
    assert table.loc["aia_0131", "delta_tss"] > 0.3
    assert table.loc["flux_total", "delta_tss"] < 0.15


def test_unknown_focus_group_raises():
    X, y = _data(n=40)
    with pytest.raises(ValueError, match="unknown focus groups"):
        drop_one_retrain(lambda *a: _StubModel(), X, y, X, y, X, y, NAMES, focus_groups=["nope"])


# --- M6: atmospheric-layer combination ablation -----------------------------
LAYER_NAMES = [
    "aia_0094_max",
    "aia_0094_max_d1",
    "aia_0131_max",
    "aia_0171_max",
    "aia_0304_max",
    "aia_1600_max",
    "flux_total",
    "flux_total_d1",
    "area_px",
]


def test_available_aia_channels_parsed():
    assert available_aia_channels(LAYER_NAMES) == [94, 131, 171, 304, 1600]


def test_case_feature_indices_keeps_magnetic_plus_selected_aia():
    # case4 flaring corona = AIA 94 + 131; magnetic/area always kept
    idx = case_feature_indices(LAYER_NAMES, [94, 131])
    kept = [LAYER_NAMES[i] for i in idx]
    assert kept == [
        "aia_0094_max",
        "aia_0094_max_d1",
        "aia_0131_max",
        "flux_total",
        "flux_total_d1",
        "area_px",
    ]
    assert not any("0304" in k or "0171" in k or "1600" in k for k in kept)


def test_case1_is_hmi_only_and_case6_is_all():
    case1 = case_feature_indices(
        LAYER_NAMES, resolve_case_channels(LAYER_CASES["case1_baseline_surface"], LAYER_NAMES)
    )
    assert [LAYER_NAMES[i] for i in case1] == ["flux_total", "flux_total_d1", "area_px"]
    case6 = resolve_case_channels(LAYER_CASES["case6_full_spectrum"], LAYER_NAMES)
    assert case6 == available_aia_channels(LAYER_NAMES)
    # a case channel absent from the data is dropped (no crash)
    assert resolve_case_channels((211, 94), LAYER_NAMES) == [94]


def test_ablate_layers_runs_each_case():
    from solarflare.forecast.ablation import ablate_layers

    rng = np.random.default_rng(2)
    n, t = 120, 6
    X = rng.normal(0.0, 0.3, (n, t, len(LAYER_NAMES))).astype(np.float32)
    y = (rng.random(n) < 0.35).astype(int)
    X[y == 1, :, 0] += 1.0  # make AIA 94 informative
    samples = pd.DataFrame(
        {"t0": pd.date_range("2099-01-01", periods=n, freq="6h"), "label_24h_M1.0": y}
    )
    table, oof = ablate_layers(
        X, samples, LAYER_NAMES, n_folds=3, embargo_hours=0, model_names=("climatology",)
    )
    assert set(table["case"]) == set(LAYER_CASES)  # all six cases scored
    assert (table.loc[table["case"] == "case1_baseline_surface", "n_aia_channels"] == 0).all()
    assert (table.loc[table["case"] == "case4_flaring_corona", "n_aia_channels"] == 2).all()
    assert {"tss_mean", "hss_mean", "bss_mean", "label_col"} <= set(table.columns)
    assert set(oof) == set(LAYER_CASES)
