"""Time-blocked CV harness: chronology, embargo, aggregation."""

import numpy as np
import pandas as pd
import pytest

from solarflare.forecast.validate import (
    aggregate_table,
    crossval_table,
    time_blocked_folds,
)


def _times(n=100):
    return pd.Series(pd.date_range("2099-01-01", periods=n, freq="6h"))


class TestFolds:
    def test_no_overlap_and_chronological_blocks(self):
        t = _times()
        folds = time_blocked_folds(t, n_folds=5, embargo_hours=0)
        assert len(folds) == 5
        seen = set()
        for tr, va in folds:
            assert set(tr) & set(va) == set()
            assert not (set(va) & seen)
            seen |= set(va)
            # validation block is contiguous in time
            vt = t.iloc[sorted(va)]
            assert vt.is_monotonic_increasing
        assert seen == set(range(100))

    def test_embargo_removes_neighbours(self):
        t = _times()
        no_embargo = time_blocked_folds(t, 5, embargo_hours=0)
        embargoed = time_blocked_folds(t, 5, embargo_hours=24)
        # middle fold: embargo (24h = 4 samples each side) shrinks training set
        tr0, va0 = no_embargo[2]
        tr1, va1 = embargoed[2]
        assert set(va0) == set(va1)
        assert len(tr1) < len(tr0)
        v_start = t.iloc[va1].min()
        margin = pd.Timedelta(hours=24)
        assert all(not (v_start - margin <= t.iloc[i] < v_start) for i in tr1)


def test_crossval_climatology_has_no_skill():
    rng = np.random.default_rng(0)
    n = 120
    X = rng.normal(size=(n, 8, 2)).astype(np.float32)
    y = (rng.random(n) < 0.3).astype(int)
    per_fold, oof = crossval_table(
        X,
        y,
        _times(n),
        ["a", "b"],
        horizon_steps=4,
        n_folds=3,
        embargo_hours=0,
        model_names=("climatology",),
    )
    assert len(per_fold) == 3
    assert np.allclose(per_fold["tss"], 0.0, atol=1e-9)
    y_oof, p_oof = oof["climatology"]
    assert len(y_oof) == len(p_oof) == n


def test_holdout_freezes_threshold_and_reports(tmp_path):
    from solarflare.forecast.validate import holdout_evaluate, roc_plot

    rng = np.random.default_rng(3)
    n = 200
    X = rng.normal(size=(n, 6, 2)).astype(np.float32)
    y = (rng.random(n) < 0.3).astype(int)
    X[y == 1, :, 0] += 1.5
    t0 = _times(n)
    table, predictions, fitted = holdout_evaluate(
        X[:150],
        y[:150],
        t0[:150],
        X[150:],
        y[150:],
        ["a", "b"],
        horizon_steps=3,
        model_names=("climatology", "holt_winters"),
    )
    assert set(table["model"]) == {"climatology", "holt_winters"}
    assert {"tss", "hss", "bss", "threshold"} <= set(table.columns)
    y_te, p_te = predictions["holt_winters"]
    assert len(y_te) == len(p_te) == 50
    out = roc_plot(predictions, tmp_path / "roc.png")
    assert out.exists()


def test_crossval_grid_runs_each_cell_and_skips_degenerate():
    """M3: one call scores every {horizon x class} cell; single-class cells skip."""
    from solarflare.forecast.validate import crossval_grid

    rng = np.random.default_rng(5)
    n = 120
    X = rng.normal(size=(n, 4, 2)).astype(np.float32)
    y24 = (rng.random(n) < 0.3).astype(int)
    y72 = (rng.random(n) < 0.4).astype(int)
    samples = pd.DataFrame(
        {
            "t0": _times(n),
            "label": y24,
            "label_24h_M1.0": y24,
            "label_72h_M1.0": y72,
            "label_24h_X1.0": np.zeros(n, dtype=int),  # all-negative -> skipped
        }
    )
    table, oof = crossval_grid(
        X,
        samples,
        ["a", "b"],
        horizon_steps=4,
        n_folds=3,
        embargo_hours=0,
        model_names=("climatology",),
    )
    assert set(table["label_col"]) == {"label_24h_M1.0", "label_72h_M1.0"}
    assert {"horizon_h", "class", "n_positive", "tss_mean"} <= set(table.columns)
    assert set(oof) == {"label_24h_M1.0", "label_72h_M1.0"}
    assert set(table.loc[table["label_col"] == "label_72h_M1.0", "horizon_h"]) == {72.0}


def test_aggregate_table_orders_by_tss():
    per_fold = pd.DataFrame(
        [
            {
                "model": "a",
                "fold": 0,
                "tss": 0.2,
                "hss": 0.1,
                "bss": 0.0,
                "brier": 0.2,
                "precision": 0.5,
                "recall": 0.5,
            },
            {
                "model": "a",
                "fold": 1,
                "tss": 0.4,
                "hss": 0.2,
                "bss": 0.1,
                "brier": 0.2,
                "precision": 0.5,
                "recall": 0.5,
            },
            {
                "model": "b",
                "fold": 0,
                "tss": 0.8,
                "hss": 0.6,
                "bss": 0.3,
                "brier": 0.1,
                "precision": 0.7,
                "recall": 0.7,
            },
            {
                "model": "b",
                "fold": 1,
                "tss": 0.6,
                "hss": 0.5,
                "bss": 0.2,
                "brier": 0.1,
                "precision": 0.7,
                "recall": 0.7,
            },
        ]
    )
    table = aggregate_table(per_fold)
    assert list(table["model"]) == ["b", "a"]
    assert table.iloc[0]["tss_mean"] == 0.7
    assert table.iloc[1]["tss_std"] == pytest.approx(np.std([0.2, 0.4], ddof=1))
