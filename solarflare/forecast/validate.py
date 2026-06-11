"""Time-blocked validation harness, metrics table, reliability diagram, sweeps.

Splits are ALWAYS chronological (master prompt §7): contiguous blocks ordered
by issuance time, with an embargo gap removed from training around each
validation block to kill autocorrelation leakage. Thresholds are chosen on the
validation block and frozen before any test evaluation.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from solarflare.eval.metrics import best_tss_threshold, reliability_curve, summarize

log = logging.getLogger(__name__)


def time_blocked_folds(
    t0s: pd.Series, n_folds: int, embargo_hours: float = 48.0
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Chronological contiguous folds: (train_idx, val_idx) per fold.

    Training indices exclude the embargo margin around the validation block.
    """
    t = pd.to_datetime(t0s).reset_index(drop=True)
    order = np.argsort(t.to_numpy(), kind="stable")
    blocks = np.array_split(order, n_folds)
    folds = []
    margin = pd.Timedelta(hours=embargo_hours)
    for k, val_idx in enumerate(blocks):
        if len(val_idx) == 0:
            continue
        v_start, v_end = t.iloc[val_idx].min(), t.iloc[val_idx].max()
        train_mask = np.ones(len(t), dtype=bool)
        train_mask[val_idx] = False
        inside_embargo = (t >= v_start - margin) & (t <= v_end + margin)
        train_mask &= ~inside_embargo.to_numpy()
        train_idx = np.where(train_mask)[0]
        if len(train_idx) == 0:
            log.warning("fold %d has no training data after embargo; skipped", k)
            continue
        folds.append((train_idx, np.asarray(val_idx)))
    return folds


def make_models(feature_names: list[str], horizon_steps: int, seed: int,
                curves_dir: Path | None = None, lstm_overrides: dict | None = None) -> dict:
    from solarflare.forecast.baselines import (
        ClimatologyForecaster,
        EnsembleForecaster,
        HoltWintersForecaster,
    )
    from solarflare.forecast.lstm import LSTMConfig, LSTMForecaster

    def fresh(name: str):
        if name == "climatology":
            return ClimatologyForecaster()
        if name == "holt_winters":
            return HoltWintersForecaster(feature_names, horizon_steps=horizon_steps,
                                         seed=seed)
        if name == "lstm":
            cfg = LSTMConfig(seed=seed, **(lstm_overrides or {}))
            return LSTMForecaster(cfg, curves_dir=curves_dir, name=name)
        if name == "ensemble":
            cfg = LSTMConfig(seed=seed, **(lstm_overrides or {}))
            return EnsembleForecaster([
                HoltWintersForecaster(feature_names, horizon_steps=horizon_steps,
                                      seed=seed),
                LSTMForecaster(cfg, curves_dir=curves_dir, name="ensemble_lstm"),
            ])
        raise ValueError(f"unknown model {name!r}")

    from functools import partial

    return {name: partial(fresh, name)
            for name in ("climatology", "holt_winters", "lstm", "ensemble")}


def fit_model(factory, X_tr, y_tr, X_val, y_val):
    """Instantiate + fit; LSTM-family members get the val split for early stopping."""
    from solarflare.forecast.baselines import EnsembleForecaster
    from solarflare.forecast.lstm import LSTMForecaster

    model = factory()
    if isinstance(model, LSTMForecaster):
        model.fit(X_tr, y_tr, X_val, y_val)
    elif isinstance(model, EnsembleForecaster):
        for member in model.members:
            if isinstance(member, LSTMForecaster):
                member.fit(X_tr, y_tr, X_val, y_val)
            else:
                member.fit(X_tr, y_tr)
    else:
        model.fit(X_tr, y_tr)
    return model


def crossval_table(
    X: np.ndarray, y: np.ndarray, t0s: pd.Series, feature_names: list[str],
    horizon_steps: int, n_folds: int = 5, embargo_hours: float = 48.0,
    seed: int = 1337, model_names: tuple[str, ...] = ("climatology", "holt_winters",
                                                      "lstm", "ensemble"),
    curves_dir: Path | None = None, lstm_overrides: dict | None = None,
) -> tuple[pd.DataFrame, dict[str, tuple[np.ndarray, np.ndarray]]]:
    """Chronological k-fold CV.

    Returns (per-(model, fold) metric rows, out-of-fold {model: (y, p)} pairs
    concatenated across folds for reliability diagrams).
    """
    factories = make_models(feature_names, horizon_steps, seed, curves_dir,
                            lstm_overrides)
    rows = []
    oof: dict[str, list] = {name: [] for name in model_names}
    folds = time_blocked_folds(t0s, n_folds, embargo_hours)
    for k, (tr, va) in enumerate(folds):
        # inner chronological val split (20% tail of train) for early stop/threshold
        order = np.argsort(pd.to_datetime(t0s).iloc[tr].to_numpy(), kind="stable")
        tr_sorted = tr[order]
        cut = max(int(0.8 * len(tr_sorted)), 1)
        inner_tr, inner_va = tr_sorted[:cut], tr_sorted[cut:]
        if len(inner_va) == 0:
            inner_va = inner_tr[-max(len(inner_tr) // 5, 1):]
        base_rate = float(np.mean(y[inner_tr]))
        for name in model_names:
            fitted = fit_model(factories[name], X[inner_tr], y[inner_tr],
                               X[inner_va], y[inner_va])
            # operating threshold chosen on the inner val split, frozen for the fold
            thr, _ = best_tss_threshold(y[inner_va], fitted.predict_proba(X[inner_va]))
            p_fold = fitted.predict_proba(X[va])
            row = {"model": name, "fold": k,
                   **summarize(y[va], p_fold, thr, base_rate)}
            rows.append(row)
            oof[name].append((y[va], p_fold))
            log.info("fold %d %s: tss=%.3f hss=%.3f bss=%.3f", k, name,
                     row["tss"], row["hss"], row["bss"])
    oof_concat = {
        name: (np.concatenate([y_ for y_, _ in pairs]),
               np.concatenate([p_ for _, p_ in pairs]))
        for name, pairs in oof.items() if pairs
    }
    return pd.DataFrame(rows), oof_concat


def aggregate_table(per_fold: pd.DataFrame) -> pd.DataFrame:
    """mean +/- std per model over folds, ordered by mean TSS."""
    metrics = ["tss", "hss", "bss", "brier", "precision", "recall"]
    agg = per_fold.groupby("model")[metrics].agg(["mean", "std"])
    agg.columns = [f"{m}_{s}" for m, s in agg.columns]
    return agg.sort_values("tss_mean", ascending=False).reset_index()


def reliability_plot(
    results: dict[str, tuple[np.ndarray, np.ndarray]], out_path: str | Path,
    n_bins: int = 10,
) -> Path:
    """Reliability diagram for {model: (y_true, p)} on one shared axis."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax, axh) = plt.subplots(
        2, 1, figsize=(6, 7), height_ratios=[3, 1], constrained_layout=True)
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect")
    for name, (y, p) in results.items():
        centers, observed, counts = reliability_curve(y, p, n_bins)
        ok = counts > 0
        ax.plot(centers[ok], observed[ok], "o-", label=name)
        axh.semilogy(centers[ok], counts[ok], "o-", label=name)
    ax.set_xlabel("forecast probability")
    ax.set_ylabel("observed frequency")
    ax.legend()
    ax.set_title("Reliability diagram")
    axh.set_xlabel("forecast probability")
    axh.set_ylabel("count")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path
