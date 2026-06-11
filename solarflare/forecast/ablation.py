"""Channel/feature ablation: how much does each input contribute to TSS?

Primary method: grouped PERMUTATION importance — shuffle one feature group
across samples (keeping each sequence's time structure intact) and measure the
TSS drop at the frozen operating threshold. Model-agnostic, needs one trained
model, and is the statistically honest option on CPU budgets.

Secondary: drop-one RETRAIN for a few focus channels (e.g. AIA 131/304 and the
magnetogram, per the confined-vs-eruptive question) — more faithful but one
full training per group.

A feature group bundles a base feature with its derived columns (e.g.
`aia_0131_max` + `aia_0131_max_d1`), so ablating a channel removes ALL its
information, not just the undifferenced copy.
"""

from __future__ import annotations

import logging
import re

import numpy as np
import pandas as pd

from solarflare.eval.metrics import tss

log = logging.getLogger(__name__)

_DERIVED_SUFFIXES = ("_d1",)


def feature_groups(feature_names: list[str]) -> dict[str, list[int]]:
    """Group columns by base feature (gradients join their parent)."""
    groups: dict[str, list[int]] = {}
    for idx, name in enumerate(feature_names):
        base = name
        for suffix in _DERIVED_SUFFIXES:
            base = base.removesuffix(suffix)
        base = re.sub(r"_max$", "", base)
        groups.setdefault(base, []).append(idx)
    return groups


def permutation_importance(
    model,
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    threshold: float,
    groups: dict[str, list[int]] | None = None,
    n_repeats: int = 3,
    seed: int = 1337,
) -> pd.DataFrame:
    """Grouped permutation importance on a fixed evaluation set.

    delta_tss > 0 means the group carries information the model uses.
    """
    groups = groups or feature_groups(feature_names)
    rng = np.random.default_rng(seed)
    p_base = model.predict_proba(X)
    tss_base = tss(y, p_base >= threshold)
    rows = []
    for group_name, idxs in groups.items():
        deltas = []
        for _ in range(n_repeats):
            perm = rng.permutation(len(X))
            X_perm = X.copy()
            X_perm[:, :, idxs] = X[perm][:, :, idxs]
            p = model.predict_proba(X_perm)
            deltas.append(tss_base - tss(y, p >= threshold))
        rows.append({
            "group": group_name,
            "n_columns": len(idxs),
            "tss_base": tss_base,
            "delta_tss_mean": float(np.mean(deltas)),
            "delta_tss_std": float(np.std(deltas)),
        })
    table = (pd.DataFrame(rows)
             .sort_values("delta_tss_mean", ascending=False)
             .reset_index(drop=True))
    log.info("permutation importance over %d groups (base TSS %.3f)",
             len(groups), tss_base)
    return table


def drop_one_retrain(
    train_fn,
    X_train: np.ndarray, y_train: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
    X_eval: np.ndarray, y_eval: np.ndarray,
    feature_names: list[str],
    focus_groups: list[str],
    seed: int = 1337,
) -> pd.DataFrame:
    """Retrain with each focus group removed; report TSS deltas vs full model.

    `train_fn(X_tr, y_tr, X_va, y_va) -> model with predict_proba` must pick
    its own threshold internals; here the threshold is re-chosen on val for
    each retrain (each model gets its fair operating point).
    """
    from solarflare.eval.metrics import best_tss_threshold

    groups = feature_groups(feature_names)
    unknown = [g for g in focus_groups if g not in groups]
    if unknown:
        raise ValueError(f"unknown focus groups {unknown}; have {sorted(groups)}")

    def fit_and_score(keep_idxs: list[int]) -> float:
        model = train_fn(X_train[:, :, keep_idxs], y_train,
                         X_val[:, :, keep_idxs], y_val)
        thr, _ = best_tss_threshold(y_val, model.predict_proba(X_val[:, :, keep_idxs]))
        p = model.predict_proba(X_eval[:, :, keep_idxs])
        return tss(y_eval, p >= thr)

    all_idxs = list(range(len(feature_names)))
    tss_full = fit_and_score(all_idxs)
    rows = [{"group": "<full model>", "tss": tss_full, "delta_tss": 0.0}]
    for group_name in focus_groups:
        keep = [i for i in all_idxs if i not in groups[group_name]]
        score = fit_and_score(keep)
        rows.append({"group": group_name, "tss": score,
                     "delta_tss": tss_full - score})
        log.info("drop-one %s: tss=%.3f (delta %.3f)", group_name, score,
                 tss_full - score)
    return pd.DataFrame(rows)


def ablation_bar_chart(table: pd.DataFrame, out_path, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, max(3, 0.35 * len(table))),
                           constrained_layout=True)
    data = table.sort_values("delta_tss_mean")
    ax.barh(data["group"], data["delta_tss_mean"],
            xerr=data.get("delta_tss_std"), color="tab:blue")
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("TSS drop when permuted (importance)")
    ax.set_title(title)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
