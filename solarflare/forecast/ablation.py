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
        rows.append(
            {
                "group": group_name,
                "n_columns": len(idxs),
                "tss_base": tss_base,
                "delta_tss_mean": float(np.mean(deltas)),
                "delta_tss_std": float(np.std(deltas)),
            }
        )
    table = pd.DataFrame(rows).sort_values("delta_tss_mean", ascending=False).reset_index(drop=True)
    log.info("permutation importance over %d groups (base TSS %.3f)", len(groups), tss_base)
    return table


def drop_one_retrain(
    train_fn,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_eval: np.ndarray,
    y_eval: np.ndarray,
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
        model = train_fn(X_train[:, :, keep_idxs], y_train, X_val[:, :, keep_idxs], y_val)
        thr, _ = best_tss_threshold(y_val, model.predict_proba(X_val[:, :, keep_idxs]))
        p = model.predict_proba(X_eval[:, :, keep_idxs])
        return tss(y_eval, p >= thr)

    all_idxs = list(range(len(feature_names)))
    tss_full = fit_and_score(all_idxs)
    rows = [{"group": "<full model>", "tss": tss_full, "delta_tss": 0.0}]
    for group_name in focus_groups:
        keep = [i for i in all_idxs if i not in groups[group_name]]
        score = fit_and_score(keep)
        rows.append({"group": group_name, "tss": score, "delta_tss": tss_full - score})
        log.info("drop-one %s: tss=%.3f (delta %.3f)", group_name, score, tss_full - score)
    return pd.DataFrame(rows)


def ablation_bar_chart(table: pd.DataFrame, out_path, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, max(3, 0.35 * len(table))), constrained_layout=True)
    data = table.sort_values("delta_tss_mean")
    ax.barh(data["group"], data["delta_tss_mean"], xerr=data.get("delta_tss_std"), color="tab:blue")
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("TSS drop when permuted (importance)")
    ax.set_title(title)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


# --- atmospheric-layer combination ablation (the proposal's 6-case matrix) ---
#: HMI (the magnetic root) is ALWAYS included; each case adds a set of AIA layers.
#: "all" resolves to every AIA channel present in the dataset (full spectrum).
LAYER_CASES: dict[str, tuple[int, ...] | str] = {
    "case1_baseline_surface": (),  # HMI magnetogram only
    "case2_low_atmosphere": (1600, 304),  # photosphere -> chromosphere
    "case3_quiet_corona": (171, 193, 211),  # coronal loops / dimming
    "case4_flaring_corona": (94, 131),  # super-heated flare plasma
    "case5_core_synergy": (171, 94),  # surface + loop + thermal flash
    "case6_full_spectrum": "all",  # HMI + all AIA channels
}


def available_aia_channels(feature_names: list[str]) -> list[int]:
    """AIA channels present in the feature matrix, parsed from `aia_{C}_max` names."""
    seen = set()
    for name in feature_names:
        if name.startswith("aia_"):
            seen.add(int(name.split("_", 2)[1]))
    return sorted(seen)


def case_feature_indices(feature_names: list[str], channels) -> list[int]:
    """Column indices for one case: every magnetic/area feature (always on, the
    HMI root) plus the `aia_{C}_*` columns for C in `channels` (gradients follow)."""
    keep_aia = {f"aia_{int(c):04d}" for c in channels}
    idxs = []
    for j, name in enumerate(feature_names):
        if name.startswith("aia_"):
            if name.split("_max", 1)[0] in keep_aia:
                idxs.append(j)
        else:
            idxs.append(j)  # flux_total/signed_flux/b_peak/area_px (+gradients)
    return idxs


def resolve_case_channels(value, feature_names: list[str]) -> list[int]:
    """Resolve a LAYER_CASES value to the channels actually present in the data."""
    available = available_aia_channels(feature_names)
    if value == "all":
        return available
    return [c for c in value if c in available]


def ablate_layers(
    X: np.ndarray,
    samples: pd.DataFrame,
    feature_names: list[str],
    cases: dict[str, tuple[int, ...] | str] | None = None,
    n_folds: int = 3,
    embargo_hours: float = 48.0,
    seed: int = 1337,
    model_names: tuple[str, ...] = ("holt_winters", "lstm"),
    lstm_overrides: dict | None = None,
) -> tuple[pd.DataFrame, dict[str, dict]]:
    """Score each atmospheric-layer case on the dataset's {horizon x class} grid.

    For every case the feature matrix is dynamically masked to its channel subset
    (no re-fetch), then `crossval_grid` scores TSS/HSS/BSS per cell. Returns
    (combined table tagged with `case` + `n_aia_channels`, {case: per-cell oof}).
    """
    from solarflare.forecast.validate import crossval_grid

    cases = cases or LAYER_CASES
    tables, oof_by_case = [], {}
    for case_name, value in cases.items():
        channels = resolve_case_channels(value, feature_names)
        idxs = case_feature_indices(feature_names, channels)
        Xc = X[:, :, idxs]
        names_c = [feature_names[i] for i in idxs]
        tbl, oof = crossval_grid(
            Xc,
            samples,
            names_c,
            horizon_steps=Xc.shape[1],
            n_folds=n_folds,
            embargo_hours=embargo_hours,
            seed=seed,
            model_names=model_names,
            lstm_overrides=lstm_overrides,
        )
        if tbl.empty:
            log.warning("layer case %s produced no usable cells; skipped", case_name)
            continue
        tbl.insert(0, "n_aia_channels", len(channels))
        tbl.insert(0, "case", case_name)
        tables.append(tbl)
        oof_by_case[case_name] = oof
        log.info("layer case %s (%d AIA ch): %d rows", case_name, len(channels), len(tbl))
    combined = pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()
    return combined, oof_by_case


def layer_case_bar_chart(
    table: pd.DataFrame, out_path, model: str, label_col: str, title: str | None = None
) -> None:
    """TSS-by-case bar chart for one (model, {horizon x class} cell), case order kept."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sub = table[(table["model"] == model) & (table["label_col"] == label_col)].copy()
    order = [c for c in LAYER_CASES if c in set(sub["case"])]
    sub["case"] = pd.Categorical(sub["case"], categories=order, ordered=True)
    sub = sub.sort_values("case")
    fig, ax = plt.subplots(figsize=(8, max(3, 0.5 * len(sub))), constrained_layout=True)
    ax.barh(sub["case"].astype(str), sub["tss_mean"], xerr=sub.get("tss_std"), color="tab:green")
    ax.invert_yaxis()
    ax.set_xlabel("TSS (mean over folds)")
    ax.set_title(title or f"Atmospheric-layer ablation — {model}, {label_col}")
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
