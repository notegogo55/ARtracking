"""Forecast verification metrics. TSS is the primary metric (master prompt §7).

All binary metrics take y_true in {0,1} and a binary prediction; probabilistic
metrics take probabilities. Threshold selection (`best_tss_threshold`) must be
done on VALIDATION data and then applied frozen to test data — never tuned on
the split being reported.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Contingency:
    tp: int
    fp: int
    tn: int
    fn: int


def contingency(y_true: np.ndarray, y_bin: np.ndarray) -> Contingency:
    y = np.asarray(y_true).astype(bool)
    p = np.asarray(y_bin).astype(bool)
    return Contingency(
        tp=int(np.sum(p & y)),
        fp=int(np.sum(p & ~y)),
        tn=int(np.sum(~p & ~y)),
        fn=int(np.sum(~p & y)),
    )


def tss(y_true: np.ndarray, y_bin: np.ndarray) -> float:
    """True Skill Statistic = POD - POFD = TP/(TP+FN) - FP/(FP+TN). Range [-1, 1]."""
    c = contingency(y_true, y_bin)
    pod = c.tp / (c.tp + c.fn) if (c.tp + c.fn) else 0.0
    pofd = c.fp / (c.fp + c.tn) if (c.fp + c.tn) else 0.0
    return float(pod - pofd)


def hss(y_true: np.ndarray, y_bin: np.ndarray) -> float:
    """Heidke Skill Score (vs random chance)."""
    c = contingency(y_true, y_bin)
    num = 2.0 * (c.tp * c.tn - c.fn * c.fp)
    den = (c.tp + c.fn) * (c.fn + c.tn) + (c.tp + c.fp) * (c.fp + c.tn)
    return float(num / den) if den else 0.0


def brier(y_true: np.ndarray, p: np.ndarray) -> float:
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(p, dtype=float)
    return float(np.mean((p - y) ** 2))


def bss(y_true: np.ndarray, p: np.ndarray, climatology_rate: float) -> float:
    """Brier Skill Score vs the TRAIN-period climatological rate."""
    bs = brier(y_true, p)
    bs_ref = brier(y_true, np.full_like(np.asarray(y_true, dtype=float),
                                        climatology_rate))
    return float(1.0 - bs / bs_ref) if bs_ref > 0 else 0.0


def precision_recall(y_true: np.ndarray, y_bin: np.ndarray) -> tuple[float, float]:
    c = contingency(y_true, y_bin)
    precision = c.tp / (c.tp + c.fp) if (c.tp + c.fp) else 0.0
    recall = c.tp / (c.tp + c.fn) if (c.tp + c.fn) else 0.0
    return float(precision), float(recall)


def best_tss_threshold(y_true: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    """(threshold, TSS) maximizing TSS over the candidate probability values.

    Use on a VALIDATION split only; apply the frozen threshold elsewhere.
    """
    p = np.asarray(p, dtype=float)
    candidates = np.unique(np.concatenate([[0.0, 0.5, 1.0], p]))
    best_t, best_s = 0.5, -np.inf
    for t in candidates:
        s = tss(y_true, p >= t)
        if s > best_s:
            best_t, best_s = float(t), float(s)
    return best_t, best_s


def reliability_curve(
    y_true: np.ndarray, p: np.ndarray, n_bins: int = 10
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(bin centers, observed frequency, counts) for a reliability diagram.

    Empty bins return NaN observed frequency.
    """
    p = np.asarray(p, dtype=float)
    y = np.asarray(y_true, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    observed = np.full(n_bins, np.nan)
    counts = np.zeros(n_bins, dtype=int)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)
    for b in range(n_bins):
        sel = idx == b
        counts[b] = int(sel.sum())
        if counts[b]:
            observed[b] = float(y[sel].mean())
    return centers, observed, counts


def summarize(
    y_true: np.ndarray, p: np.ndarray, threshold: float, climatology_rate: float
) -> dict:
    """One row of the metrics table for a fixed, externally chosen threshold."""
    y_bin = np.asarray(p, dtype=float) >= threshold
    precision, recall = precision_recall(y_true, y_bin)
    return {
        "n": int(len(np.asarray(y_true))),
        "base_rate": float(np.mean(y_true)),
        "threshold": float(threshold),
        "tss": tss(y_true, y_bin),
        "hss": hss(y_true, y_bin),
        "bss": bss(y_true, p, climatology_rate),
        "brier": brier(y_true, p),
        "precision": precision,
        "recall": recall,
    }
