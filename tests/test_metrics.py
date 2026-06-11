"""Verification metrics against hand-computed values."""

import numpy as np
import pytest

from solarflare.eval.metrics import (
    best_tss_threshold,
    brier,
    bss,
    contingency,
    hss,
    reliability_curve,
    summarize,
    tss,
)

Y = np.array([1, 1, 0, 0, 1])
P_BIN = np.array([1, 0, 0, 1, 1])  # TP=2 FN=1 FP=1 TN=1


def test_contingency():
    c = contingency(Y, P_BIN)
    assert (c.tp, c.fn, c.fp, c.tn) == (2, 1, 1, 1)


def test_tss_hand_computed():
    # POD = 2/3, POFD = 1/2 -> TSS = 1/6
    assert tss(Y, P_BIN) == pytest.approx(2 / 3 - 1 / 2)


def test_tss_perfect_and_inverted():
    assert tss(Y, Y) == 1.0
    assert tss(Y, 1 - Y) == -1.0
    assert tss(Y, np.ones_like(Y)) == 0.0  # always-yes has no skill


def test_hss_hand_computed():
    # 2*(2*1 - 1*1) / ((3)(2) + (3)(2)) = 2/12
    assert hss(Y, P_BIN) == pytest.approx(2 * (2 * 1 - 1 * 1) / 12)


def test_brier_and_bss():
    y = np.array([1, 0, 1, 0])
    p = np.array([0.8, 0.2, 0.6, 0.4])
    assert brier(y, p) == pytest.approx((0.04 + 0.04 + 0.16 + 0.16) / 4)
    # vs climatology 0.5: BS_ref = 0.25
    assert bss(y, p, 0.5) == pytest.approx(1 - 0.1 / 0.25)
    assert bss(y, np.full(4, 0.5), 0.5) == pytest.approx(0.0)


def test_best_tss_threshold_separable():
    y = np.array([0, 0, 0, 1, 1, 1])
    p = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    thr, score = best_tss_threshold(y, p)
    assert score == 1.0
    assert 0.3 < thr <= 0.7


def test_reliability_curve_bins():
    y = np.array([0, 0, 1, 1, 1, 1])
    p = np.array([0.05, 0.05, 0.95, 0.95, 0.95, 0.55])
    centers, observed, counts = reliability_curve(y, p, n_bins=10)
    assert counts.sum() == 6
    assert observed[0] == pytest.approx(0.0)      # low-prob bin: all negative
    assert observed[9] == pytest.approx(1.0)      # high-prob bin: all positive
    assert np.isnan(observed[2])                  # empty bin


def test_summarize_keys():
    out = summarize(Y, np.array([0.9, 0.1, 0.2, 0.8, 0.7]), 0.5, 0.4)
    assert {"tss", "hss", "bss", "brier", "precision", "recall",
            "base_rate", "threshold", "n"} <= set(out)
