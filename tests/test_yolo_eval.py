"""Detector-vs-SHARP IoU evaluation metrics (pure; no model, no network)."""

import pandas as pd
import pytest

from solarflare.detect.yolo import evaluate_vs_truth


def _truth() -> pd.DataFrame:
    return pd.DataFrame([
        {"image": "a.png", "x_min": 10, "x_max": 50, "y_min": 10, "y_max": 50},
        {"image": "a.png", "x_min": 100, "x_max": 140, "y_min": 100, "y_max": 140},
        {"image": "b.png", "x_min": 20, "x_max": 60, "y_min": 20, "y_max": 60},
    ])


def test_perfect_predictions():
    preds = _truth().assign(conf=0.9)
    m = evaluate_vs_truth(preds, _truth())
    assert m["recall"] == 1.0 and m["precision"] == 1.0
    assert m["mean_matched_iou"] == pytest.approx(1.0)


def test_partial_overlap_and_misses():
    preds = pd.DataFrame([
        # shifted version of truth box 1 (IoU ~ 0.6)
        {"image": "a.png", "x_min": 15, "x_max": 55, "y_min": 10, "y_max": 50, "conf": 0.8},
        # false positive far away
        {"image": "a.png", "x_min": 300, "x_max": 340, "y_min": 300, "y_max": 340, "conf": 0.7},
    ])
    m = evaluate_vs_truth(preds, _truth(), iou_match=0.5)
    assert m["n_matched"] == 1
    assert m["recall"] == pytest.approx(1 / 3)
    assert m["precision"] == pytest.approx(1 / 2)
    assert 0.5 < m["mean_matched_iou"] < 0.8


def test_no_predictions():
    preds = pd.DataFrame(columns=["image", "x_min", "x_max", "y_min", "y_max", "conf"])
    m = evaluate_vs_truth(preds, _truth())
    assert m["recall"] == 0.0 and m["n_matched"] == 0


def test_one_pred_cannot_match_twice():
    preds = pd.DataFrame([
        {"image": "a.png", "x_min": 10, "x_max": 140, "y_min": 10, "y_max": 140, "conf": 0.9},
    ])
    m = evaluate_vs_truth(preds, _truth(), iou_match=0.05)
    assert m["n_matched"] == 1
