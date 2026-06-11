"""Ultralytics YOLO fine-tuning + inference + IoU evaluation vs SHARP boxes.

Transfer learning only: starts from a pretrained nano checkpoint, single class.
Evaluation is IoU against the HARP-bootstrapped boxes on window-held-out splits
(the Gate G2 number), alongside Ultralytics' own mAP metrics.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from solarflare.track.iou import box_iou

log = logging.getLogger(__name__)


def train_detector(
    cfg,
    dataset_yaml: str | Path,
    out_dir: str | Path,
    epochs: int | None = None,
    device: str | None = None,
) -> Path:
    """Fine-tune the configured pretrained YOLO model; returns best-weights path."""
    from ultralytics import YOLO

    model = YOLO(cfg.detect.yolo_model)  # pretrained checkpoint (auto-downloaded)
    results = model.train(
        data=str(dataset_yaml),
        epochs=epochs or cfg.detect.yolo_epochs,
        imgsz=cfg.detect.yolo_imgsz,
        seed=cfg.project.seed,
        deterministic=True,
        single_cls=True,
        project=str(out_dir),
        name="ar_yolo",
        exist_ok=True,
        device=device,
        plots=True,
    )
    best = Path(results.save_dir) / "weights" / "best.pt"
    if not best.exists():
        raise RuntimeError(f"training finished but {best} is missing")
    log.info("trained weights: %s", best)
    return best


def predict_boxes(
    weights: str | Path, image_paths: list[Path], conf: float = 0.25
) -> pd.DataFrame:
    """Run inference; returns [image, x_min, x_max, y_min, y_max, conf] in pixels."""
    from ultralytics import YOLO

    model = YOLO(str(weights))
    rows = []
    for result in model.predict([str(p) for p in image_paths], conf=conf,
                                verbose=False, stream=True):
        image = Path(result.path).name
        for xyxy, score in zip(result.boxes.xyxy.tolist(),
                               result.boxes.conf.tolist(), strict=True):
            x0, y0, x1, y1 = xyxy
            rows.append({"image": image, "x_min": x0, "x_max": x1,
                         "y_min": y0, "y_max": y1, "conf": float(score)})
    return pd.DataFrame(rows, columns=["image", "x_min", "x_max", "y_min", "y_max", "conf"])


def evaluate_vs_truth(
    predictions: pd.DataFrame, truth: pd.DataFrame, iou_match: float = 0.5
) -> dict:
    """Greedy per-image matching of predictions to truth boxes; Gate G2 metrics.

    Returns recall/precision at the IoU threshold plus the IoU distribution of
    matched pairs (mean/median) — 'IoU vs the SHARP-derived boxes'.
    """
    matched_ious: list[float] = []
    n_truth = len(truth)
    n_pred = len(predictions)
    n_matched = 0
    images = set(truth["image"]) | set(predictions["image"]) if n_pred else set(truth["image"])
    for image in images:
        t_boxes = [tuple(r) for r in
                   truth.loc[truth["image"] == image,
                             ["x_min", "x_max", "y_min", "y_max"]].to_numpy()]
        p_rows = predictions[predictions["image"] == image] if n_pred else predictions
        p_boxes = [tuple(r) for r in
                   p_rows[["x_min", "x_max", "y_min", "y_max"]].to_numpy()] if len(p_rows) else []
        pairs = sorted(
            ((box_iou(t, p), ti, pi) for ti, t in enumerate(t_boxes)
             for pi, p in enumerate(p_boxes)),
            reverse=True,
        )
        used_t: set[int] = set()
        used_p: set[int] = set()
        for iou, ti, pi in pairs:
            if iou < iou_match or ti in used_t or pi in used_p:
                continue
            used_t.add(ti)
            used_p.add(pi)
            matched_ious.append(iou)
            n_matched += 1
    return {
        "n_truth": n_truth,
        "n_pred": n_pred,
        "n_matched": n_matched,
        "recall": n_matched / n_truth if n_truth else np.nan,
        "precision": n_matched / n_pred if n_pred else np.nan,
        "mean_matched_iou": float(np.mean(matched_ious)) if matched_ious else np.nan,
        "median_matched_iou": float(np.median(matched_ious)) if matched_ious else np.nan,
        "iou_match_threshold": iou_match,
    }
