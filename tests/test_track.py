"""Temporal-IoU tracker on synthetic drifting boxes (offline; sunpy rotation model)."""

import numpy as np
import pandas as pd
import pytest

from solarflare.track.iou import (
    box_iou,
    compensate_box,
    rotation_shift_deg,
    track_boxes,
    track_report,
)


def test_box_iou_basics():
    a = (0, 10, 0, 10)
    assert box_iou(a, a) == pytest.approx(1.0)
    assert box_iou(a, (10, 20, 0, 10)) == 0.0          # touching edges
    assert box_iou(a, (5, 15, 0, 10)) == pytest.approx(5 * 10 / 150)
    assert box_iou(a, (20, 30, 20, 30)) == 0.0


def test_rotation_shift_sign_and_magnitude():
    # Synodic Howard drift at low latitude: ~13-14 deg/day, westward (positive lon)
    shift = rotation_shift_deg(1.0, 20.0)
    assert 12.0 < shift < 15.0
    assert rotation_shift_deg(0.0, 20.0) == pytest.approx(0.0)
    # higher latitude rotates slower
    assert rotation_shift_deg(1.0, 60.0) < shift


def _drifting_boxes(lat: float, lon0: float, times: pd.DatetimeIndex,
                    harpnum: int | None = None) -> pd.DataFrame:
    """Boxes drifting at the Howard rate (what real Stonyhurst HARP boxes do)."""
    rows = []
    t0 = times[0]
    for t in times:
        dt_days = (t - t0).total_seconds() / 86400.0
        lon = lon0 + rotation_shift_deg(dt_days, lat)
        row = {"time": t, "lon_min": lon - 5, "lon_max": lon + 5,
               "lat_min": lat - 4, "lat_max": lat + 4}
        if harpnum is not None:
            row["harpnum"] = harpnum
        rows.append(row)
    return pd.DataFrame(rows)


def test_single_ar_single_track():
    times = pd.date_range("2099-01-01", periods=12, freq="6h")
    tracked = track_boxes(_drifting_boxes(-20.0, -30.0, times, harpnum=11))
    assert tracked["track_id"].nunique() == 1
    assert (tracked["harp_id"] == 11).all()


def test_two_ars_two_stable_tracks():
    times = pd.date_range("2099-01-01", periods=12, freq="6h")
    boxes = pd.concat([
        _drifting_boxes(-20.0, -30.0, times, harpnum=11),
        _drifting_boxes(15.0, 10.0, times, harpnum=22),
    ]).sort_values("time").reset_index(drop=True)
    tracked = track_boxes(boxes)
    assert tracked["track_id"].nunique() == 2
    # no ID switches: each track maps to exactly one harpnum
    purity = tracked.groupby("track_id")["harpnum"].nunique()
    assert (purity == 1).all()
    report = track_report(tracked)
    assert (report["mean_consec_iou"] > 0.8).all()


def test_gap_bridging_and_retirement():
    times = pd.date_range("2099-01-01", periods=10, freq="6h")
    boxes = _drifting_boxes(-20.0, -30.0, times, harpnum=11)
    # drop two consecutive observations (within max_gap_frames=2)
    boxes_gap = boxes.drop(index=[4, 5]).reset_index(drop=True)
    tracked = track_boxes(boxes_gap, max_gap_frames=2)
    assert tracked["track_id"].nunique() == 1
    # with zero tolerance the same gap splits the track
    tracked_strict = track_boxes(boxes_gap, max_gap_frames=0)
    assert tracked_strict["track_id"].nunique() == 2


def test_compensation_is_what_makes_it_work():
    # 12 h steps at lat -20: drift ~6.7 deg/step vs 10-deg-wide boxes.
    times = pd.date_range("2099-01-01", periods=8, freq="12h")
    boxes = _drifting_boxes(-20.0, -30.0, times)
    tracked = track_boxes(boxes, iou_threshold=0.5)
    assert tracked["track_id"].nunique() == 1  # compensated: one clean track
    # sanity: uncompensated IoU across one step is poor vs compensated
    b0 = tuple(boxes.loc[0, ["lon_min", "lon_max", "lat_min", "lat_max"]])
    b1 = tuple(boxes.loc[1, ["lon_min", "lon_max", "lat_min", "lat_max"]])
    assert box_iou(b0, b1) < 0.55
    assert box_iou(compensate_box(b0, 0.5), b1) > 0.95


def test_empty_input():
    out = track_boxes(pd.DataFrame(columns=["time", "lon_min", "lon_max",
                                            "lat_min", "lat_max"]))
    assert out.empty and "track_id" in out.columns


def test_harp_majority_attachment():
    times = pd.date_range("2099-01-01", periods=6, freq="6h")
    boxes = _drifting_boxes(-20.0, -30.0, times, harpnum=11)
    boxes.loc[3, "harpnum"] = np.nan  # one missing attribution
    tracked = track_boxes(boxes)
    assert (tracked["harp_id"] == 11).all()
