"""Temporal-IoU tracking of AR boxes with differential-rotation compensation.

Boxes live in Stonyhurst heliographic coordinates (deg), where an AR's only
systematic motion is the latitude-dependent synodic rotation drift. Before
matching a track to candidate boxes at a later time, the track's last box is
shifted in longitude by sunpy's Howard differential-rotation model, so IoU
matching is rotation-free. Persistent integer track IDs; HARP numbers are
attached by majority vote when the source boxes carry them (fallback per spec).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

BOX_COLS = ["lon_min", "lon_max", "lat_min", "lat_max"]


def box_iou(a, b) -> float:
    """IoU of two axis-aligned (lon, lat) boxes given as 4-sequences (BOX_COLS order)."""
    ix = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[2], b[2]))
    inter = ix * iy
    if inter <= 0:
        return 0.0
    area_a = (a[1] - a[0]) * (a[3] - a[2])
    area_b = (b[1] - b[0]) * (b[3] - b[2])
    return float(inter / (area_a + area_b - inter))


def rotation_shift_deg(dt_days: float, latitude_deg: float) -> float:
    """Stonyhurst longitude drift over dt (Howard model, synodic frame)."""
    import astropy.units as u
    from sunpy.sun.models import differential_rotation

    shift = differential_rotation(
        dt_days * u.day, latitude_deg * u.deg, model="howard", frame_time="synodic"
    )
    return float(shift.to_value(u.deg))


def compensate_box(box, dt_days: float):
    """Shift a (lon_min, lon_max, lat_min, lat_max) box by the rotation drift."""
    lat_center = 0.5 * (box[2] + box[3])
    shift = rotation_shift_deg(dt_days, lat_center)
    return (box[0] + shift, box[1] + shift, box[2], box[3])


@dataclass
class _Track:
    track_id: int
    last_time: pd.Timestamp
    last_box: tuple
    harpnums: list = field(default_factory=list)


def track_boxes(
    boxes: pd.DataFrame,
    iou_threshold: float = 0.2,
    max_gap_frames: int = 2,
) -> pd.DataFrame:
    """Assign persistent track_id to a time-ordered box table (BOX_COLS [+ harpnum]).

    Greedy best-IoU association per timestep on rotation-compensated boxes.
    Gap tolerance is measured in TIME (max_gap_frames x the median cadence of
    the table), so wholly missing frames count toward the gap; older tracks are
    retired. Returns a copy with track_id and harp_id (majority HARP, or -1).
    """
    if boxes.empty:
        out = boxes.copy()
        out["track_id"] = pd.Series(dtype=int)
        out["harp_id"] = pd.Series(dtype=int)
        return out

    df = boxes.sort_values("time").reset_index(drop=True).copy()
    df["track_id"] = -1
    has_harp = "harpnum" in df.columns

    unique_times = df["time"].drop_duplicates().sort_values()
    if len(unique_times) > 1:
        cadence_h = float(unique_times.diff().dropna().median().total_seconds()) / 3600.0
        allowed_gap_h = (max_gap_frames + 1) * cadence_h * 1.001
    else:
        allowed_gap_h = float("inf")

    active: list[_Track] = []
    next_id = 0
    for time, idx in df.groupby("time", sort=True).groups.items():
        idx = list(idx)
        # retire tracks whose last observation is too old to bridge
        active = [
            tr for tr in active
            if (pd.Timestamp(time) - tr.last_time).total_seconds() / 3600.0 <= allowed_gap_h
        ]
        cand_boxes = [tuple(df.loc[i, BOX_COLS]) for i in idx]
        # IoU of every active track (rotation-compensated) vs every candidate
        pairs = []
        for ti, tr in enumerate(active):
            dt_days = (pd.Timestamp(time) - tr.last_time).total_seconds() / 86400.0
            comp = compensate_box(tr.last_box, dt_days)
            for ci, cb in enumerate(cand_boxes):
                iou = box_iou(comp, cb)
                if iou >= iou_threshold:
                    pairs.append((iou, ti, ci))
        pairs.sort(reverse=True)
        used_tracks: set[int] = set()
        used_cands: set[int] = set()
        for _iou, ti, ci in pairs:
            if ti in used_tracks or ci in used_cands:
                continue
            used_tracks.add(ti)
            used_cands.add(ci)
            tr = active[ti]
            i = idx[ci]
            df.loc[i, "track_id"] = tr.track_id
            tr.last_time = pd.Timestamp(time)
            tr.last_box = cand_boxes[ci]
            if has_harp and pd.notna(df.loc[i, "harpnum"]):
                tr.harpnums.append(int(df.loc[i, "harpnum"]))
        # new tracks for unmatched candidates
        for ci, i in enumerate(idx):
            if ci in used_cands:
                continue
            tr = _Track(next_id, pd.Timestamp(time), cand_boxes[ci])
            if has_harp and pd.notna(df.loc[i, "harpnum"]):
                tr.harpnums.append(int(df.loc[i, "harpnum"]))
            df.loc[i, "track_id"] = tr.track_id
            active.append(tr)
            next_id += 1

    # majority-vote HARP attachment per track
    harp_for_track: dict[int, int] = {}
    if has_harp:
        for tid, grp in df[df["harpnum"].notna()].groupby("track_id"):
            harp_for_track[int(tid)] = int(grp["harpnum"].mode().iloc[0])
    df["harp_id"] = df["track_id"].map(harp_for_track).fillna(-1).astype(int)
    return df


def track_report(tracked: pd.DataFrame) -> pd.DataFrame:
    """Per-track stability summary: span, length, consecutive compensated IoU, purity."""
    rows = []
    for tid, grp in tracked.groupby("track_id"):
        grp = grp.sort_values("time")
        ious = []
        prev = None
        for _, row in grp.iterrows():
            box = tuple(row[BOX_COLS])
            if prev is not None:
                dt_days = (row["time"] - prev[0]).total_seconds() / 86400.0
                ious.append(box_iou(compensate_box(prev[1], dt_days), box))
            prev = (row["time"], box)
        harp_purity = np.nan
        if "harpnum" in grp.columns and grp["harpnum"].notna().any():
            counts = grp["harpnum"].value_counts()
            harp_purity = float(counts.iloc[0] / counts.sum())
        rows.append({
            "track_id": int(tid),
            "harp_id": int(grp["harp_id"].iloc[0]) if "harp_id" in grp else -1,
            "n_obs": len(grp),
            "t_start": grp["time"].iloc[0],
            "t_end": grp["time"].iloc[-1],
            "mean_consec_iou": float(np.mean(ious)) if ious else np.nan,
            "min_consec_iou": float(np.min(ious)) if ious else np.nan,
            "harp_purity": harp_purity,
        })
    return pd.DataFrame(rows).sort_values("n_obs", ascending=False).reset_index(drop=True)
