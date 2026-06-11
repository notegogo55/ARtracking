"""Climatological base-rate baseline — the first number TSS must beat.

MVP definition (full-Sun, day-level): the fraction of fixed-length time bins in
the study period containing at least one flare >= the class threshold. This is
the climatological probability p; forecasting the constant p has TSS = 0 and
Brier score p*(1-p), which later models must beat (TSS > 0, BSS > 0).

Phase D refines this to a per-AR, per-sample base rate on the real sample frame.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta

import pandas as pd

from solarflare.data.goes_events import goes_class_to_flux

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class BaseRateResult:
    n_bins: int
    n_positive_bins: int
    n_events: int
    bin_hours: float
    min_class: str
    base_rate: float
    brier_climatology: float

    def summary(self) -> str:
        return (
            f"Climatological base rate (>= {self.min_class}, {self.bin_hours:g} h bins): "
            f"{self.base_rate:.4f}  "
            f"[{self.n_positive_bins}/{self.n_bins} bins positive, {self.n_events} events]\n"
            f"Constant-climatology forecast: TSS = 0.0 (the bar to beat), "
            f"Brier = {self.brier_climatology:.4f} (BSS reference)"
        )


def compute_base_rate(
    events: pd.DataFrame,
    windows: list[tuple[datetime, datetime]],
    bin_hours: float = 24.0,
    min_class: str = "M1.0",
) -> BaseRateResult:
    """Compute the >= `min_class` base rate over `windows` using `bin_hours` bins.

    `events` needs columns `peak_time` (datetime-like) and `goes_class`.
    Bins are anchored at each window's start; a trailing partial bin is dropped.
    A bin is positive if any qualifying event peaks within [bin_start, bin_end).
    """
    if not windows:
        raise ValueError("at least one (start, end) window is required")
    min_flux = goes_class_to_flux(min_class)
    peaks = pd.to_datetime(events["peak_time"])
    qualifies = events["goes_class"].map(goes_class_to_flux) >= min_flux

    n_bins = 0
    n_positive = 0
    n_events = 0
    step = timedelta(hours=bin_hours)
    for start, end in windows:
        if end <= start:
            raise ValueError(f"window end {end} not after start {start}")
        in_window = qualifies & (peaks >= start) & (peaks < end)
        n_events += int(in_window.sum())
        bin_start = start
        while bin_start + step <= end:
            bin_end = bin_start + step
            hit = bool((in_window & (peaks >= bin_start) & (peaks < bin_end)).any())
            n_bins += 1
            n_positive += int(hit)
            bin_start = bin_end

    if n_bins == 0:
        raise ValueError(f"no complete {bin_hours:g} h bins fit in the given windows")
    p = n_positive / n_bins
    return BaseRateResult(
        n_bins=n_bins,
        n_positive_bins=n_positive,
        n_events=n_events,
        bin_hours=bin_hours,
        min_class=min_class,
        base_rate=p,
        brier_climatology=p * (1.0 - p),
    )
