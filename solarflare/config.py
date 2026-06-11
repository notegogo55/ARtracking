"""Typed configuration schema (pydantic v2) and YAML loader.

All times are naive ISO-8601 strings interpreted as UTC. Unknown keys are
rejected (`extra="forbid"`) so YAML typos fail loudly at load time.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

#: AIA channels (Angstrom) available from the SDO/AIA instrument.
VALID_AIA_CHANNELS = {94, 131, 171, 193, 211, 304, 335, 1600, 1700, 4500}


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ProjectConfig(_StrictModel):
    name: str = "solarflare-mvp"
    seed: int = 1337


class PathsConfig(_StrictModel):
    data_root: Path = Path("data")
    cache_dir: Path = Path("data/cache")
    outputs_dir: Path = Path("outputs")
    experiment_log: Path = Path("outputs/experiments.csv")


class StudyTarget(_StrictModel):
    noaa: int = Field(gt=0, description="NOAA active-region number")
    harp: int | None = Field(
        default=None,
        gt=0,
        description="JSOC HARPNUM; verify via the official harpnum<->NOAA mapping",
    )
    notes: str = ""


class StudyWindow(_StrictModel):
    name: str
    start: datetime
    end: datetime
    kind: Literal["active", "quiet"] = "active"
    targets: list[StudyTarget] = Field(default_factory=list)

    @model_validator(mode="after")
    def _end_after_start(self) -> StudyWindow:
        if self.end <= self.start:
            raise ValueError(
                f"window '{self.name}': end ({self.end}) must be after start ({self.start})"
            )
        return self


class StudyConfig(_StrictModel):
    mvp_window: str
    windows: list[StudyWindow] = Field(min_length=1)

    @model_validator(mode="after")
    def _mvp_window_exists(self) -> StudyConfig:
        names = {w.name for w in self.windows}
        if len(names) != len(self.windows):
            raise ValueError("study window names must be unique")
        if self.mvp_window not in names:
            raise ValueError(
                f"mvp_window '{self.mvp_window}' not found among windows {sorted(names)}"
            )
        return self


class DataConfig(_StrictModel):
    aia_channels: list[int] = Field(
        default_factory=lambda: [94, 131, 171, 193, 211, 304, 335, 1600]
    )
    hmi_sharp_series: str = "hmi.sharp_cea_720s"
    sample_cadence_minutes: int = Field(default=60, gt=0)

    @field_validator("aia_channels")
    @classmethod
    def _channels_valid(cls, v: list[int]) -> list[int]:
        bad = set(v) - VALID_AIA_CHANNELS
        if bad:
            raise ValueError(
                f"unknown AIA channels {sorted(bad)}; valid: {sorted(VALID_AIA_CHANNELS)}"
            )
        if len(set(v)) != len(v):
            raise ValueError("duplicate AIA channels")
        return v


class ForecastConfig(_StrictModel):
    lookback_hours: float = Field(default=24, gt=0)
    lead_hours: float = Field(default=24, gt=0)
    flare_class_threshold: str = "M1.0"
    bin_hours: float = Field(default=24, gt=0)

    @field_validator("flare_class_threshold")
    @classmethod
    def _threshold_parses(cls, v: str) -> str:
        from solarflare.data.goes_events import goes_class_to_flux

        goes_class_to_flux(v)  # raises ValueError if malformed
        return v


class GeometryConfig(_StrictModel):
    max_cm_longitude_deg: float = Field(default=65, ge=0, le=90)


class SplitConfig(_StrictModel):
    strategy: Literal["time_blocked", "by_rotation"] = "time_blocked"
    n_folds: int = Field(default=5, ge=2)
    embargo_hours: float = Field(default=48, ge=0)


class ClimatologyConfig(_StrictModel):
    start: datetime
    end: datetime

    @model_validator(mode="after")
    def _end_after_start(self) -> ClimatologyConfig:
        if self.end <= self.start:
            raise ValueError("climatology: end must be after start")
        return self


class Config(_StrictModel):
    project: ProjectConfig = ProjectConfig()
    paths: PathsConfig = PathsConfig()
    study: StudyConfig
    data: DataConfig = DataConfig()
    forecast: ForecastConfig = ForecastConfig()
    geometry: GeometryConfig = GeometryConfig()
    split: SplitConfig = SplitConfig()
    climatology: ClimatologyConfig

    def short_hash(self) -> str:
        """Stable 8-char hash of the resolved config, for experiment logging."""
        payload = json.dumps(self.model_dump(mode="json"), sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()[:8]


def load_config(path: str | Path) -> Config:
    """Load and validate a YAML config file into a `Config`."""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return Config.model_validate(raw)
