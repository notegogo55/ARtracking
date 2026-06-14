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
        default_factory=lambda: [94, 131, 171, 193, 211, 304, 1600, 1700]
    )
    hmi_sharp_series: str = "hmi.sharp_cea_720s"
    hmi_segments: list[str] = Field(default_factory=lambda: ["magnetogram", "continuum"])
    aia_euv_series: str = "aia.lev1_euv_12s"
    aia_uv_series: str = "aia.lev1_uv_24s"
    hmi_cadence_seconds: int = Field(default=720, gt=0)
    aia_cadence_seconds: int = Field(default=720, gt=0)
    aia_match_tolerance_seconds: int = Field(default=360, gt=0)
    cutout_pad_arcsec: float = Field(default=40.0, ge=0)
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


class FeaturesConfig(_StrictModel):
    """Stage C: per-AR feature extraction and sequence assembly."""

    min_valid_fraction: float = Field(default=0.8, gt=0, le=1)
    dataset_version: str = "v1"


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


class DetectConfig(_StrictModel):
    """Stage B detection: HARP-bootstrapped labels + YOLO fine-tuning."""

    bootstrap_cadence_hours: int = Field(default=6, gt=0)
    fulldisk_series: str = "hmi.m_720s"
    fulldisk_segment: str = "magnetogram"
    rebin_scale: float = Field(default=0.25, gt=0, le=1)  # 4096 px -> 1024 px
    image_clip_gauss: float = Field(default=300.0, gt=0)
    min_box_arcsec: float = Field(default=20.0, ge=0)
    yolo_model: str = "data/weights/yolo26n.pt"
    yolo_imgsz: int = Field(default=640, gt=0)
    yolo_epochs: int = Field(default=40, gt=0)
    # Window-blocked splits (never random): names must reference study windows.
    train_windows: list[str] = Field(default_factory=list)
    val_windows: list[str] = Field(default_factory=list)
    test_windows: list[str] = Field(default_factory=list)


class SegmentConfig(_StrictModel):
    """Stage B segmentation: a pluggable `Segmenter` chosen by `model`.

    Implementations live in `solarflare.detect.segmenter` and are resolved via
    its registry, so swapping models is this one line:
      - "threshold": intensity/|B| threshold + morphology (no training, no GPU);
      - "unet":      U-Net distilled from the threshold masks (`train-unet` first);
      - "surya"/"sam2": stubs that raise with setup guidance (GPU + weights).

    The U-Net is trained on the threshold masks as pseudo-labels (no hand labels
    exist), with a time-blocked train/val split per sample (no leakage).
    """

    model: Literal["threshold", "unet", "surya", "sam2"] = "threshold"
    spot_threshold: float = Field(default=0.85, gt=0, lt=1)  # fraction of quiet-Sun median
    bfield_threshold_gauss: float = Field(default=100.0, gt=0)
    min_region_pixels: int = Field(default=64, ge=1)
    morph_radius_px: int = Field(default=2, ge=1)
    # --- U-Net (segmentation-models-pytorch), used when method == "unet" ---
    unet_encoder: str = "resnet18"
    unet_pretrained: bool = True  # ImageNet encoder weights (downloaded on first run)
    unet_epochs: int = Field(default=20, gt=0)
    unet_lr: float = Field(default=1e-3, gt=0)
    unet_batch_size: int = Field(default=8, gt=0)
    unet_tile_px: int = Field(default=224, ge=32)  # random-crop size; multiple of 32
    unet_crops_per_epoch: int = Field(default=256, gt=0)
    unet_val_fraction: float = Field(default=0.2, gt=0, lt=1)  # tail frames of each sample
    unet_prob_threshold: float = Field(default=0.5, gt=0, lt=1)
    unet_weights: Path = Path("outputs/segment/unet/unet_best.pt")

    @field_validator("unet_tile_px")
    @classmethod
    def _tile_divisible(cls, v: int) -> int:
        if v % 32:
            raise ValueError("unet_tile_px must be a multiple of 32 (encoder stride)")
        return v


class TrackConfig(_StrictModel):
    """Stage B tracking: temporal IoU with differential-rotation compensation."""

    iou_threshold: float = Field(default=0.2, gt=0, lt=1)
    max_gap_frames: int = Field(default=2, ge=0)


class QAConfig(_StrictModel):
    """Bad-frame flagging thresholds. Frames are flagged, never silently dropped."""

    max_nan_fraction: float = Field(default=0.5, ge=0, le=1)
    min_coverage: float = Field(default=0.6, ge=0, le=1)


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
    qa: QAConfig = QAConfig()
    detect: DetectConfig = DetectConfig()
    segment: SegmentConfig = SegmentConfig()
    track: TrackConfig = TrackConfig()
    features: FeaturesConfig = FeaturesConfig()
    forecast: ForecastConfig = ForecastConfig()
    geometry: GeometryConfig = GeometryConfig()
    split: SplitConfig = SplitConfig()
    climatology: ClimatologyConfig

    @model_validator(mode="after")
    def _detect_splits_valid(self) -> Config:
        names = {w.name for w in self.study.windows}
        splits = self.detect.train_windows + self.detect.val_windows + self.detect.test_windows
        unknown = set(splits) - names
        if unknown:
            raise ValueError(f"detect split windows not in study.windows: {sorted(unknown)}")
        if len(splits) != len(set(splits)):
            raise ValueError("detect train/val/test windows must be disjoint")
        return self

    def short_hash(self) -> str:
        """Stable 8-char hash of the resolved config, for experiment logging."""
        payload = json.dumps(self.model_dump(mode="json"), sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()[:8]


def load_config(path: str | Path) -> Config:
    """Load and validate a YAML config file into a `Config`."""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return Config.model_validate(raw)
