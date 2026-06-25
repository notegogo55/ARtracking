"""Pluggable `Segmenter` strategy interface + registry.

A single contract turns a cached AR sample into per-frame masks and an area
table, so the segmentation model is a one-line config switch (`segment.model`).
Implementations (registered by their `name`):

  - "threshold": intensity/|B| threshold + morphology. Zero-ML, no GPU, no
    training -- the smoke baseline that brings the whole pipeline up.
  - "unet":      segmentation-models-pytorch U-Net distilled from the threshold
    masks (the dependable trained baseline; `solarflare train-unet` first).
  - "surya":     NASA-IMPACT Surya `ar_segmentation` head (solarflare.detect.
    foundation). GPU-gated: runs on a GPU box with weights, else falls back with
    setup guidance. Decode/write path is offline-testable via a `backbone` arg.
  - "sam2":      Meta Segment Anything 2 temporal mask propagation (foundation).
    Seeds frame 0 with the HMI mask bbox, propagates across frames. GPU-gated
    with the same honest fallback.

Adding a model = adding a `Segmenter` subclass with a `name`, not editing the
pipeline. Swap with `segment.model: threshold | unet | surya | sam2` -- one line.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

if TYPE_CHECKING:
    import pandas as pd

    from solarflare.config import SegmentConfig


class Segmenter(ABC):
    """Strategy that turns a cached AR sample into masks + a per-frame area table.

    Contract (shared by every implementation): `segment_sample` writes
    `ar_masks.npy` (uint8, shape ``(T, H, W)``) and `ar_mask_areas.csv` into the
    sample directory and returns ``(masks_path, areas_df)``. Phase C reads those
    files, so any segmenter is a drop-in replacement.
    """

    #: Config key that selects this segmenter (`segment.model`).
    name: ClassVar[str]

    def __init__(self, cfg: SegmentConfig) -> None:
        self.cfg = cfg

    @abstractmethod
    def segment_sample(self, sample) -> tuple[Path, pd.DataFrame]:
        """Segment every frame of `sample`; write + return (masks_path, areas)."""
        raise NotImplementedError


# --- registry ---------------------------------------------------------------
_SEGMENTERS: dict[str, type[Segmenter]] = {}


def register_segmenter(cls: type[Segmenter]) -> type[Segmenter]:
    """Class decorator: register a `Segmenter` under its `name`."""
    name = getattr(cls, "name", None)
    if not name:
        raise ValueError(f"{cls.__name__} must set a non-empty `name`")
    if name in _SEGMENTERS and _SEGMENTERS[name] is not cls:
        raise ValueError(f"segmenter name {name!r} already registered")
    _SEGMENTERS[name] = cls
    return cls


def available_segmenters() -> list[str]:
    """Sorted list of registered `segment.model` values."""
    return sorted(_SEGMENTERS)


def get_segmenter(cfg: SegmentConfig) -> Segmenter:
    """Resolve `cfg.model` to a constructed `Segmenter` (the one-line swap)."""
    try:
        cls = _SEGMENTERS[cfg.model]
    except KeyError:
        raise ValueError(
            f"unknown segment.model {cfg.model!r}; available: {available_segmenters()}"
        ) from None
    return cls(cfg)


# --- implementations --------------------------------------------------------
@register_segmenter
class ThresholdSegmenter(Segmenter):
    """Intensity/|B| threshold + morphology baseline (no training, no GPU)."""

    name = "threshold"

    def segment_sample(self, sample) -> tuple[Path, pd.DataFrame]:
        from solarflare.detect.segment import segment_sample

        return segment_sample(sample, self.cfg)


@register_segmenter
class UNetSegmenter(Segmenter):
    """segmentation-models-pytorch U-Net distilled from the threshold masks."""

    name = "unet"

    def segment_sample(self, sample) -> tuple[Path, pd.DataFrame]:
        from solarflare.detect.unet import segment_sample_unet

        return segment_sample_unet(sample, self.cfg)


@register_segmenter
class SuryaSegmenter(Segmenter):
    """NASA-IMPACT Surya `ar_segmentation` head (GPU-gated; falls back with guidance)."""

    name = "surya"

    def segment_sample(self, sample) -> tuple[Path, pd.DataFrame]:
        from solarflare.detect.foundation import segment_sample_surya

        return segment_sample_surya(sample, self.cfg)


@register_segmenter
class SAM2Segmenter(Segmenter):
    """Meta Segment Anything 2 temporal mask propagation (GPU-gated; falls back)."""

    name = "sam2"

    def segment_sample(self, sample) -> tuple[Path, pd.DataFrame]:
        from solarflare.detect.foundation import segment_sample_sam2

        return segment_sample_sam2(sample, self.cfg)


@register_segmenter
class FullDiskSegmenter(Segmenter):
    """Threshold full-disk HMI, reproject mask to HARP CEA (operational mode).

    Requires full-disk frames pre-fetched via `solarflare fetch-fulldisk`.
    """

    name = "fulldisk"

    def segment_sample(self, sample) -> tuple[Path, pd.DataFrame]:
        from solarflare.detect.fulldisk_seg import segment_sample_fulldisk

        return segment_sample_fulldisk(sample, self.cfg)
