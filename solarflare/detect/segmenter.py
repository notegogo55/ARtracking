"""Pluggable `Segmenter` strategy interface + registry.

A single contract turns a cached AR sample into per-frame masks and an area
table, so the segmentation model is a one-line config switch (`segment.model`).
Implementations (registered by their `name`):

  - "threshold": intensity/|B| threshold + morphology. Zero-ML, no GPU, no
    training -- the smoke baseline that brings the whole pipeline up.
  - "unet":      segmentation-models-pytorch U-Net distilled from the threshold
    masks (the dependable trained baseline; `solarflare train-unet` first).
  - "surya":     NASA-IMPACT Surya `ar_segmentation` head. STUB -- raises with
    setup guidance (needs a CUDA GPU + HuggingFace weights).
  - "sam2":      Meta Segment Anything 2 mask propagation. STUB -- raises with
    setup guidance (needs a GPU + a SAM2 checkpoint).

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
    """NASA-IMPACT Surya `ar_segmentation` head -- not wired up in this offline build."""

    name = "surya"

    def segment_sample(self, sample) -> tuple[Path, pd.DataFrame]:
        raise NotImplementedError(
            "segment.model='surya' is a stub in this offline build. To enable it:\n"
            "  1. install Surya (github.com/NASA-IMPACT/Surya);\n"
            "  2. fetch the Surya-1.0 weights from HuggingFace "
            "(nasa-ibm-ai4science/Surya-1.0);\n"
            "  3. fine-tune downstream_examples/ar_segmentation "
            "(download_data.sh + finetune.py via torchrun, LoRA);\n"
            "  4. implement SuryaSegmenter.segment_sample to run inference and write "
            "ar_masks.npy + ar_mask_areas.csv (the ThresholdSegmenter contract).\n"
            "Needs a CUDA GPU. Until then use segment.model: unet (or threshold)."
        )


@register_segmenter
class SAM2Segmenter(Segmenter):
    """Meta Segment Anything 2 mask propagation -- not wired up in this offline build."""

    name = "sam2"

    def segment_sample(self, sample) -> tuple[Path, pd.DataFrame]:
        raise NotImplementedError(
            "segment.model='sam2' is a stub in this offline build. To enable it:\n"
            "  1. install SAM2 (github.com/facebookresearch/sam2) + a SAM2 checkpoint;\n"
            "  2. implement SAM2Segmenter.segment_sample to propagate AR masks across "
            "frames (seed the prompts with HARP boxes or U-Net masks) and write "
            "ar_masks.npy + ar_mask_areas.csv (the ThresholdSegmenter contract).\n"
            "Best suited to the Phase-3 video/tracking propagation; needs a GPU. "
            "Until then use segment.model: unet (or threshold)."
        )
