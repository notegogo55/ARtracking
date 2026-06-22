"""Foundation segmenters (surya / sam2), fully offline — no GPU, no downloads.

The model-specific load is gated; these tests cover (a) the registry resolves the
names, (b) the GPU/capability gate raises actionable guidance (not an ImportError)
on a no-GPU box, and (c) the decode + write-contract path via an injected fake
backbone (Surya) / fake predictor (SAM2), so the masks are interchangeable with
the threshold/U-Net baselines.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from solarflare.config import SegmentConfig
from solarflare.detect import foundation
from solarflare.detect.segmenter import available_segmenters, get_segmenter

N_FRAMES, H, W = 6, 48, 48


@pytest.fixture()
def tiny_sample(tmp_path):
    """Cached synthetic sample: quiet Sun with a dark spot + strong-field blob."""
    from solarflare.data.cache import load_sample, write_sample

    cont = np.full((N_FRAMES, H, W), 1000.0, dtype=np.float32)
    blos = np.zeros((N_FRAMES, H, W), dtype=np.float32)
    yy, xx = np.ogrid[:H, :W]
    spot = (yy - 24) ** 2 + (xx - 24) ** 2 <= 7**2
    for i in range(N_FRAMES):
        cont[i][spot] = 400.0
        blos[i][spot] = 600.0
    times = pd.DataFrame(
        {
            "frame_idx": range(N_FRAMES),
            "time_utc": pd.date_range("2099-01-01", periods=N_FRAMES, freq="1h"),
        }
    )
    sample_dir = write_sample(
        tmp_path / "sample",
        {"hmi_continuum": cont, "hmi_magnetogram": blos},
        times,
        qa=pd.DataFrame(),
        labels=pd.DataFrame(),
        meta={"noaa": 99999, "harp": 1, "window": "synthetic"},
    )
    return load_sample(sample_dir)


def _cfg() -> SegmentConfig:
    return SegmentConfig(min_region_pixels=10)


def test_registry_resolves_foundation_models():
    assert {"surya", "sam2"} <= set(available_segmenters())
    assert type(get_segmenter(SegmentConfig(model="surya"))).__name__ == "SuryaSegmenter"
    assert type(get_segmenter(SegmentConfig(model="sam2"))).__name__ == "SAM2Segmenter"


def test_capability_gate_raises_guidance_without_gpu(monkeypatch, tiny_sample):
    """No CUDA -> a clear NotImplementedError mentioning the GPU, not an ImportError."""
    monkeypatch.setattr(foundation, "cuda_available", lambda: False)
    cfg = _cfg().model_copy(update={"model": "surya"})
    with pytest.raises(NotImplementedError, match="GPU"):
        get_segmenter(cfg).segment_sample(tiny_sample)
    cfg2 = _cfg().model_copy(update={"model": "sam2"})
    with pytest.raises(NotImplementedError, match="GPU"):
        get_segmenter(cfg2).segment_sample(tiny_sample)


def test_surya_decode_write_contract_with_fake_backbone(tiny_sample):
    """A fake Surya backbone exercises the threshold + write-contract path."""
    cfg = _cfg()

    # backbone returns prob 1.0 inside a fixed square, 0 elsewhere (grid-shaped).
    def fake_backbone(stack: np.ndarray) -> np.ndarray:
        assert stack.ndim == 3  # (C, H, W) — every channel propagated on one grid
        prob = np.zeros((H, W), dtype=np.float32)
        prob[10:25, 10:25] = 1.0
        return prob

    masks_path, areas = foundation.segment_sample_surya(tiny_sample, cfg, backbone=fake_backbone)
    masks = np.load(masks_path)
    assert masks.shape == (N_FRAMES, H, W) and masks.dtype == np.uint8
    assert set(np.unique(masks)) <= {0, 1}
    assert masks[0, 15, 15] == 1 and masks[0, 0, 0] == 0
    assert list(areas.columns) == ["frame_idx", "ar_pixels"]
    assert (areas["ar_pixels"] == 15 * 15).all()
    assert (Path(tiny_sample.sample_dir) / "ar_mask_areas.csv").exists()


class _FakePredictor:
    """Minimal stand-in for SAM2VideoPredictor: records the seed box, yields a mask."""

    def __init__(self):
        self.seed_box = None
        self.frames_dir = None

    def init_state(self, video_path):
        self.frames_dir = Path(video_path)
        # The real init_state consumes a directory of <idx>.jpg frames.
        self.n = len(list(self.frames_dir.glob("*.jpg")))
        return {"n": self.n}

    def add_new_points_or_box(self, state, frame_idx, obj_id, box=None, **kw):
        self.seed_box = np.asarray(box, dtype=np.float32)
        return frame_idx, [obj_id], None

    def propagate_in_video(self, state):
        # Yield a logits map (1,1,H,W) per frame: positive inside the seed box.
        x0, y0, x1, y1 = (int(round(v)) for v in self.seed_box)
        for i in range(state["n"]):
            logits = -np.ones((1, 1, H, W), dtype=np.float32)
            logits[0, 0, y0:y1, x0:x1] = 1.0
            yield i, [1], logits


def test_sam2_propagation_with_fake_predictor(tiny_sample):
    """A fake predictor exercises frame materialization + seed + decode + contract."""
    cfg = _cfg()
    predictor = _FakePredictor()
    masks_path, areas = foundation.segment_sample_sam2(tiny_sample, cfg, predictor=predictor)

    # frames were materialized as one JPEG per frame and handed to init_state
    assert predictor.n == N_FRAMES
    # seed box came from the frame-0 HMI mask (the magnetic-root prompt), near the blob
    assert predictor.seed_box is not None
    cx = 0.5 * (predictor.seed_box[0] + predictor.seed_box[2])
    cy = 0.5 * (predictor.seed_box[1] + predictor.seed_box[3])
    assert 14 < cx < 34 and 14 < cy < 34  # blob centred at (24, 24)

    masks = np.load(masks_path)
    assert masks.shape == (N_FRAMES, H, W) and masks.dtype == np.uint8
    assert masks.sum() > 0
    assert list(areas.columns) == ["frame_idx", "ar_pixels"]


def test_sam2_raises_when_frame0_has_no_ar(tiny_sample):
    """If frame 0 has no AR to prompt with, fail clearly rather than guess."""
    cfg = _cfg().model_copy(update={"bfield_threshold_gauss": 1e9, "spot_threshold": 0.01})
    with pytest.raises(ValueError, match="no AR found on frame 0"):
        foundation.segment_sample_sam2(tiny_sample, cfg, predictor=_FakePredictor())
