"""Foundation-model segmenters: NASA-IMPACT **Surya** and Meta **SAM2**.

Both sit behind the same `Segmenter` contract as the threshold/U-Net baselines
(`segment_sample_*` writes `ar_masks.npy` (uint8, T,H,W) + `ar_mask_areas.csv`
and returns ``(masks_path, areas)``), so `segment.model: surya | sam2` is a
one-line swap that never touches Phase C.

They are **GPU-gated**: each loader checks for CUDA + the model package + weights
and, when any is missing (e.g. this offline/CI box), raises a clear setup-guidance
`NotImplementedError` rather than crashing the pipeline — the unet/threshold path
stays the dependable default. The model-specific load is isolated in one function
per model so the decode + write-contract path is fully unit-testable by injecting
a fake backbone/predictor (no GPU needed).

Verified facts (do not fabricate beyond these):
  - Surya (github.com/NASA-IMPACT/Surya): a SpectFormer+LoRA SDO foundation model;
    weights HF `nasa-ibm-ai4science/Surya-1.0`. Its `ar_segmentation` downstream
    ships a **full-disk CLI** (`downstream_examples/ar_segmentation/infer.py`):
    input (13, 4096, 4096) [AIA 94/131/171/193/211/304/335/1600 + HMI mag/Bx/By/Bz/V],
    output a binary (4096, 4096) AR mask. There is no documented per-cutout
    `predict()` API, so we expose a `backbone` injection point instead of guessing.
  - SAM2 (github.com/facebookresearch/sam2): `SAM2VideoPredictor.from_pretrained`
    / `build_sam2_video_predictor`; `init_state(video_path)`,
    `add_new_points_or_box(state, frame_idx, obj_id, box=...)` ->
    `(frame_idx, obj_ids, masks)`, and `propagate_in_video(state)` yielding the
    same triple per frame (`masks` are logits, shape (n_obj, 1, H, W)).
"""

from __future__ import annotations

import logging
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

#: |B_los| clip (Gauss) used to render HMI frames to 8-bit for SAM2.
_BLOS_CLIP_GAUSS = 300.0


# --- capability gate --------------------------------------------------------
def cuda_available() -> bool:
    """True iff a CUDA device is visible to torch (False if torch is absent)."""
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:  # noqa: BLE001 - torch missing/broken => not available
        return False


def _require_capability(model_name: str, package: str, device: str) -> None:
    """Raise actionable guidance unless the GPU + model package are available.

    CUDA is checked first so the message is stable on a no-GPU box regardless of
    whether the (heavy, optional) model package is installed.
    """
    if device == "cuda" and not cuda_available():
        raise NotImplementedError(
            f"segment.model={model_name!r} needs a CUDA GPU (none visible). "
            "Run it on a GPU box, or use segment.model: unet (or threshold) here. "
            "Set segment.foundation_device: cpu only to force a (very slow) CPU run."
        )
    import importlib

    repos = {"surya": "github.com/NASA-IMPACT/Surya", "sam2": "github.com/facebookresearch/sam2"}
    repo = repos[model_name]
    try:
        importlib.import_module(package)
    except ImportError as err:
        raise NotImplementedError(
            f"segment.model={model_name!r}: the {package!r} package is not installed. "
            f"Install it ({repo}), then re-run. "
            "Until then use segment.model: unet (or threshold)."
        ) from err


# --- shared write-contract --------------------------------------------------
def write_mask_contract(
    sample, mask_list: list[np.ndarray], min_region_pixels: int
) -> tuple[Path, pd.DataFrame]:
    """Write `ar_masks.npy` + `ar_mask_areas.csv` exactly like the baselines.

    Applies the same minimum-region cleanup as the threshold/U-Net path so every
    segmenter produces interchangeable masks for Phase C.
    """
    from skimage import morphology

    if not mask_list:
        raise ValueError("no frames segmented")
    h, w = np.asarray(mask_list[0]).shape
    masks = np.zeros((len(mask_list), h, w), dtype=np.uint8)
    rows = []
    for i, m in enumerate(mask_list):
        clean = morphology.remove_small_objects(
            np.asarray(m, dtype=bool), max_size=min_region_pixels - 1
        )
        masks[i] = clean.astype(np.uint8)
        rows.append({"frame_idx": i, "ar_pixels": int(masks[i].sum())})
    areas = pd.DataFrame(rows)
    masks_path = Path(sample.sample_dir) / "ar_masks.npy"
    np.save(masks_path, masks)
    areas.to_csv(Path(sample.sample_dir) / "ar_mask_areas.csv", index=False)
    log.info(
        "segmented %d frames -> %s (median AR area %d px)",
        len(mask_list),
        masks_path,
        int(areas["ar_pixels"].median()),
    )
    return masks_path, areas


def _mask_bbox(mask: np.ndarray) -> tuple[float, float, float, float] | None:
    """Pixel XYXY bounding box of a boolean mask, or None if it is empty."""
    ys, xs = np.where(np.asarray(mask, dtype=bool))
    if xs.size == 0:
        return None
    return float(xs.min()), float(ys.min()), float(xs.max()) + 1.0, float(ys.max()) + 1.0


# =========================================================================== #
# Surya                                                                       #
# =========================================================================== #
def _load_surya_backbone(cfg):
    """Load a Surya `ar_segmentation` inference callable (C,H,W)->(H,W) prob.

    Capability-gated. Because Surya's downstream ships a full-disk CLI rather than
    a documented per-cutout predict API, this raises precise guidance at the
    unverifiable boundary instead of calling a guessed signature: run the official
    `infer.py` (or supply a `backbone` callable to `segment_sample_surya`).
    """
    _require_capability("surya", "surya", cfg.foundation_device)
    if not Path(cfg.surya_weights).exists():
        raise NotImplementedError(
            f"segment.model='surya': no fine-tuned weights at {cfg.surya_weights}. "
            "Fine-tune downstream_examples/ar_segmentation (download_data.sh + finetune.py "
            "via torchrun, LoRA; base weights HF nasa-ibm-ai4science/Surya-1.0) and place the "
            "checkpoint there."
        )
    raise NotImplementedError(
        "segment.model='surya': Surya's ar_segmentation ships a full-disk CLI "
        "(downstream_examples/ar_segmentation/infer.py; 13-channel 4096x4096 input -> binary "
        "4096x4096 mask), not a documented per-cutout predict() API. To wire it up: run that "
        "infer.py on the full-disk 13-channel cubes for the sample's timestamps, then reproject "
        "its masks onto the sample CEA grid, OR pass a verified `backbone` callable "
        "(C,H,W)->(H,W) probability to segment_sample_surya. Verify the exact model class "
        "against github.com/NASA-IMPACT/Surya before relying on it."
    )


def segment_sample_surya(sample, cfg, backbone=None) -> tuple[Path, pd.DataFrame]:
    """Surya AR masks for every frame of a cached sample (same outputs as the baseline).

    `backbone(stack)` maps a per-frame channel stack (C, H, W) float32 to a (H, W)
    foreground probability in [0, 1]. When None it is loaded via `_load_surya_backbone`
    (GPU + weights gated). Tests inject a fake backbone to cover the decode/write path.
    """
    if backbone is None:
        backbone = _load_surya_backbone(cfg)
    keys = sorted(sample.arrays)
    masks = []
    for i in range(sample.n_frames):
        stack = np.stack([np.asarray(sample.arrays[k][i], dtype=np.float32) for k in keys])
        prob = np.asarray(backbone(stack), dtype=np.float32)
        masks.append(prob >= cfg.foundation_prob_threshold)
    return write_mask_contract(sample, masks, cfg.min_region_pixels)


# =========================================================================== #
# SAM2                                                                        #
# =========================================================================== #
def _frame_to_uint8_rgb(blos: np.ndarray, clip: float = _BLOS_CLIP_GAUSS) -> np.ndarray:
    """One HMI magnetogram frame -> (H, W, 3) uint8 RGB (NaN -> mid-gray)."""
    scaled = (np.clip(blos, -clip, clip) + clip) / (2 * clip)
    gray = np.where(np.isfinite(blos), scaled * 255.0, 127.0).astype(np.uint8)
    return np.repeat(gray[:, :, None], 3, axis=2)


def _materialize_frames(sample, out_dir: Path) -> Path:
    """Write the sample's HMI magnetogram frames as <idx>.jpg (SAM2 init_state input)."""
    from PIL import Image

    out_dir.mkdir(parents=True, exist_ok=True)
    blos = sample.arrays["hmi_magnetogram"]
    for i in range(sample.n_frames):
        img = _frame_to_uint8_rgb(np.asarray(blos[i], dtype=np.float32))
        Image.fromarray(img, mode="RGB").save(out_dir / f"{i:05d}.jpg", quality=95)
    return out_dir


def _seed_box_from_hmi(sample, cfg) -> tuple[float, float, float, float] | None:
    """XYXY box of the frame-0 HMI threshold AR mask — the magnetic-root prompt."""
    from solarflare.detect.segment import ar_mask

    cont = np.asarray(sample.arrays["hmi_continuum"][0], dtype=np.float32)
    blos = np.asarray(sample.arrays["hmi_magnetogram"][0], dtype=np.float32)
    return _mask_bbox(ar_mask(cont, blos, cfg))


def _load_sam2_predictor(cfg):
    """Build a SAM2 video predictor (capability-gated)."""
    _require_capability("sam2", "sam2", cfg.foundation_device)
    if cfg.sam2_checkpoint and cfg.sam2_model_cfg:
        from sam2.build_sam import build_sam2_video_predictor

        return build_sam2_video_predictor(cfg.sam2_model_cfg, str(cfg.sam2_checkpoint))
    from sam2.sam2_video_predictor import SAM2VideoPredictor

    return SAM2VideoPredictor.from_pretrained(cfg.sam2_hf_id)


def _logits_to_mask(mask_logits, shape: tuple[int, int]) -> np.ndarray:
    """SAM2 per-object logits (n_obj,1,H,W) -> (H, W) boolean foreground mask."""
    arr = mask_logits
    if hasattr(arr, "detach"):  # torch tensor
        arr = arr.detach().to("cpu").float().numpy()
    arr = np.asarray(arr)
    while arr.ndim > 2:  # take the first object / channel
        arr = arr[0]
    mask = arr > 0.0
    if mask.shape != shape:
        from skimage.transform import resize

        mask = resize(mask.astype(np.float32), shape, order=0, preserve_range=True) > 0.5
    return mask


def segment_sample_sam2(sample, cfg, predictor=None) -> tuple[Path, pd.DataFrame]:
    """SAM2 temporal mask propagation for a cached sample (same outputs as the baseline).

    Seeds frame 0 with the bounding box of the HMI threshold AR mask (the magnetic
    root -- on-thesis: one HMI prompt propagated through time), then propagates the
    mask across every frame. `predictor` is injectable for offline tests.
    """
    # Capability gate first so selecting sam2 without a GPU fails fast with guidance.
    if predictor is None:
        predictor = _load_sam2_predictor(cfg)

    n = sample.n_frames
    blos0 = np.asarray(sample.arrays["hmi_magnetogram"][0])
    shape = blos0.shape

    seed_box = _seed_box_from_hmi(sample, cfg)
    if seed_box is None:
        raise ValueError(
            "SAM2: no AR found on frame 0 to prompt with; use segment.model: unet "
            "(or threshold) for this sample, or seed a later frame."
        )

    frames_dir = Path(tempfile.mkdtemp(prefix="sam2_frames_"))
    try:
        _materialize_frames(sample, frames_dir)
        state = predictor.init_state(str(frames_dir))
        predictor.add_new_points_or_box(
            state, frame_idx=0, obj_id=1, box=np.asarray(seed_box, dtype=np.float32)
        )
        by_frame: dict[int, np.ndarray] = {}
        for frame_idx, _obj_ids, mask_logits in predictor.propagate_in_video(state):
            by_frame[int(frame_idx)] = _logits_to_mask(mask_logits, shape)
    finally:
        shutil.rmtree(frames_dir, ignore_errors=True)

    masks = [by_frame.get(i, np.zeros(shape, dtype=bool)) for i in range(n)]
    return write_mask_contract(sample, masks, cfg.min_region_pixels)
