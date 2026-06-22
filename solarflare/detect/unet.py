"""U-Net segmentation (segmentation-models-pytorch) behind the threshold-mask interface.

No hand labels exist, so the network distills the threshold+morphology baseline:
inputs are (continuum / quiet-Sun median, B_los / 300 G), targets are the
baseline AR masks (pseudo-labels). The train/val split is time-blocked per
sample — the tail `unet_val_fraction` of each sample's frames is validation
only — mirroring the project's no-leakage rule. Inference writes the same
`ar_masks.npy` / `ar_mask_areas.csv` files Phase C reads, so the upgrade is a
drop-in behind `segment_sample_auto`.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from solarflare.detect.segment import ar_mask

log = logging.getLogger(__name__)

#: |B_los| input clip in Gauss for the U-Net magnetogram channel.
BLOS_CLIP_GAUSS = 300.0
#: Encoder downsampling factor: U-Net input dims must be multiples of this.
STRIDE = 32


# --- input preparation ------------------------------------------------------
def normalize_inputs(
    continuum: np.ndarray, blos: np.ndarray, clip_gauss: float = BLOS_CLIP_GAUSS
) -> np.ndarray:
    """One (continuum, B_los) frame -> (2, H, W) float32 network input.

    Channel 0: continuum / quiet-Sun median, clipped to [0, 1.5] (NaN -> quiet);
    channel 1: B_los / clip_gauss, clipped to [-1, 1] (NaN -> 0).
    """
    cont = np.asarray(continuum, dtype=np.float32)
    finite = np.isfinite(cont)
    quiet = float(np.median(cont[finite])) if finite.any() else 0.0
    if quiet > 0:
        c0 = np.clip(np.nan_to_num(cont, nan=quiet) / quiet, 0.0, 1.5)
    else:
        c0 = np.ones_like(cont)  # no usable continuum: pretend quiet Sun
    c1 = np.clip(np.nan_to_num(np.asarray(blos, dtype=np.float32), nan=0.0) / clip_gauss, -1.0, 1.0)
    return np.stack([c0, c1]).astype(np.float32)


def pad_to_stride(x: np.ndarray, stride: int = STRIDE) -> tuple[np.ndarray, tuple[int, int]]:
    """Edge-pad (C, H, W) so H and W are multiples of `stride`; returns (padded, (H, W))."""
    _, h, w = x.shape
    ph, pw = (-h) % stride, (-w) % stride
    if ph or pw:
        x = np.pad(x, ((0, 0), (0, ph), (0, pw)), mode="edge")
    return x, (h, w)


def build_model(encoder: str, pretrained: bool):
    """2-channel single-class smp.Unet (ImageNet encoder init when pretrained)."""
    import segmentation_models_pytorch as smp

    return smp.Unet(
        encoder_name=encoder,
        encoder_weights="imagenet" if pretrained else None,
        in_channels=2,
        classes=1,
    )


# --- training ----------------------------------------------------------------
def _pseudo_label_frames(samples, cfg) -> tuple[list, list]:
    """Per-frame (input (2,H,W), target (H,W)) pairs, time-blocked into (train, val).

    Targets are the threshold-baseline AR masks; the tail `unet_val_fraction`
    of each sample's frames goes to validation (never random).
    """
    train, val = [], []
    for sample in samples:
        cont = sample.arrays["hmi_continuum"]
        blos = sample.arrays["hmi_magnetogram"]
        n = sample.n_frames
        cut = max(int(round(n * (1 - cfg.unet_val_fraction))), 1)
        for i in range(n):
            c, b = np.asarray(cont[i]), np.asarray(blos[i])
            pair = (normalize_inputs(c, b), ar_mask(c, b, cfg).astype(np.float32))
            (train if i < cut else val).append(pair)
    return train, val


def _epoch_crops(frames: list, tile: int, n_crops: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """`n_crops` random tile x tile crops (edge-padded when a frame is smaller)."""
    rng = np.random.default_rng(seed)
    xs = np.empty((n_crops, 2, tile, tile), dtype=np.float32)
    ys = np.empty((n_crops, 1, tile, tile), dtype=np.float32)
    for k in range(n_crops):
        x, y = frames[int(rng.integers(len(frames)))]
        _, h, w = x.shape
        if h < tile or w < tile:
            ph, pw = max(tile - h, 0), max(tile - w, 0)
            x = np.pad(x, ((0, 0), (0, ph), (0, pw)), mode="edge")
            y = np.pad(y, ((0, ph), (0, pw)), mode="edge")
            _, h, w = x.shape
        i0 = int(rng.integers(h - tile + 1))
        j0 = int(rng.integers(w - tile + 1))
        xs[k] = x[:, i0 : i0 + tile, j0 : j0 + tile]
        ys[k, 0] = y[i0 : i0 + tile, j0 : j0 + tile]
    return xs, ys


def _val_iou(model, frames: list, prob_threshold: float, device) -> float:
    """Pixel IoU of full-frame predictions vs the pseudo-labels (pooled over frames)."""
    import torch

    inter = union = 0
    with torch.no_grad():
        for x, y in frames:
            xp, (h, w) = pad_to_stride(x)
            logits = model(torch.from_numpy(xp[None]).to(device))
            pred = (torch.sigmoid(logits)[0, 0, :h, :w] >= prob_threshold).cpu().numpy()
            gt = y.astype(bool)
            inter += int(np.logical_and(pred, gt).sum())
            union += int(np.logical_or(pred, gt).sum())
    return inter / union if union else float("nan")


def train_unet(
    samples: list,
    cfg,
    weights_path: str | Path | None = None,
    epochs: int | None = None,
    device: str = "cpu",
    seed: int = 1337,
) -> tuple[Path, pd.DataFrame]:
    """Distill the threshold baseline into a U-Net over the given cached samples.

    Saves the best-val-IoU checkpoint to `weights_path` (default:
    cfg.unet_weights) plus a training_log.csv next to it; returns both.
    """
    import torch

    from solarflare.utils.seed import set_global_seed

    set_global_seed(seed)
    dev = torch.device(device)
    weights_path = Path(weights_path or cfg.unet_weights)
    weights_path.parent.mkdir(parents=True, exist_ok=True)

    train_frames, val_frames = _pseudo_label_frames(samples, cfg)
    if not train_frames or not val_frames:
        raise ValueError(
            f"need both train and val frames (got {len(train_frames)}/{len(val_frames)}) - "
            "check unet_val_fraction and the cached samples"
        )
    model = build_model(cfg.unet_encoder, cfg.unet_pretrained).to(dev)

    # AR pixels are a small minority of each frame: weight them up (capped).
    n_pos = float(sum(f[1].sum() for f in train_frames))
    n_tot = float(sum(f[1].size for f in train_frames))
    pos_weight = min((n_tot - n_pos) / max(n_pos, 1.0), 50.0)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight).to(dev))
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.unet_lr)

    n_epochs = epochs or cfg.unet_epochs
    best_iou = -np.inf
    history: list[dict] = []
    for epoch in range(n_epochs):
        xs, ys = _epoch_crops(
            train_frames, cfg.unet_tile_px, cfg.unet_crops_per_epoch, seed + 100003 * epoch
        )
        model.train()
        losses = []
        for k0 in range(0, len(xs), cfg.unet_batch_size):
            xb = torch.from_numpy(xs[k0 : k0 + cfg.unet_batch_size]).to(dev)
            yb = torch.from_numpy(ys[k0 : k0 + cfg.unet_batch_size]).to(dev)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        model.eval()
        iou = _val_iou(model, val_frames, cfg.unet_prob_threshold, dev)
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "val_iou": iou})
        log.info(
            "epoch %d/%d: loss %.4f, val IoU %.4f",
            epoch + 1,
            n_epochs,
            history[-1]["train_loss"],
            iou,
        )
        if np.isnan(iou) or iou > best_iou:
            best_iou = iou if not np.isnan(iou) else best_iou
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "encoder": cfg.unet_encoder,
                    "in_channels": 2,
                    "prob_threshold": cfg.unet_prob_threshold,
                    "val_iou": iou,
                    "pos_weight": pos_weight,
                },
                weights_path,
            )
    hist = pd.DataFrame(history)
    hist.to_csv(weights_path.parent / "training_log.csv", index=False)
    log.info("best val IoU %.4f -> %s", best_iou, weights_path)
    return weights_path, hist


# --- inference ---------------------------------------------------------------
def load_unet(weights_path: str | Path, device: str = "cpu"):
    """Rebuild the model from a checkpoint (encoder comes from the checkpoint)."""
    import torch

    ckpt = torch.load(Path(weights_path), map_location=device, weights_only=True)
    model = build_model(ckpt["encoder"], pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    model.to(torch.device(device)).eval()
    return model, ckpt


def segment_sample_unet(
    sample,
    cfg,
    weights_path: str | Path | None = None,
    device: str = "cpu",
) -> tuple[Path, pd.DataFrame]:
    """U-Net masks for every frame of a cached sample (same outputs as segment_sample).

    Writes ar_masks.npy (uint8, T,H,W) + ar_mask_areas.csv into the sample dir.
    """
    import torch
    from skimage import morphology

    weights_path = Path(weights_path or cfg.unet_weights)
    if not weights_path.exists():
        raise FileNotFoundError(
            f"no U-Net weights at {weights_path} - run `solarflare train-unet` first "
            "(or set segment.model: threshold)"
        )
    model, ckpt = load_unet(weights_path, device=device)
    prob_threshold = float(ckpt.get("prob_threshold", cfg.unet_prob_threshold))

    continuum = sample.arrays["hmi_continuum"]
    blos = sample.arrays["hmi_magnetogram"]
    masks = np.zeros(blos.shape, dtype=np.uint8)
    rows = []
    with torch.no_grad():
        for i in range(sample.n_frames):
            x = normalize_inputs(np.asarray(continuum[i]), np.asarray(blos[i]))
            xp, (h, w) = pad_to_stride(x)
            logits = model(torch.from_numpy(xp[None]).to(device))
            pred = (torch.sigmoid(logits)[0, 0, :h, :w] >= prob_threshold).cpu().numpy()
            # same minimum-region rule as the threshold baseline
            pred = morphology.remove_small_objects(pred, max_size=cfg.min_region_pixels - 1)
            masks[i] = pred.astype(np.uint8)
            rows.append({"frame_idx": i, "ar_pixels": int(masks[i].sum())})
    areas = pd.DataFrame(rows)
    masks_path = Path(sample.sample_dir) / "ar_masks.npy"
    np.save(masks_path, masks)
    areas.to_csv(Path(sample.sample_dir) / "ar_mask_areas.csv", index=False)
    log.info(
        "U-Net segmented %d frames -> %s (median AR area %d px)",
        sample.n_frames,
        masks_path,
        int(areas["ar_pixels"].median()),
    )
    return masks_path, areas
