"""Single-mask propagation contract (the project's core concept).

Each AR is segmented ONCE on HMI; that single mask is the only region every AIA
layer is read through. These tests pin that contract on `frame_features`:

  1. Out-of-mask pixels never move any feature (HMI flux OR any AIA max) — the
     same mask gates every layer, with no per-channel re-segmentation.
  2. Every layer is read through the *same* mask object (the AIA max is the max
     over exactly the masked pixels of that channel).
  3. A channel not co-registered onto the mask grid is rejected (a mask is only
     valid on the grid it was propagated onto).
"""

import numpy as np
import pytest

from solarflare.features.extract import frame_features


def _channels():
    """A magnetogram + two AIA layers on a shared 4x4 grid, with a corner mask."""
    mask = np.zeros((4, 4), dtype=bool)
    mask[0:2, 0:2] = True  # 2x2 active corner
    blos = np.zeros((4, 4), dtype=np.float32)
    blos[0, 0], blos[1, 1] = 200.0, -150.0  # in-mask field
    aia94 = np.zeros((4, 4), dtype=np.float32)
    aia94[0, 1] = 50.0  # in-mask peak
    aia131 = np.zeros((4, 4), dtype=np.float32)
    aia131[1, 0] = 80.0  # in-mask peak
    channels = {"hmi_magnetogram": blos, "aia_0094": aia94, "aia_0131": aia131}
    return channels, mask


def test_features_read_only_inside_the_mask():
    """Poisoning every out-of-mask pixel must not change a single feature."""
    channels, mask = _channels()
    base = frame_features(channels, mask)

    poisoned = {}
    for key, frame in channels.items():
        f = frame.copy()
        f[~mask] = 9.9e9  # enormous values everywhere outside the AR mask
        poisoned[key] = f
    after = frame_features(poisoned, mask)

    assert base == after  # identical: out-of-mask pixels are never read


def test_same_mask_drives_every_layer():
    """area + each layer's max come from exactly the masked pixels."""
    channels, mask = _channels()
    feats = frame_features(channels, mask)

    assert feats["area_px"] == float(mask.sum())  # one mask, one area
    # Each AIA max is the max over the masked pixels of THAT channel.
    for key, frame in channels.items():
        if key.startswith("aia_"):
            assert feats[f"{key}_max"] == pytest.approx(float(frame[mask].max()))
    # Magnetic features come from the same masked region.
    masked_b = channels["hmi_magnetogram"][mask]
    assert feats["flux_total"] == pytest.approx(float(np.abs(masked_b).sum()))
    assert feats["b_peak"] == pytest.approx(float(np.abs(masked_b).max()))


def test_uncoregistered_layer_is_rejected():
    """A channel whose grid differs from the mask violates the contract."""
    channels, mask = _channels()
    channels["aia_0131"] = np.zeros((8, 8), dtype=np.float32)  # wrong grid
    with pytest.raises(ValueError, match="propagation contract"):
        frame_features(channels, mask)
