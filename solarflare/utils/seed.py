"""Global determinism helper. One seed governs random, numpy, and (when present) torch."""

from __future__ import annotations

import logging
import os
import random

import numpy as np

log = logging.getLogger(__name__)


def set_global_seed(seed: int) -> int:
    """Seed every RNG the pipeline uses. Returns the seed for logging convenience."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass  # torch arrives in Phase B/D; seeding must not require it
    log.debug("global seed set to %d", seed)
    return seed
