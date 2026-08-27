"""Shared numerical and labeling helpers."""
from __future__ import annotations

import numpy as np


def nan_to_zero(arr: np.ndarray) -> np.ndarray:
    """Replace NaN with 0, matching core's ``torch.nan_to_num(nan=0)``."""
    out = np.where(np.isnan(arr), 0.0, arr)
    return out


def neighbor_any(mask: np.ndarray, neighborhood: int) -> np.ndarray:
    """Boolean dilation with a square kernel, zero-padded (no wrap).

    Equivalent to ``F.max_pool2d(mask.float(), kernel_size=neighborhood,
    stride=1, padding=neighborhood // 2) > 0`` but in pure numpy.
    Uses ``np.pad`` with ``constant_values=False`` to match core's zero-padding
    behavior at grid boundaries.
    """
    if neighborhood <= 1:
        return mask
    k = int(neighborhood) // 2
    h, w = mask.shape
    padded = np.pad(mask, k, mode="constant", constant_values=False)
    out = np.zeros_like(mask)
    for di in range(neighborhood):
        for dj in range(neighborhood):
            out |= padded[di : di + h, dj : dj + w]
    return out


def lead_label(index: int, freq: int) -> str:
    """Return a human-readable lead label based on frequency.

    - freq >= 24: shows day number (`D1`, `D2`, ...)
    - freq < 24: shows hours (`6h`, `12h`, `18h`, ...)
    """
    if freq >= 24:
        return f"D{index + 1}"
    return f"{freq * (index + 1)}h"
