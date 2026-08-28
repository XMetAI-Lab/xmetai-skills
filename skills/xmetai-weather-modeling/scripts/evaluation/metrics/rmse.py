"""RMSE and top-percent RMSE."""
from __future__ import annotations

from typing import Any
import numpy as np

from ..common import nan_to_zero


def compute_rmse(
    pred: np.ndarray,
    obs: np.ndarray,
    levels: list[str],
    weight: np.ndarray | None = None,
    top_percent: float = 0.0,
) -> dict[str, Any]:
    """RMSE per (level, lead) aggregated over init dates and space.

    Follows core's ``MSE._process_output``:
    - NaN values are replaced with 0 before computing error.
    - If ``weight`` is provided (shape ``(1, H, 1)``), it is broadcast over
      ``(T, C, H, W)`` and multiplied element-wise before taking the mean.
    - If ``top_percent > 0``, also computes RMSE over the top X% of grid
      points ranked by |target|, matching core's ``top_rmse`` logic.
    """
    # Replace NaN with 0 (core convention)
    pred = nan_to_zero(pred)
    obs = nan_to_zero(obs)

    # pred/obs shape: (T, C, H, W)
    err2 = (pred - obs) ** 2

    result: dict[str, Any] = {"levels": levels}

    if weight is not None:
        # weight shape: (1, H, 1) -> broadcast to (T, C, H, W)
        w = weight  # already (1, H, 1)
        weighted_err2 = err2 * w
        # Mean over spatial dims and batch, weighted
        rmse = np.sqrt(weighted_err2.sum(axis=(2, 3)) / w.sum())
        result["rmse_per_level"] = {
            level: [float(v) for v in rmse[:, c]]
            for c, level in enumerate(levels)
        }
    else:
        rmse = np.sqrt(np.mean(err2, axis=(2, 3)))
        result["rmse_per_level"] = {
            level: [float(v) for v in rmse[:, c]]
            for c, level in enumerate(levels)
        }

    # Top-percent extreme value RMSE (core's top_rmse logic)
    if top_percent > 0.0:
        top_rmse_per_level: dict[str, list[float]] = {level: [] for level in levels}
        t_dim, c_dim = pred.shape[0], pred.shape[1]
        for t_idx in range(t_dim):
            for c_idx, level in enumerate(levels):
                target_c = np.abs(obs[t_idx, c_idx]).flatten()  # (H*W,)
                err2_c = err2[t_idx, c_idx]  # (H, W)
                top_k = max(1, int(target_c.size * top_percent))
                threshold = np.sort(target_c)[-top_k]
                extreme_mask = np.abs(obs[t_idx, c_idx]) >= threshold  # (H, W)
                if weight is not None:
                    w_2d = weight.squeeze()  # (H,)
                    if w_2d.ndim == 1:
                        w_2d = w_2d[:, np.newaxis] * np.ones((1, pred.shape[3]))
                    extreme_weight = w_2d * extreme_mask
                else:
                    extreme_weight = extreme_mask.astype(np.float64)
                denom = max(extreme_weight.sum(), 1.0)
                top_val = float(np.sqrt((err2_c * extreme_weight).sum() / denom))
                top_rmse_per_level[level].append(top_val)
        result["top_rmse_per_level"] = top_rmse_per_level

    return result
