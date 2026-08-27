"""Threshold contingency metrics: TS, POD, FAR, and FB."""
from __future__ import annotations

from typing import Any
import numpy as np

from ..common import nan_to_zero, neighbor_any


def compute_threshold_metrics(
    pred: np.ndarray,
    obs: np.ndarray,
    levels: list[str],
    thresholds: list[float],
    channel: str,
    neighborhood: int,
) -> dict[str, Any]:
    """Compute TS/POD/FAR/FB for a single channel at given thresholds.

    Follows core's ``_ThresholdContingencyMetric``:
    - Uses ``neighbor_any`` (zero-padded dilation) for neighborhood matching.
    - Counts are accumulated per lead time (frame_index) across all init dates.
    - Final scores are computed from aggregated counts, matching core's
      ``evaluate()`` method.
    """
    if channel not in levels:
        raise SystemExit(f"channel {channel!r} not found in {levels}")

    # Replace NaN with 0 (core convention)
    pred = nan_to_zero(pred)
    obs = nan_to_zero(obs)

    c = levels.index(channel)
    p = pred[:, c]  # (T, H, W)
    o = obs[:, c]

    n_leads = p.shape[0]
    counts: dict[str, dict[str, np.ndarray]] = {
        str(t): {
            "hit": np.zeros(n_leads),
            "fa": np.zeros(n_leads),
            "miss": np.zeros(n_leads),
        }
        for t in thresholds
    }

    for t_idx in range(n_leads):
        for t in thresholds:
            pm = p[t_idx] >= t  # (H, W)
            om = o[t_idx] >= t
            on = neighbor_any(om, neighborhood)
            pn = neighbor_any(pm, neighborhood)
            key = str(t)
            counts[key]["hit"][t_idx] = float((pm & on).sum())
            counts[key]["fa"][t_idx] = float((pm & ~on).sum())
            counts[key]["miss"][t_idx] = float((om & ~pn).sum())

    threshold_metrics: dict[str, Any] = {}
    eps = 1.0e-6  # core uses eps=1e-6 to avoid division by zero
    for t in thresholds:
        key = str(t)
        hit = counts[key]["hit"]
        fa = counts[key]["fa"]
        miss = counts[key]["miss"]
        ts = hit / (hit + fa + miss + eps)
        pod = hit / (hit + miss + eps)
        far = fa / (hit + fa + eps)
        fb = (hit + fa) / (hit + miss + eps)
        threshold_metrics[key] = {
            "ts": [float(v) for v in ts],
            "pod": [float(v) for v in pod],
            "far": [float(v) for v in far],
            "fb": [float(v) for v in fb],
        }

    return threshold_metrics
