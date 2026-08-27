"""TCC metric implementation."""
from __future__ import annotations

from typing import Any
import numpy as np


def compute_tcc(
    all_preds: list[np.ndarray],
    all_obs: list[np.ndarray],
    levels: list[str],
    freq: int = 24,
) -> dict[str, Any]:
    """Temporal Correlation Coefficient across init dates, averaged over space.

    For each channel and lead time, computes the Pearson correlation between
    forecast and observation time series across init dates at each grid point,
    then takes the spatial mean.

    TCC = mean_grid( corr_t(pred[init_dates], obs[init_dates]) )

    Parameters
    ----------
    all_preds : list of np.ndarray
        Each element has shape (T, C, H, W) for one init date.
    all_obs : list of np.ndarray
        Same shape as all_preds.
    levels : list of str
        Channel names.
    freq : int
        Forecast frequency in hours. Used for weekly grouping.

    Returns
    -------
    dict with keys:
        - ``tcc_per_level``: dict[level -> list[float]], TCC per lead
        - ``tcc_weekly``: dict[level -> list[float]], TCC averaged per week
        - ``week_labels``: list of str, week labels
    """
    n_init = len(all_preds)
    n_leads = all_preds[0].shape[0]
    n_levels = all_preds[0].shape[1]

    # Stack: (n_init, T, C, H, W)
    pred_stack = np.stack(all_preds, axis=0)
    obs_stack = np.stack(all_obs, axis=0)

    tcc_per_level: dict[str, list[float]] = {lev: [] for lev in levels}

    for c in range(n_levels):
        for t in range(n_leads):
            # pred_vals shape: (n_init, H, W)
            p_vals = pred_stack[:, t, c, :, :]
            o_vals = obs_stack[:, t, c, :, :]
            h, w = p_vals.shape[1], p_vals.shape[2]

            # Reshape to (n_init, H*W)
            p_flat = p_vals.reshape(n_init, -1)
            o_flat = o_vals.reshape(n_init, -1)

            # Pearson correlation per grid point
            # Need at least 3 init dates for meaningful correlation
            if n_init < 3:
                tcc_per_level[levels[c]].append(float("nan"))
                continue

            p_mean = p_flat.mean(axis=0, keepdims=True)
            o_mean = o_flat.mean(axis=0, keepdims=True)
            p_anom = p_flat - p_mean
            o_anom = o_flat - o_mean

            cov = (p_anom * o_anom).mean(axis=0)
            p_std = np.sqrt((p_anom ** 2).mean(axis=0))
            o_std = np.sqrt((o_anom ** 2).mean(axis=0))

            denom = p_std * o_std
            # Avoid division by zero (constant fields)
            valid = denom > 1e-12
            tcc_grid = np.zeros(h * w)
            tcc_grid[valid] = cov[valid] / denom[valid]
            tcc_grid[~valid] = np.nan

            # Spatial mean (ignoring NaN)
            tcc_val = float(np.nanmean(tcc_grid))
            tcc_per_level[levels[c]].append(tcc_val)

    # Weekly grouping: S2S 42 leads = 6 weeks (7 days each)
    leads_per_week = 7 * 24 // freq  # For freq=24: 7 leads/week; freq=6: 28 leads/week
    n_weeks = max(1, n_leads // leads_per_week) if leads_per_week > 0 else 1
    week_labels = []
    tcc_weekly: dict[str, list[float]] = {lev: [] for lev in levels}

    for w_idx in range(n_weeks):
        start = w_idx * leads_per_week
        end = min(start + leads_per_week, n_leads)
        week_labels.append(f"Week {w_idx + 1}")
        for lev in levels:
            vals = tcc_per_level[lev][start:end]
            valid_vals = [v for v in vals if np.isfinite(v)]
            avg = float(np.mean(valid_vals)) if valid_vals else float("nan")
            tcc_weekly[lev].append(avg)

    return {
        "tcc_per_level": tcc_per_level,
        "tcc_weekly": tcc_weekly,
        "week_labels": week_labels,
    }
