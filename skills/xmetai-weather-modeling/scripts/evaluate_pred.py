#!/usr/bin/env python3
"""Offline evaluation of paired ``pred_*.nc`` / ``obs_*.nc`` forecast files.

Reads the prediction/observation files written by core's
``base_predictor.test()`` (see ``references/evaluation-planning.md``), computes
metrics per channel and per forecast lead, and prints a compact table plus an
optional JSON report. The files already contain physical values (core applies
``inv_normalize`` before saving), so no sidecar denormalization is needed.

Metrics:

- ``rmse``: root mean square error per channel and lead, with optional spatial
  weighting (``--weight-file``) and top-percent extreme value RMSE
  (``--top-percent``), matching core's ``MSE`` evaluator.
- ``ts`` / ``pod`` / ``far`` / ``fb``: threshold contingency metrics
  (Threat Score, Probability of Detection, False Alarm Rate, Frequency Bias)
  for one channel at the given thresholds, with an optional odd-sized
  neighborhood (default 1 = pointwise), matching the core convention.
- ``tcc``: Temporal Correlation Coefficient (Pearson correlation across init
  dates at each grid point, spatially averaged). Requires multiple init dates
  (at least 3). Outputs per-lead TCC and weekly-averaged TCC for S2S models.

Core alignment notes:

- **NaN handling**: core replaces NaN with 0 before evaluation
  (``torch.nan_to_num``). This script follows the same convention.
- **Spatial weighting**: core loads a ``weight.nc`` buffer (latitude-based,
  shape ``(1, H, 1)``) and multiplies element-wise before taking the mean.
  Pass ``--weight-file`` to replicate.
- **Top-percent RMSE**: core's ``MSE`` evaluator supports ``top_percent`` to
  compute RMSE over the top X% of grid points ranked by |target|.  Use
  ``--top-percent`` to replicate.
- **Neighborhood**: core uses ``F.max_pool2d`` (zero-padded, no wrap-around).
  This script uses a padded max-pool equivalent (no longitude wrap) to match.

Thresholds use the channel's physical unit; for ``tp`` (total precipitation in
metres, daily accumulation for s2s) the common 0.1/10/25/50 mm/day levels are
``0.0001, 0.01, 0.025, 0.05``.

Usage:

    python evaluate_pred.py --pred-dir path/to/pred
    python evaluate_pred.py --pred-dir path/to/pred --metrics rmse,ts,pod,far,fb \\
        --thresholds 0.0001,0.01,0.025,0.05 --channel tp --output report.json
    python evaluate_pred.py --pred-dir path/to/pred --weight-file weight.nc --top-percent 0.02
    python evaluate_pred.py --pred-dir path/to/pred --metrics tcc

Read-only: this script never writes prediction or observation data.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr

PRED_PATTERN = re.compile(r"^pred_(\d{8})\.nc$", re.IGNORECASE)
OBS_PATTERN = re.compile(r"^obs_(\d{8})\.nc$", re.IGNORECASE)

# Precipitation grade definitions (threshold in metres, label)
PRECIP_GRADES = [
    (0.0001, "light rain (>0.1mm)"),
    (0.01, "moderate rain (>10mm)"),
    (0.025, "heavy rain (>25mm)"),
    (0.05, "torrential rain (>50mm)"),
]


# ---------------------------------------------------------------------------
# File scanning and loading
# ---------------------------------------------------------------------------

def scan_dir(directory: Path) -> list[tuple[Path, Path]]:
    """Return (pred_path, obs_path) pairs matched by init date."""
    preds = {m.group(1): p for p in directory.glob("pred_*.nc") if (m := PRED_PATTERN.match(p.name))}
    obss = {m.group(1): p for p in directory.glob("obs_*.nc") if (m := OBS_PATTERN.match(p.name))}
    pairs = []
    for init_date in sorted(preds):
        if init_date not in obss:
            print(f"warning: pred_{init_date}.nc has no matching obs file; skipped", file=sys.stderr)
            continue
        pairs.append((preds[init_date], obss[init_date]))
    if not pairs:
        raise SystemExit(f"no pred/obs pairs found in {directory}")
    return pairs


def load_pair(pred_path: Path, obs_path: Path):
    with xr.open_dataset(pred_path) as p, xr.open_dataset(obs_path) as o:
        pred = np.asarray(p["data"].values, dtype=np.float64)
        obs = np.asarray(o["data"].values, dtype=np.float64)
        levels = [str(v) for v in p["level"].values]
    if pred.shape != obs.shape:
        raise SystemExit(f"shape mismatch {pred_path.name}: {pred.shape} vs {obs.shape}")
    return pred, obs, levels


def load_weight(weight_file: Path, h: int, w: int) -> np.ndarray | None:
    """Load spatial weight from a ``weight.nc`` file.

    Returns a ``(1, H, 1)`` array matching core's buffer convention, or
    ``None`` if the file does not exist or cannot be loaded.
    """
    if weight_file is None or not weight_file.is_file():
        return None
    with xr.open_dataset(weight_file) as ds:
        var = ds[list(ds.data_vars)[0]]
        wt = np.asarray(var.values, dtype=np.float64).squeeze()
    # Core reshapes to (1, H, 1) for broadcasting with (B, T, C, H, W).
    if wt.ndim == 1:
        wt = wt.reshape(1, -1, 1)
    elif wt.ndim == 2:
        wt = wt.reshape(1, wt.shape[0], 1)
    else:
        wt = wt.reshape(1, -1, 1)
    return wt


# ---------------------------------------------------------------------------
# NaN handling: match core's torch.nan_to_num(input, nan=0.0)
# ---------------------------------------------------------------------------

def nan_to_zero(arr: np.ndarray) -> np.ndarray:
    """Replace NaN with 0, matching core's ``torch.nan_to_num(nan=0)``."""
    out = np.where(np.isnan(arr), 0.0, arr)
    return out


# ---------------------------------------------------------------------------
# Neighborhood: match core's F.max_pool2d (zero-padded, no wrap-around)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# RMSE computation: aligned with core's MSE evaluator
# ---------------------------------------------------------------------------

def lead_label(index: int, freq: int) -> str:
    """Return a human-readable lead label based on frequency.

    - freq >= 24: shows day number (`D1`, `D2`, ...)
    - freq < 24: shows hours (`6h`, `12h`, `18h`, ...)
    """
    if freq >= 24:
        return f"D{index + 1}"
    return f"{freq * (index + 1)}h"


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


# ---------------------------------------------------------------------------
# Threshold contingency metrics: aligned with core's _ThresholdContingencyMetric
# ---------------------------------------------------------------------------

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



# ---------------------------------------------------------------------------
# TCC: Temporal Correlation Coefficient (weekly average for S2S)
# ---------------------------------------------------------------------------

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

# ---------------------------------------------------------------------------
# Main computation
# ---------------------------------------------------------------------------

def fmt(v: float) -> str:
    return "   -  " if not np.isfinite(v) else f"{v:.4f}"


def compute_all_metrics(
    pairs: list[tuple[Path, Path]],
    metrics: list[str],
    thresholds: list[float],
    channel: str,
    neighborhood: int,
    weight: np.ndarray | None = None,
    top_percent: float = 0.0,
    freq: int = 24,
    grade_labels: dict[float, str] | None = None,
) -> dict[str, Any]:
    """Compute the requested metrics over all pred/obs pairs.

    Returns a report dict with ``levels``, ``n_leads``, ``rmse_per_level``,
    optional ``top_rmse_per_level``, and ``threshold_metrics``.
    """
    if grade_labels is None:
        grade_labels = {}
    report: dict[str, Any] = {
        "files": len(pairs),
        "metrics": list(metrics),
        "thresholds": [float(v) for v in thresholds],
        "grade_labels": grade_labels,
        "channel": channel,
        "freq_hours": freq,
    }
    if weight is not None:
        report["weighted"] = True
    if top_percent > 0.0:
        report["top_percent"] = top_percent

    n_leads = None
    rmse_acc = None
    top_rmse_acc = None


    tcc_all_preds = []
    tcc_all_obs = []
    for i, (pred_path, obs_path) in enumerate(pairs):
        pred, obs, levels = load_pair(pred_path, obs_path)
        if i == 0:
            report["levels"] = levels
            n_leads = int(pred.shape[0])
            report["n_leads"] = n_leads
            if "rmse" in metrics:
                rmse_acc = np.zeros((n_leads, len(levels)))
                if top_percent > 0.0:
                    top_rmse_acc = np.zeros((n_leads, len(levels)))
        if pred.shape[0] != n_leads:
            raise SystemExit(
                f"lead count mismatch in {pred_path.name}: {pred.shape[0]} vs {report['n_leads']}"
            )
        if "rmse" in metrics:
            rmse_result = compute_rmse(pred, obs, levels, weight, top_percent)
            for c, level in enumerate(levels):
                rmse_acc[:, c] += np.asarray(rmse_result["rmse_per_level"][level]) ** 2
                if top_percent > 0.0 and "top_rmse_per_level" in rmse_result:
                    top_rmse_acc[:, c] += np.asarray(rmse_result["top_rmse_per_level"][level]) ** 2

        # Collect pred/obs for TCC computation
        tcc_all_preds.append(pred.copy())
        tcc_all_obs.append(obs.copy())
    if "rmse" in metrics and rmse_acc is not None:
        rmse_acc = np.sqrt(rmse_acc / len(pairs))
        report["rmse_per_level"] = {
            level: [float(rmse_acc[l, c]) for l in range(n_leads)]
            for c, level in enumerate(report["levels"])
        }
        if top_percent > 0.0 and top_rmse_acc is not None:
            top_rmse_acc = np.sqrt(top_rmse_acc / len(pairs))
            report["top_rmse_per_level"] = {
                level: [float(top_rmse_acc[l, c]) for l in range(n_leads)]
                for c, level in enumerate(report["levels"])
            }

# Re-implement threshold aggregation with raw counts
    if any(m in ("ts", "pod", "far", "fb") for m in metrics):
        raw_counts: dict[str, dict[str, np.ndarray]] = {
            str(t): {
                "hit": np.zeros(n_leads),
                "fa": np.zeros(n_leads),
                "miss": np.zeros(n_leads),
            }
            for t in thresholds
        }
        for pred_path, obs_path in pairs:
            pred_raw, obs_raw, levels_raw = load_pair(pred_path, obs_path)
            pred_clean = nan_to_zero(pred_raw)
            obs_clean = nan_to_zero(obs_raw)
            if channel not in levels_raw:
                raise SystemExit(f"channel {channel!r} not found in {levels_raw}")
            c_idx = levels_raw.index(channel)
            p_ch = pred_clean[:, c_idx]  # (T, H, W)
            o_ch = obs_clean[:, c_idx]
            for t_idx in range(n_leads):
                for t in thresholds:
                    pm = p_ch[t_idx] >= t
                    om = o_ch[t_idx] >= t
                    on = neighbor_any(om, neighborhood)
                    pn = neighbor_any(pm, neighborhood)
                    key = str(t)
                    raw_counts[key]["hit"][t_idx] += float((pm & on).sum())
                    raw_counts[key]["fa"][t_idx] += float((pm & ~on).sum())
                    raw_counts[key]["miss"][t_idx] += float((om & ~pn).sum())

        eps = 1.0e-6
        threshold_metrics: dict[str, Any] = {}
        for t in thresholds:
            key = str(t)
            hit = raw_counts[key]["hit"]
            fa = raw_counts[key]["fa"]
            miss = raw_counts[key]["miss"]
            threshold_metrics[key] = {
                "ts": [float(v) for v in hit / (hit + fa + miss + eps)],
                "pod": [float(v) for v in hit / (hit + miss + eps)],
                "far": [float(v) for v in fa / (hit + fa + eps)],
                "fb": [float(v) for v in (hit + fa) / (hit + miss + eps)],
            }
        report["threshold_metrics"] = threshold_metrics

    # TCC computation
    if "tcc" in metrics:
        if len(tcc_all_preds) >= 3:
            tcc_result = compute_tcc(tcc_all_preds, tcc_all_obs, report["levels"], freq)
            report["tcc_per_level"] = tcc_result["tcc_per_level"]
            report["tcc_weekly"] = tcc_result["tcc_weekly"]
            report["week_labels"] = tcc_result["week_labels"]
        else:
            report["tcc_per_level"] = {lev: [float("nan")] * n_leads for lev in report["levels"]}
            report["tcc_weekly"] = {lev: [float("nan")] for lev in report["levels"]}
            report["week_labels"] = ["Week 1"]
    return report




# ---------------------------------------------------------------------------
# Table printing
# ---------------------------------------------------------------------------

def print_table(headers: list[str], rows: list[list[str]]) -> None:
    widths = [max(len(headers[i]), *(len(r[i]) for r in rows)) for i in range(len(headers))]
    line = "| " + " | ".join(h.ljust(widths[i]) for i, h in enumerate(headers)) + " |"
    sep = "|" + "|".join("-" * (w + 2) for w in widths) + "|"
    print(line)
    print(sep)
    for r in rows:
        print("| " + " | ".join(c.ljust(widths[i]) for i, c in enumerate(r)) + " |")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--pred-dir", required=True, help="directory containing pred_*.nc / obs_*.nc pairs")
    parser.add_argument("--metrics", default="rmse", help="comma-separated metrics: rmse, ts, pod, far, fb, tcc")
    parser.add_argument("--thresholds", default="0.0001,0.01,0.025,0.05", help="thresholds in the channel\'s physical unit")
    parser.add_argument("--precip-grades", action="store_true", help="use standard precipitation grade thresholds with labels (overrides --thresholds)")
    parser.add_argument("--channel", default="tp", help="channel for threshold metrics")
    parser.add_argument("--neighborhood", type=int, default=1, help="odd neighborhood size for threshold metrics (1 = pointwise)")
    parser.add_argument("--weight-file", default=None, help="path to weight.nc for spatial weighting (core convention)")
    parser.add_argument("--freq", type=int, default=24, help="forecast frequency in hours (default: 24 for daily, 6 for IWC)")
    parser.add_argument("--top-percent", type=float, default=0.0, help="top X%% extreme value RMSE (0 to disable, e.g. 0.02 for 2%%)")
    parser.add_argument("--output", default=None, help="optional JSON report path")
    args = parser.parse_args(argv)

    directory = Path(args.pred_dir).expanduser()
    if not directory.is_dir():
        raise SystemExit(f"pred dir not found: {directory}")
    if args.neighborhood <= 0 or args.neighborhood % 2 == 0:
        raise SystemExit("--neighborhood must be a positive odd integer")
    if not (0.0 <= args.top_percent < 1.0):
        raise SystemExit("--top-percent must be in [0, 1)")

    metrics = [m.strip().lower() for m in args.metrics.split(",") if m.strip()]
    if args.precip_grades:
        thresholds = [t for t, _ in PRECIP_GRADES]
        grade_labels = {t: label for t, label in PRECIP_GRADES}
    else:
        thresholds = [float(v) for v in args.thresholds.split(",") if v.strip()]
        grade_labels = {}

    # Load weight if provided
    weight = None
    if args.weight_file:
        weight_path = Path(args.weight_file).expanduser()
        # Probe first file to get spatial dims
        pairs = scan_dir(directory)
        with xr.open_dataset(pairs[0][0]) as ds:
            h_dim = ds.dims.get("lat", ds.dims.get("y", None))
            w_dim = ds.dims.get("lon", ds.dims.get("x", None))
        weight = load_weight(weight_path, h_dim, w_dim)
        if weight is None:
            print(f"warning: could not load weight from {weight_path}; using unweighted RMSE", file=sys.stderr)
    else:
        pairs = scan_dir(directory)

    report = compute_all_metrics(pairs, metrics, thresholds, args.channel, args.neighborhood, weight, args.top_percent, args.freq, grade_labels)
    n_leads = report["n_leads"]

    if "rmse" in metrics:
        label = "RMSE (weighted)" if weight is not None else "RMSE (physical units)"
        rows = [["lead"] + [lead_label(l, args.freq) for l in range(n_leads)]]
        for c, level in enumerate(report["levels"]):
            rows.append([level] + [fmt(report["rmse_per_level"][level][l]) for l in range(n_leads)])
        print(label)
        print_table(rows[0], rows[1:])

        if args.top_percent > 0.0 and "top_rmse_per_level" in report:
            pct_label = f"Top {args.top_percent * 100:g}% RMSE"
            rows = [["lead"] + [lead_label(l, args.freq) for l in range(n_leads)]]
            for c, level in enumerate(report["levels"]):
                rows.append([level] + [fmt(report["top_rmse_per_level"][level][l]) for l in range(n_leads)])
            print(pct_label)
            print_table(rows[0], rows[1:])

    if any(m in ("ts", "pod", "far", "fb") for m in metrics):
        # Print summary table with all thresholds
        if len(thresholds) > 1:
            print(f"\nThreshold Metrics Summary ({args.channel})")
            header = ["grade/threshold"] + [lead_label(l, args.freq) for l in range(n_leads)]
            rows = []
            for t in thresholds:
                key = str(t)
                tm = report["threshold_metrics"][key]
                label = grade_labels.get(t, f">= {t}")
                if "ts" in metrics:
                    rows.append([label] + [fmt(tm["ts"][l]) for l in range(n_leads)])
            if rows:
                print("TS:")
                print_table(header, rows)
            rows = []
            for t in thresholds:
                key = str(t)
                tm = report["threshold_metrics"][key]
                label = grade_labels.get(t, f">= {t}")
                if "pod" in metrics:
                    rows.append([label] + [fmt(tm["pod"][l]) for l in range(n_leads)])
            if rows:
                print("POD:")
                print_table(header, rows)
            rows = []
            for t in thresholds:
                key = str(t)
                tm = report["threshold_metrics"][key]
                label = grade_labels.get(t, f">= {t}")
                if "far" in metrics:
                    rows.append([label] + [fmt(tm["far"][l]) for l in range(n_leads)])
            if rows:
                print("FAR:")
                print_table(header, rows)
            rows = []
            for t in thresholds:
                key = str(t)
                tm = report["threshold_metrics"][key]
                label = grade_labels.get(t, f">= {t}")
                if "fb" in metrics:
                    rows.append([label] + [fmt(tm["fb"][l]) for l in range(n_leads)])
            if rows:
                print("FB:")
                print_table(header, rows)
        else:
            for t in thresholds:
                key = str(t)
                tm = report["threshold_metrics"][key]
                if "ts" in metrics:
                    print(f"TS {args.channel} >= {t}")
                print_table(["lead", "TS"], [[lead_label(l, args.freq), fmt(tm["ts"][l])] for l in range(n_leads)])
            if "pod" in metrics:
                print(f"POD {args.channel} >= {t}")
                print_table(["lead", "POD"], [[lead_label(l, args.freq), fmt(tm["pod"][l])] for l in range(n_leads)])
            if "far" in metrics:
                print(f"FAR {args.channel} >= {t}")
                print_table(["lead", "FAR"], [[lead_label(l, args.freq), fmt(tm["far"][l])] for l in range(n_leads)])
            if "fb" in metrics:
                print(f"FB {args.channel} >= {t}")
                print_table(["lead", "FB"], [[lead_label(l, args.freq), fmt(tm["fb"][l])] for l in range(n_leads)])


    if "tcc" in metrics and "tcc_per_level" in report:
        print("\nTCC (Temporal Correlation Coefficient)")
        print("  (Pearson correlation across init dates, spatially averaged)")
        tcc_data = report["tcc_per_level"]
        rows = [["lead"] + [lead_label(l, args.freq) for l in range(n_leads)]]
        for level in report["levels"]:
            rows.append([level] + [fmt(tcc_data[level][l]) for l in range(n_leads)])
        print_table(rows[0], rows[1:])

        if "tcc_weekly" in report:
            print("\nTCC Weekly Average")
            week_labels = report["week_labels"]
            rows = [["channel"] + week_labels]
            for level in report["levels"]:
                rows.append([level] + [fmt(v) for v in report["tcc_weekly"][level]])
            print_table(rows[0], rows[1:])

    if args.output:
        out = Path(args.output).expanduser()
        out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"report: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

