"""Metric selection and cross-initialization aggregation."""
from __future__ import annotations

from pathlib import Path
from typing import Any
import numpy as np

from .common import nan_to_zero, neighbor_any
from .io import load_pair
from .metrics.ips import compute_ips
from .metrics.ps import compute_ps
from .metrics.rmse import compute_rmse
from .metrics.tcc import compute_tcc

THRESHOLD_METRICS = frozenset({"ts", "pod", "far", "fb"})
CLIMATE_METRICS = frozenset({"tcc", "ps", "ips"})


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
    threshold_requested = bool(THRESHOLD_METRICS.intersection(metrics))
    climate_requested = bool(CLIMATE_METRICS.intersection(metrics))
    raw_counts = None
    climate_preds = []
    climate_obs = []
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
            if threshold_requested:
                raw_counts = {
                    str(threshold): {
                        "hit": np.zeros(n_leads),
                        "fa": np.zeros(n_leads),
                        "miss": np.zeros(n_leads),
                    }
                    for threshold in thresholds
                }
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

        if threshold_requested:
            pred_clean = nan_to_zero(pred)
            obs_clean = nan_to_zero(obs)
            if channel not in levels:
                raise SystemExit(f"channel {channel!r} not found in {levels}")
            c_idx = levels.index(channel)
            for t_idx in range(n_leads):
                for threshold in thresholds:
                    predicted = pred_clean[t_idx, c_idx] >= threshold
                    observed = obs_clean[t_idx, c_idx] >= threshold
                    observed_near = neighbor_any(observed, neighborhood)
                    predicted_near = neighbor_any(predicted, neighborhood)
                    counts = raw_counts[str(threshold)]
                    counts["hit"][t_idx] += float((predicted & observed_near).sum())
                    counts["fa"][t_idx] += float((predicted & ~observed_near).sum())
                    counts["miss"][t_idx] += float((observed & ~predicted_near).sum())

        # Climate metrics currently require the complete initialization axis.
        # Do not retain these arrays for RMSE/threshold-only evaluations.
        if climate_requested:
            climate_preds.append(pred.copy())
            climate_obs.append(obs.copy())
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

    if threshold_requested:
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
        if len(climate_preds) >= 3:
            tcc_result = compute_tcc(climate_preds, climate_obs, report["levels"], freq)
            report["tcc_per_level"] = tcc_result["tcc_per_level"]
            report["tcc_weekly"] = tcc_result["tcc_weekly"]
            report["week_labels"] = tcc_result["week_labels"]
        else:
            report["tcc_per_level"] = {lev: [float("nan")] * n_leads for lev in report["levels"]}
            report["tcc_weekly"] = {lev: [float("nan")] for lev in report["levels"]}
            report["week_labels"] = ["Week 1"]

    # PS computation
    if "ps" in metrics:
        ps_result = compute_ps(climate_preds, climate_obs, report["levels"])
        report["ps_per_level"] = ps_result["ps_per_level"]
        report["ps_overall"] = ps_result["ps_overall"]

    # IPS computation
    if "ips" in metrics:
        ips_result = compute_ips(climate_preds, climate_obs, report["levels"], freq)
        report["ips_per_level"] = ips_result["ips_per_level"]
        report["ips_overall"] = ips_result["ips_overall"]
        report["pentad_labels"] = ips_result["pentad_labels"]

    return report
