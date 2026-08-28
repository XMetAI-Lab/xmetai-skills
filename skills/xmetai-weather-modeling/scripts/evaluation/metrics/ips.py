"""IPS metric implementation."""
from __future__ import annotations

from typing import Any
import numpy as np


def compute_ips(
    all_preds: list[np.ndarray],
    all_obs: list[np.ndarray],
    levels: list[str],
    freq: int = 24,
    test_times: list[int] | None = None,
    start_pentad: int = 3,
    end_pentad: int = 12,
    pentad_size: int = 5,
) -> dict[str, Any]:
    """Integrated Pattern Score for pentad anomaly forecasts.

    IPS_j = ((((PCC_j + 1) / 2) + AS_j) / 2) * 100

    Parameters
    ----------
    all_preds : list of np.ndarray
        Each element has shape (T, C, H, W) for one init date.
    all_obs : list of np.ndarray
        Same shape as all_preds.
    levels : list of str
        Channel names.
    freq : int
        Forecast frequency in hours.
    test_times : list of int, optional
        Actual lead-time labels used by core, normally ``freq, 2*freq, ...``.
    start_pentad : int
        First pentad to evaluate (default 3).
    end_pentad : int
        Last pentad to evaluate (default 12).
    pentad_size : int
        Number of days per pentad (default 5).

    Returns
    -------
    dict with keys: ips_per_level, ips_overall, pentad_labels
    """
    n_leads = all_preds[0].shape[0]
    if test_times is None:
        test_times = [(idx + 1) * freq for idx in range(n_leads)]
    if len(test_times) != n_leads:
        raise ValueError(f"test_times has {len(test_times)} entries but data has {n_leads} leads")
    if all(lead % 24 == 0 for lead in test_times) and max(test_times) >= 24:
        lead_unit = "hour"
    else:
        lead_unit = "day" if max(test_times) > end_pentad else "pentad"

    pentads = list(range(start_pentad, end_pentad + 1))
    pentad_indices = {p: [] for p in pentads}
    for idx, lead in enumerate(test_times):
        if lead_unit == "pentad":
            pentad = lead
        elif lead_unit == "hour":
            lead_day = (lead + 23) // 24
            pentad = (lead_day - 1) // pentad_size + 1
        else:
            pentad = (lead - 1) // pentad_size + 1
        if pentad in pentad_indices:
            pentad_indices[pentad].append(idx)

    def _lead_weight(p):
        if 3 <= p <= 4: return 5.0
        if 5 <= p <= 6: return 4.0
        if 7 <= p <= 8: return 3.0
        if 9 <= p <= 10: return 2.0
        if 11 <= p <= 12: return 1.0
        return 1.0

    scores = {lev: {p: {"pcc": [], "as": [], "ips": []} for p in pentads} for lev in levels}
    for pred, obs in zip(all_preds, all_obs):
        for c_idx, lev in enumerate(levels):
            for pentad, indices in pentad_indices.items():
                if not indices: continue
                p_pentad = pred[indices, c_idx].mean(axis=0)
                o_pentad = obs[indices, c_idx].mean(axis=0)
                p_flat = p_pentad.flatten()
                o_flat = o_pentad.flatten()
                valid = np.isfinite(p_flat) & np.isfinite(o_flat)
                p_v, o_v = p_flat[valid], o_flat[valid]
                if not valid.any():
                    pcc = 0.0
                elif np.allclose(p_v, o_v):
                    # Match core's explicit perfect-pattern shortcut,
                    # including identical constant fields.
                    pcc = 1.0
                else:
                    p_anom = p_v - p_v.mean()
                    o_anom = o_v - o_v.mean()
                    cov = (p_anom * o_anom).mean()
                    denom = np.sqrt((p_anom**2).mean()) * np.sqrt((o_anom**2).mean())
                    pcc = float(np.clip(cov / denom, -1.0, 1.0)) if denom > 1e-8 else 0.0
                sign_match = ((p_pentad >= 0) & (o_pentad >= 0)) | ((p_pentad < 0) & (o_pentad < 0))
                sign_match_flat = sign_match.flatten()
                as_score = float(sign_match_flat[valid].mean()) if valid.any() else 0.0
                ips = ((((pcc + 1.0) / 2.0) + as_score) / 2.0) * 100.0
                scores[lev][pentad]["pcc"].append(pcc)
                scores[lev][pentad]["as"].append(as_score)
                scores[lev][pentad]["ips"].append(ips)

    pentad_labels = [f"P{p}" for p in pentads]
    ips_per_level = {}
    ips_overall = {}
    for lev in levels:
        lead_ips, lead_pcc, lead_as = [], [], []
        wsum = wips = 0.0
        for pentad in pentads:
            pv = scores[lev][pentad]["pcc"]
            av = scores[lev][pentad]["as"]
            iv = scores[lev][pentad]["ips"]
            lead_pcc.append(float(np.mean(pv)) if pv else 0.0)
            lead_as.append(float(np.mean(av)) if av else 0.0)
            lead_ips.append(float(np.mean(iv)) if iv else 0.0)
            if iv:
                w = _lead_weight(pentad)
                wips += w * lead_ips[-1]
                wsum += w
        ips_per_level[lev] = {"ips": lead_ips, "pcc": lead_pcc, "as": lead_as}
        ips_overall[lev] = float(wips / wsum) if wsum > 0 else 0.0

    return {"ips_per_level": ips_per_level, "ips_overall": ips_overall, "pentad_labels": pentad_labels}
