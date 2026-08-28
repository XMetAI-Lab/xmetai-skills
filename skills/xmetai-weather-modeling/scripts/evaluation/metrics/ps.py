"""PS metric implementation."""
from __future__ import annotations

from typing import Any
import numpy as np


def compute_ps(
    all_preds: list[np.ndarray],
    all_obs: list[np.ndarray],
    levels: list[str],
    threshold_l1: float = 0.20,
    threshold_l2: float = 0.50,
) -> dict[str, Any]:
    """Climate business PS score.

    Formula: PS = (2*N0 + 2*N1 + 4*N2) / (N + N0 + 2*N1 + 4*N2) * 100

    Anomaly level classification:
    - Normal: |anom| < threshold_l1
    - Level-1: threshold_l1 <= |anom| < threshold_l2
    - Level-2: |anom| >= threshold_l2

    Counts:
    - N0: grid points where predicted and observed anomaly share the same sign.
    - N1: N0 subset where both are >= level-1 anomaly.
    - N2: N1 subset where both are level-2 anomaly.
    - N: total grid points.

    Parameters
    ----------
    all_preds : list of np.ndarray
        Each element has shape (T, C, H, W) for one init date.
    all_obs : list of np.ndarray
        Same shape as all_preds.
    levels : list of str
        Channel names.
    threshold_l1 : float
        Level-1 anomaly threshold in fractional form (default 0.20 = 20%).
    threshold_l2 : float
        Level-2 anomaly threshold in fractional form (default 0.50 = 50%).

    Returns
    -------
    dict with keys: ps_per_level, ps_overall
    """
    n_leads = all_preds[0].shape[0]
    n_levels = all_preds[0].shape[1]

    counts = {
        lev: [{"N0": 0.0, "N1": 0.0, "N2": 0.0, "N": 0.0} for _ in range(n_leads)]
        for lev in levels
    }

    for pred, obs in zip(all_preds, all_obs):
        for c in range(n_levels):
            for t in range(n_leads):
                p = pred[t, c].flatten()
                o = obs[t, c].flatten()
                p_abs = np.abs(p)
                o_abs = np.abs(o)
                p_level = np.zeros_like(p_abs, dtype=np.int32)
                p_level[p_abs >= threshold_l1] = 1
                p_level[p_abs >= threshold_l2] = 2
                o_level = np.zeros_like(o_abs, dtype=np.int32)
                o_level[o_abs >= threshold_l1] = 1
                o_level[o_abs >= threshold_l2] = 2
                # Match torch.sign() equality in core: zero only agrees with
                # zero, rather than being grouped with positive anomalies.
                same_sign = np.sign(p) == np.sign(o)
                n0 = float(same_sign.sum())
                n1 = float(((o_level >= 1) & (p_level >= 1) & same_sign).sum())
                n2 = float(((o_level == 2) & (p_level == 2) & same_sign).sum())
                n = float(p.size)
                counts[levels[c]][t]["N0"] += n0
                counts[levels[c]][t]["N1"] += n1
                counts[levels[c]][t]["N2"] += n2
                counts[levels[c]][t]["N"] += n

    def _ps_formula(n0, n1, n2, n):
        if n <= 0.0:
            return 0.0
        numerator = 2.0 * n0 + 2.0 * n1 + 4.0 * n2
        denominator = n + n0 + 2.0 * n1 + 4.0 * n2
        return float(numerator / denominator * 100.0)

    ps_per_level = {lev: [] for lev in levels}
    ps_overall = {}
    for lev in levels:
        n0t = n1t = n2t = nt = 0.0
        for t in range(n_leads):
            c = counts[lev][t]
            ps_per_level[lev].append(_ps_formula(c["N0"], c["N1"], c["N2"], c["N"]))
            n0t += c["N0"]; n1t += c["N1"]; n2t += c["N2"]; nt += c["N"]
        ps_overall[lev] = _ps_formula(n0t, n1t, n2t, nt)

    return {"ps_per_level": ps_per_level, "ps_overall": ps_overall}
