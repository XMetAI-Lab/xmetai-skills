"""Prediction/observation discovery and loading."""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import xarray as xr

PRED_PATTERN = re.compile(r"^pred_(\d{8})\.nc$", re.IGNORECASE)
OBS_PATTERN = re.compile(r"^obs_(\d{8})\.nc$", re.IGNORECASE)


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
