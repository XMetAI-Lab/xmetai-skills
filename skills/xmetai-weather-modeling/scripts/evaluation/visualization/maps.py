"""Single-initialization prediction/observation map panels."""
from __future__ import annotations

import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import cartopy.crs as ccrs

from ..io import load_pair
from .common import CMAP_BY_VAR, lead_label, render_map, robust_limits


def render_compare_frame(channel: str, lead: int, pred, obs, levels, lat, lon, lead_times, freq: int = 24, extent=None, mode: str = "compare"):
    """Render compare figure based on mode: compare (3-panel), pred-obs (2-panel), pred (1-panel)."""
    c = levels.index(channel)
    cmap = CMAP_BY_VAR.get(channel, "viridis")
    p_lo, p_hi = robust_limits(pred[lead, c])

    if mode == "pred":
        fig, ax = plt.subplots(1, 1, figsize=(5, 4.4), subplot_kw={"projection": ccrs.PlateCarree()})
        mesh = render_map(ax, pred[lead, c], lat, lon, f"Prediction | {lead_label(lead, freq)}", cmap, p_lo, p_hi, extent)
        fig.colorbar(mesh, ax=ax, orientation="vertical", fraction=0.035, pad=0.02, shrink=0.8)
        fig.suptitle(f"{channel} | init {lead_times[0]}", fontsize=12)
        fig.subplots_adjust(left=0.05, right=0.95, top=0.88, bottom=0.10)
        return fig

    elif mode == "pred-obs":
        fig, axes = plt.subplots(1, 2, figsize=(10, 4.4), subplot_kw={"projection": ccrs.PlateCarree()})
        for ax, field, title in (
            (axes[0], pred[lead, c], f"Prediction | {lead_label(lead, freq)}"),
            (axes[1], obs[lead, c], "Observation"),
        ):
            mesh = render_map(ax, field, lat, lon, title, cmap, p_lo, p_hi, extent)
            fig.colorbar(mesh, ax=ax, orientation="vertical", fraction=0.035, pad=0.02, shrink=0.8)
        fig.suptitle(f"{channel} | init {lead_times[0]}", fontsize=12)
        fig.subplots_adjust(left=0.03, right=0.99, top=0.86, bottom=0.10, wspace=0.28)
        return fig

    else:  # compare (default 3-panel)
        err = pred[lead, c] - obs[lead, c]
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), subplot_kw={"projection": ccrs.PlateCarree()})
        e_lim = float(np.nanmax(np.abs(err))) if np.any(np.isfinite(err)) else 1.0
        for ax, field, title, cm, lo, hi in (
            (axes[0], pred[lead, c], f"Prediction | {lead_label(lead, freq)}", cmap, p_lo, p_hi),
            (axes[1], obs[lead, c], "Observation", cmap, p_lo, p_hi),
            (axes[2], err, "Error (pred - obs)", "RdBu_r", -e_lim, e_lim),
        ):
            mesh = render_map(ax, field, lat, lon, title, cm, lo, hi, extent)
            fig.colorbar(mesh, ax=ax, orientation="vertical", fraction=0.035, pad=0.02, shrink=0.8)
        fig.suptitle(f"{channel} | init {lead_times[0]}", fontsize=12)
        fig.subplots_adjust(left=0.03, right=0.99, top=0.86, bottom=0.10, wspace=0.28)
        return fig


def plot_compare(first_pair, channels: list[str], output_dir: Path, gif: bool, gif_duration_ms: int, freq: int = 24, extent=None, mode: str = "compare") -> None:
    pred_path, obs_path = first_pair
    with xr.open_dataset(pred_path) as ds:
        levels = [str(v) for v in ds["level"].values]
        lat = np.asarray(ds["lat"].values, dtype=float)
        lon = np.asarray(ds["lon"].values, dtype=float)
        lead_times = [np.datetime64(t).astype("datetime64[D]").astype(str) for t in ds["time"].values]
    pred, obs, _ = load_pair(pred_path, obs_path)

    for channel in channels:
        if channel not in levels:
            print(f"warning: channel {channel!r} not in {levels}; skipped", file=sys.stderr)
            continue
        for lead in range(pred.shape[0]):
            fig = render_compare_frame(channel, lead, pred, obs, levels, lat, lon, lead_times, freq, extent, mode)
            prefix = mode if mode != "compare" else "compare"
            out = output_dir / f"{prefix}_{channel}_lead{lead + 1:02d}.png"
            fig.savefig(out, dpi=120, bbox_inches="tight", facecolor="white")
            plt.close(fig)
            print(f"WROTE {out}")
        if gif:
            from .animation import save_compare_gif
            save_compare_gif(channel, pred, obs, levels, lat, lon, lead_times, output_dir, gif_duration_ms, freq, extent, mode)
