"""Multi-initialization mean-bias and RMSE maps."""
from __future__ import annotations

import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import cartopy.crs as ccrs

from ..io import PRED_PATTERN, load_pair
from .common import lead_label, render_map, robust_limits


def plot_composite_error(pairs: list[tuple[Path, Path]], channels: list[str], output_dir: Path, freq: int = 24, extent=None, init_dates: list[str] | None = None) -> None:
    """Compute mean bias and RMSE across multiple init dates and render maps.

    For each channel and lead, stacks error = pred - obs from all init dates
    and computes:
      - Mean bias (systematic error direction)
      - RMSE (error magnitude)
    Output: composite_<channel>_lead<NN>.png (2-panel: Mean Bias / RMSE)
    """
    if init_dates is not None:
        init_set = set(init_dates)
        pairs = [(p, o) for p, o in pairs if PRED_PATTERN.match(p.name) and PRED_PATTERN.match(p.name).group(1) in init_set]
        if not pairs:
            raise SystemExit(f"no pairs matched init-date filter: {init_dates}")

    # Load first pair to get metadata
    with xr.open_dataset(pairs[0][0]) as ds:
        levels = [str(v) for v in ds["level"].values]
        lat = np.asarray(ds["lat"].values, dtype=float)
        lon = np.asarray(ds["lon"].values, dtype=float)
        n_leads = ds.sizes["time"]
        lead_times = [np.datetime64(t).astype("datetime64[D]").astype(str) for t in ds["time"].values]

    init_labels = []
    for pred_path, _ in pairs:
        m = PRED_PATTERN.match(pred_path.name)
        init_labels.append(m.group(1) if m else pred_path.stem)

    for channel in channels:
        if channel not in levels:
            print(f"warning: channel {channel!r} not in {levels}; skipped", file=sys.stderr)
            continue
        c = levels.index(channel)

        # Stack errors: shape (n_init, n_leads, H, W)
        errors = []
        for pred_path, obs_path in pairs:
            pred, obs, _ = load_pair(pred_path, obs_path)
            errors.append(pred[:, c, :, :] - obs[:, c, :, :])
        errors = np.stack(errors, axis=0)  # (n_init, n_leads, H, W)

        mean_bias = np.nanmean(errors, axis=0)   # (n_leads, H, W)
        rmse_map = np.sqrt(np.nanmean(errors ** 2, axis=0))  # (n_leads, H, W)

        for lead in range(n_leads):
            bias_field = mean_bias[lead]
            rmse_field = rmse_map[lead]
            b_lim = float(np.nanmax(np.abs(bias_field))) if np.any(np.isfinite(bias_field)) else 1.0
            r_lo, r_hi = robust_limits(rmse_field)

            fig, axes = plt.subplots(1, 2, figsize=(10, 4.4), subplot_kw={"projection": ccrs.PlateCarree()})
            for ax, field, title, cm, lo, hi in (
                (axes[0], bias_field, "Mean Bias (pred - obs)", "RdBu_r", -b_lim, b_lim),
                (axes[1], rmse_field, "RMSE", "hot_r", r_lo, r_hi),
            ):
                mesh = render_map(ax, field, lat, lon, title, cm, lo, hi, extent)
                fig.colorbar(mesh, ax=ax, orientation="vertical", fraction=0.035, pad=0.02, shrink=0.8)
            n_init = len(pairs)
            fig.suptitle(f"{channel} | {lead_label(lead, freq)} | {n_init} init dates", fontsize=12)
            fig.subplots_adjust(left=0.03, right=0.99, top=0.86, bottom=0.10, wspace=0.28)
            out = output_dir / f"composite_{channel}_lead{lead + 1:02d}.png"
            fig.savefig(out, dpi=120, bbox_inches="tight", facecolor="white")
            plt.close(fig)
            print(f"WROTE {out}")

    # Summary file
    summary = output_dir / "composite_summary.txt"
    with open(summary, "w", encoding="utf-8") as f:
        f.write(f"Composite error across {len(pairs)} init dates\n")
        f.write(f"Init dates: {', '.join(init_labels)}\n")
        f.write(f"Channels: {', '.join(channels)}\n")
        f.write(f"Frequency: {freq}h\n")
    print(f"WROTE {summary}")
