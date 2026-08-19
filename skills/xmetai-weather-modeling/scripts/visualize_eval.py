#!/usr/bin/env python3
"""Visualize offline evaluation of ``pred_*.nc`` / ``obs_*.nc`` forecast files.

Reads the same paired forecast files as ``evaluate_pred.py`` and renders:

- ``compare_<channel>_lead<NN>.png``: three-panel maps (prediction, observation,
  error) for each requested channel and lead, based on the first init date.
- ``compare_<channel>_leads.gif``: the same frames animated over leads (with
  ``--gif``), core-style.
- ``rmse_curves.png``: per-channel RMSE vs forecast lead.
- ``ts_curves.png``: Threat Score vs forecast lead, one line per threshold.
- ``threshold_metrics.png``: TS/POD/FAR vs threshold, one line per lead.
- ``composite_<channel>_lead<NN>.png``: multi-init mean bias + RMSE maps (``--mode composite``).

Usage:

    python visualize_eval.py --pred-dir path/to/pred --output-dir figs
    python visualize_eval.py --pred-dir path/to/pred --output-dir figs --channels z500,tp
    python visualize_eval.py --pred-dir path/to/pred --output-dir figs --gif

Read-only: never writes prediction or observation data, only the figure files.
"""

from __future__ import annotations

import argparse
import io
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "mplcache"))
os.environ.setdefault("CARTOPY_DATA_DIR", str(Path(tempfile.gettempdir()) / "cartopy_data"))
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import xarray as xr  # noqa: E402
import cartopy.crs as ccrs  # noqa: E402
import cartopy.feature as cfeature  # noqa: E402
from PIL import Image  # noqa: E402

from evaluate_pred import compute_all_metrics, load_pair, scan_dir


def lead_label(index: int, freq: int) -> str:
    if freq >= 24:
        return f"Day {index + 1}"
    return f"{freq * (index + 1)}h"
  # noqa: E402

if "PROJ_DATA" not in os.environ:
    try:
        import pyproj

        os.environ["PROJ_DATA"] = pyproj.datadir.get_data_dir()
    except Exception:
        pass

CMAP_BY_VAR = {
    "sst": "RdYlBu_r",
    "t2m": "RdYlBu_r",
    "msl": "viridis",
    "z500": "viridis",
    "u200": "RdBu_r",
    "u850": "RdBu_r",
    "ttr": "magma",
    "tp": "Blues",
}


def prepare_grid(lon: np.ndarray, lat: np.ndarray, field: np.ndarray):
    """Convert 0..360 lon to -180..180 and sort, matching the pred file convention."""
    lon = np.asarray(lon, dtype=float)
    lat = np.asarray(lat, dtype=float)
    field = np.asarray(field, dtype=float)
    if lon.max() > 180:
        if lon.size > 1 and np.isclose(lon[-1], 360.0):
            lon = lon[:-1]
            field = field[..., :-1]
        lon = ((lon + 180.0) % 360.0) - 180.0
        order = np.argsort(lon)
        lon = lon[order]
        field = field[..., order]
    return lon, lat, field


def render_map(ax, field, lat, lon, title: str, cmap: str, vmin: float, vmax: float, extent=None):
    lon, lat, field = prepare_grid(lon, lat, field)
    if extent:
        ax.set_extent(extent, crs=ccrs.PlateCarree())
    else:
        ax.set_global()
    ax.add_feature(cfeature.OCEAN.with_scale("110m"), facecolor="#d7e9f7", zorder=0)
    ax.add_feature(cfeature.LAND.with_scale("110m"), facecolor="#f2efe9", edgecolor="none", zorder=1)
    ax.coastlines(linewidth=0.5, resolution="110m", zorder=2)
    gl = ax.gridlines(draw_labels=True, linewidth=0.3, color="gray", alpha=0.5, linestyle="--")
    gl.top_labels = False
    gl.right_labels = False
    mesh = ax.pcolormesh(
        lon,
        lat,
        field,
        transform=ccrs.PlateCarree(),
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        shading="auto",
        rasterized=True,
    )
    ax.set_title(title, fontsize=10)
    return mesh


def robust_limits(field: np.ndarray):
    values = field[np.isfinite(field)]
    if values.size == 0:
        return 0.0, 1.0
    low = float(np.percentile(values, 2))
    high = float(np.percentile(values, 98))
    if low == high:
        high = low + 1e-6
    return low, high


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


def figure_to_pil(fig, dpi: int = 120, tight: bool = True) -> Image.Image:
    """Render a matplotlib figure to a palette-mode PIL image (GIF frame)."""
    buf = io.BytesIO()
    kwargs: dict = {"format": "png", "dpi": dpi, "facecolor": "white"}
    if tight:
        kwargs["bbox_inches"] = "tight"
    fig.savefig(buf, **kwargs)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("P", palette=Image.Palette.ADAPTIVE)


def save_compare_gif(channel: str, pred, obs, levels, lat, lon, lead_times, output_dir: Path, duration_ms: int, freq: int = 24, extent=None, mode: str = "compare") -> None:
    """Animate the three-panel compare frames over leads, core-style GIF."""
    frames = [
        figure_to_pil(render_compare_frame(channel, lead, pred, obs, levels, lat, lon, lead_times, freq, extent, mode), tight=False)
        for lead in range(pred.shape[0])
    ]
    out = output_dir / f"compare_{channel}_leads.gif"
    frames[0].save(
        out,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
        optimize=False,
    )
    print(f"WROTE {out} ({len(frames)} frames)")


def plot_composite_error(pairs: list[tuple[Path, Path]], channels: list[str], output_dir: Path, freq: int = 24, extent=None, init_dates: list[str] | None = None) -> None:
    """Compute mean bias and RMSE across multiple init dates and render maps.

    For each channel and lead, stacks error = pred - obs from all init dates
    and computes:
      - Mean bias (systematic error direction)
      - RMSE (error magnitude)
    Output: composite_<channel>_lead<NN>.png (2-panel: Mean Bias / RMSE)
    """
    from evaluate_pred import PRED_PATTERN

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
            save_compare_gif(channel, pred, obs, levels, lat, lon, lead_times, output_dir, gif_duration_ms, freq, extent, mode)


def plot_metric_curves(report: dict, metrics: list[str], thresholds: list[float], output_dir: Path, freq: int = 24) -> None:
    leads = list(range(1, report["n_leads"] + 1))
    lead_labels = [lead_label(l - 1, freq) for l in leads]

    if "rmse" in metrics:
        levels = report["levels"]
        cols = 3
        rows = int(np.ceil(len(levels) / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(12, 3 * rows), squeeze=False)
        for idx, level in enumerate(levels):
            ax = axes[idx // cols][idx % cols]
            ax.plot(leads, report["rmse_per_level"][level], marker="o", ms=3)
            ax.set_title(level, fontsize=10)
            ax.set_xlabel("lead time")
            ax.set_xticks(leads)
            ax.set_xticklabels(lead_labels, fontsize=7, rotation=45 if freq < 24 else 0)
            ax.grid(alpha=0.3)
        for idx in range(len(levels), rows * cols):
            axes[idx // cols][idx % cols].axis("off")
        fig.suptitle("RMSE per channel", fontsize=13)
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        out = output_dir / "rmse_curves.png"
        fig.savefig(out, dpi=120, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"WROTE {out}")

    if "ts" in metrics and report.get("threshold_metrics"):
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for t in thresholds:
            key = str(t)
            ax.plot(leads, report["threshold_metrics"][key]["ts"], marker="o", ms=3, label=f">= {t}")
        ax.set_xlabel("lead time")
        ax.set_ylabel("TS")
        ax.set_title(f"Threat Score ({report['channel']})")
        ax.set_xticks(leads)
        ax.set_xticklabels(lead_labels, fontsize=7, rotation=45 if freq < 24 else 0)
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()
        out = output_dir / "ts_curves.png"
        fig.savefig(out, dpi=120, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"WROTE {out}")

        fig, ax = plt.subplots(figsize=(7, 4.5))
        tm = report["threshold_metrics"]
        for metric, style in (("ts", "-o"), ("pod", "-s"), ("far", "-^")):
            for lead in range(report["n_leads"]):
                values = [tm[str(t)][metric][lead] for t in thresholds]
                label = f"{metric} lead {lead + 1}" if lead == 0 else None
                ax.plot(thresholds, values, style, ms=3, label=label)
        ax.set_xscale("log")
        ax.set_xlabel("threshold (physical unit)")
        ax.set_ylabel("score")
        ax.set_title(f"Threshold metrics ({report['channel']})")
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=7)
        fig.tight_layout()
        out = output_dir / "threshold_metrics.png"
        fig.savefig(out, dpi=120, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"WROTE {out}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--pred-dir", required=True, help="directory containing pred_*.nc / obs_*.nc pairs")
    parser.add_argument("--output-dir", default="eval_figures", help="directory for output figures")
    parser.add_argument("--channels", default="z500,tp", help="comma-separated channels for compare/error maps")
    parser.add_argument("--metrics", default="rmse,ts", help="comma-separated metrics for curves: rmse, ts, pod, far, fb")
    parser.add_argument("--thresholds", default="0.0001,0.01,0.025,0.05", help="thresholds in the channel's physical unit")
    parser.add_argument("--channel", default="tp", help="channel for threshold metrics")
    parser.add_argument("--neighborhood", type=int, default=1, help="odd neighborhood size for threshold metrics (1 = pointwise)")
    parser.add_argument("--gif", action="store_true", help="also animate compare frames over leads as GIFs")
    parser.add_argument("--gif-duration-ms", type=int, default=500, help="GIF frame duration in milliseconds")
    parser.add_argument("--freq", type=int, default=24, help="forecast frequency in hours (default: 24 for daily, 6 for IWC)")
    parser.add_argument("--region", default=None, help="region name (e.g., 'china', 'east_china') for LLM to resolve, or 'lon_min,lon_max,lat_min,lat_max' coordinates")
    parser.add_argument("--init-date", default=None, help="comma-separated init dates to include (default: all found in pred-dir)")
    parser.add_argument("--mode", default="compare", choices=["compare", "pred-obs", "pred", "composite"], help="visualization mode: compare (3-panel), pred-obs (2-panel), pred (single panel), composite (multi-init mean bias + RMSE)")
    args = parser.parse_args(argv)

    directory = Path(args.pred_dir).expanduser()
    if not directory.is_dir():
        raise SystemExit(f"pred dir not found: {directory}")
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    channels = [c.strip() for c in args.channels.split(",") if c.strip()]
    metrics = [m.strip().lower() for m in args.metrics.split(",") if m.strip()]
    thresholds = [float(v) for v in args.thresholds.split(",") if v.strip()]

    pairs = scan_dir(directory)

    # Parse region/extent
    extent = None
    if args.region:
        r = args.region.strip()
        if ',' in r:
            # Direct coordinates: lon_min,lon_max,lat_min,lat_max
            parts = [float(x) for x in r.split(',')]
            if len(parts) == 4:
                extent = parts
            else:
                raise SystemExit("--region coordinates must be 4 values: lon_min,lon_max,lat_min,lat_max")
        else:
            raise SystemExit(f"--region name {r!r} requires LLM to resolve coordinates. Use --region lon_min,lon_max,lat_min,lat_max instead.")
    init_dates = [d.strip() for d in args.init_date.split(",") if d.strip()] if args.init_date else None

    if args.mode == "composite":
        plot_composite_error(pairs, channels, output_dir, args.freq, extent, init_dates)
        return 0

    # For single-init modes, use first pair (or first matching --init-date)
    if init_dates is not None:
        from evaluate_pred import PRED_PATTERN
        init_set = set(init_dates)
        filtered = [(p, o) for p, o in pairs if PRED_PATTERN.match(p.name) and PRED_PATTERN.match(p.name).group(1) in init_set]
        if filtered:
            pairs = filtered
    report = compute_all_metrics(pairs, metrics, thresholds, args.channel, args.neighborhood, freq=args.freq)
    plot_compare(pairs[0], channels, output_dir, args.gif, args.gif_duration_ms, args.freq, extent, args.mode)
    plot_metric_curves(report, metrics, thresholds, output_dir, args.freq)
    return 0


if __name__ == "__main__":
    sys.exit(main())



