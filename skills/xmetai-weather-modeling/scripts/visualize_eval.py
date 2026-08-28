#!/usr/bin/env python3
"""Visualize offline evaluation of ``pred_*.nc`` / ``obs_*.nc`` forecast files.

Reads the same paired forecast files as ``evaluate_pred.py`` and renders:

- ``compare_<channel>_lead<NN>.png``: three-panel maps (prediction, observation,
  error) for each requested channel and lead, based on the first init date.
- ``compare_<channel>_leads.gif``: the same frames animated over leads (with
  ``--gif``), core-style.
- ``rmse_curves.png``: per-channel RMSE vs forecast lead.
- ``ts_curves.png``: Threat Score vs forecast lead, one line per threshold.
- `threshold_metrics.png`: TS/POD/FAR vs threshold, one line per lead.
- `tcc_curves.png`: TCC vs forecast lead (per channel).
- `tcc_weekly.png`: TCC weekly average bar chart (per channel).
- ``composite_<channel>_lead<NN>.png``: multi-init mean bias + RMSE maps (``--mode composite``).

Usage:

    python visualize_eval.py --pred-dir path/to/pred --output-dir figs
    python visualize_eval.py --pred-dir path/to/pred --output-dir figs --channels z500,tp
    python visualize_eval.py --pred-dir path/to/pred --output-dir figs --gif

Read-only: never writes prediction or observation data, only the figure files.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from evaluation.io import PRED_PATTERN, scan_dir
from evaluation.runner import compute_all_metrics
from evaluation.visualization.animation import figure_to_pil, save_compare_gif
from evaluation.visualization.common import (
    CMAP_BY_VAR, lead_label, prepare_grid, render_map, robust_limits,
)
from evaluation.visualization.composite import plot_composite_error
from evaluation.visualization.curves import (
    plot_ips_curves, plot_metric_curves, plot_ps_curves, plot_tcc_curves,
)
from evaluation.visualization.maps import plot_compare, render_compare_frame

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--pred-dir", required=True, help="directory containing pred_*.nc / obs_*.nc pairs")
    parser.add_argument("--output-dir", default="eval_figures", help="directory for output figures")
    parser.add_argument("--channels", default="z500,tp", help="comma-separated channels for compare/error maps")
    parser.add_argument("--metrics", default="rmse,ts", help="comma-separated metrics for curves: rmse, ts, pod, far, fb, tcc, ps, ips")
    parser.add_argument("--thresholds", default="0.1,10,25,50", help="thresholds in the channel's physical unit")
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
