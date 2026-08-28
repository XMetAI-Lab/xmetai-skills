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
mm, daily accumulation for s2s) the common grade thresholds are 0.1/10/25/50 mm/day.

Usage:

    python evaluate_pred.py --pred-dir path/to/pred
    python evaluate_pred.py --pred-dir path/to/pred --metrics rmse,ts,pod,far,fb \\
        --thresholds 0.1,10,25,50 --channel tp --output report.json
    python evaluate_pred.py --pred-dir path/to/pred --weight-file weight.nc --top-percent 0.02
    python evaluate_pred.py --pred-dir path/to/pred --metrics tcc

Read-only: this script never writes prediction or observation data.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import xarray as xr

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from evaluation.common import lead_label, nan_to_zero, neighbor_any
from evaluation.io import OBS_PATTERN, PRED_PATTERN, load_pair, load_weight, scan_dir
from evaluation.metrics.contingency import compute_threshold_metrics
from evaluation.metrics.ips import compute_ips
from evaluation.metrics.ps import compute_ps
from evaluation.metrics.rmse import compute_rmse
from evaluation.metrics.tcc import compute_tcc
from evaluation.report import fmt, print_table
from evaluation.runner import compute_all_metrics

PRECIP_GRADES = [
    (0.1, "light rain (>=0.1mm)"),
    (10, "moderate rain (>=10mm)"),
    (25, "heavy rain (>=25mm)"),
    (50, "torrential rain (>=50mm)"),
]

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--pred-dir", required=True, help="directory containing pred_*.nc / obs_*.nc pairs")
    parser.add_argument("--metrics", default="rmse", help="comma-separated metrics: rmse, ts, pod, far, fb, tcc, ps, ips")
    parser.add_argument("--thresholds", default="0.1,10,25,50", help="thresholds in the channel\'s physical unit")
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

    if "ps" in metrics and "ps_per_level" in report:
        print("\nPS (Climate Business PS Score)")
        print("  PS = (2*N0 + 2*N1 + 4*N2) / (N + N0 + 2*N1 + 4*N2) * 100")
        ps_data = report["ps_per_level"]
        rows = [["lead"] + [lead_label(l, args.freq) for l in range(n_leads)]]
        for level in report["levels"]:
            rows.append([level] + [fmt(ps_data[level][l]) for l in range(n_leads)])
        print_table(rows[0], rows[1:])

        if "ps_overall" in report:
            print("\nPS Overall")
            print_table(["channel", "PS"], [[lev, fmt(report["ps_overall"][lev])] for lev in report["levels"]])

    if "ips" in metrics and "ips_per_level" in report:
        print("\nIPS (Integrated Pattern Score)")
        print("  IPS = ((((PCC + 1) / 2) + AS) / 2) * 100")
        ips_data = report["ips_per_level"]
        pentad_labels = report.get("pentad_labels", [])
        if pentad_labels:
            rows = [["channel"] + pentad_labels]
            for level in report["levels"]:
                rows.append([level] + [fmt(v) for v in ips_data[level]["ips"]])
            print_table(rows[0], rows[1:])

            print("\nIPS PCC Component")
            rows = [["channel"] + pentad_labels]
            for level in report["levels"]:
                rows.append([level] + [fmt(v) for v in ips_data[level]["pcc"]])
            print_table(rows[0], rows[1:])

            print("\nIPS AS Component")
            rows = [["channel"] + pentad_labels]
            for level in report["levels"]:
                rows.append([level] + [fmt(v) for v in ips_data[level]["as"]])
            print_table(rows[0], rows[1:])

        if "ips_overall" in report:
            print("\nIPS Overall (weighted)")
            print_table(["channel", "IPS"], [[lev, fmt(report["ips_overall"][lev])] for lev in report["levels"]])

    if args.output:
        out = Path(args.output).expanduser()
        out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"report: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
