# Evaluation Planning

Use this reference for offline evaluation of XMetAI model forecasts: reading the
`pred_*.nc` / `obs_*.nc` outputs produced during evaluation, computing metrics,
and producing result visualizations. It documents the evaluation input contract
verified against `D:\xmetai-core` (as of 2026-08) and the design rules for new
evaluation scripts.

## Evaluation Inputs: `pred_*.nc` / `obs_*.nc`

`base_predictor.test()` writes one prediction file and one target file per
initialization time when `save_dir` exists:

- `pred_{YYYYMMDD}.nc` — model forecast (physical values)
- `obs_{YYYYMMDD}.nc` — reference target for the same forecast times

Both files share the same layout:

- A single `data` variable, dims `(time, level, lat, lon)`.
- `time` — forecast lead times: `init_time + freq * i` hours for `i` in the
  config's `test_frames`.
- `level` — the model's `test_names` (an evaluation subset of channels, not the
  full channel set). Example s2s: `z500, u200, u850, msl, t2m, sst, ttr, tp`.
- `lat` — `linspace(90, -90, H)`.
- `lon` — `linspace(0, 360, W)`.

### Verified facts

- **Values are physical, not normalized.** `process_output()` calls
  `inv_normalize()` on both output and target: `x = x * std + mean`, and
  log-transformed channels (e.g. `tp`) are restored with `exp(x) - 1`. The
  offline evaluation script does NOT need mean/std sidecars to interpret the
  files.
- **No spatial crop.** `test_rect` exists in `base_predictor` but is not set by
  any current config, so saved fields are global.
- **`lon` coordinates are approximate.** They are generated with
  `linspace(0, 360, W)` (endpoint included), while the source ERA5 grid is
  `0 .. 358.5` with a 1.5-degree step (240 points). The endpoint mismatch is
  under one grid cell, so it is fine for plotting and domain-mean statistics,
  but not for precise point matching (typhoon center location, station
  comparison). Use the original data grid coordinates when sub-grid accuracy
  matters.

## Model Evaluation Entry Points

| Model family | `test_names` | `test_frames` | `freq` | Grid |
|---|---|---|---|---|
| s2s (`s2s_vit`, `s2s_afnonet`, `s2s_group_vae`, `s2s_puyun`, `s2s_diffusion`, `s2s_graphcast`) | 8 channels: `z500, u200, u850, msl, t2m, sst, ttr, tp` | 1–42 days | 24 h | 1.5 deg, 121×240 |
| iwc (`iwc_vit`, `iwc_graphcast`) | 6 channels: `z500, msl, t2m, sst, ttr, tp` | 1–60 (6-hour steps, 15 days) | 6 h | 0.25 deg, 720×1440 |
| land (`land_vit`) | 10 channels: s2s 8 + `stl1, swvl1` | 1–42 days | 24 h | ERA5-Land grid |

- `save_dir` convention: `output_dir + "/pred/"` in the model config.
- Evaluation is launched with `bash scripts/train.bash --stage eval --model
  configs/<model>.py` (Linux/macOS only; Windows exits with "Unsupported OS").

## Core Metrics Library

All metrics live in `xmetai/metrics/` and follow the `DatasetEvaluator`
contract: `reset()`, `process(inputs, outputs)`, `evaluate()`.

- `process` receives a dict with `output` (prediction) and `target`
  (observation) tensors, shape `(B, T, C, H, W)`, where `C` follows
  `test_names` order.
- `MSE` reports weighted RMSE per channel and lead time with keys like
  `"z500 @ 024h"`.
- `ThreatScore`, `ProbabilityOfDetection`, `FalseAlarmRate`, `FrequencyBias`
  are threshold contingency metrics with optional neighborhood pooling
  (`neighborhood_size`, default 1 = pointwise), per threshold and lead time.
- `PS` / `ClimatePSScore`, `IPS`, `TCC` are climate-skill metrics.
- Radar echo metrics (`FSS`, `SSIM`, `LPIPS`, `RAPSD` and radar variants)
  target MeteoNet/radar fields.
- `evaluate()` uses `comm.gather`; reusing these classes offline requires a
  single-process compatibility layer, or replicating their math in the offline
  script with the same thresholds and neighborhood convention.

## Offline Evaluation (`scripts/evaluate_pred.py`)

The offline script reads `pred_*.nc` / `obs_*.nc` directly — no denormalization
is needed because files already contain physical values. It currently provides:

- RMSE per channel and lead (unweighted, NaN grid points skipped).
- TS/POD/FAR/FrequencyBias for one channel at configurable thresholds with an
  optional odd neighborhood (default 1 = pointwise), matching the core
  convention.
- A compact markdown table plus a machine-readable JSON report (`--output`).

Metric parity notes:

- Core weights RMSE with the channel weight convention
  (`max(0.2, level/1000)` with optional land/ocean correction, normalized to
  max 1); the offline script is unweighted for now.
- Core replaces NaN with 0 before training (`torch.nan_to_num`); the offline
  script skips invalid grid points instead, which differs for channels with
  missing values (e.g. `sst` over land).
- Threshold units follow the channel's physical unit (for `tp`: metres, so
  0.1/10/25/50 mm/day are `0.0001, 0.01, 0.025, 0.05`).

## Evaluation Visualization (`scripts/visualize_eval.py`)

Renders figures from the same paired files:

- `compare_<channel>_lead<NN>.png`: prediction / observation / error
  three-panel maps for the requested channels and leads (first init date).
- `compare_<channel>_leads.gif`: the same compare frames animated over leads
  (`--gif`), core-style.
- `rmse_curves.png`: per-channel RMSE vs forecast lead.
- `ts_curves.png`: TS vs forecast lead, one line per threshold.
- `threshold_metrics.png`: TS/POD/FAR vs threshold, one line per lead.

The maps use a Plate Carree projection with Natural Earth 110m
ocean/land/coastline shapes (first use downloads the shapefiles once; set
``CARTOPY_DATA_DIR`` to a writable, persistent cache in restricted
environments). The 0..360 lon convention from the pred files is converted to
-180..180 for plotting.

## Core Alignment (Updated 2026-08-18)

The offline evaluation scripts have been aligned with core's metric computation:

### NaN Handling

- **Core**: `torch.nan_to_num(input, nan=0.0)` — replaces NaN with 0 before evaluation.
- **Offline script**: Now uses `nan_to_zero()` to match. Previously used `nanmean` which skipped NaN grid points.
- **Impact**: `sst` channel RMSE changes because NaN values over land are now treated as 0 instead of being skipped.

### Spatial Weighting

- **Core**: Loads `weight.nc` buffer (latitude-based, shape `(1, H, 1)`) and multiplies element-wise before taking the mean.
- **Offline script**: Supports `--weight-file weight.nc` to replicate. Without this flag, RMSE is unweighted (same as before).
- **Note**: The `channel_weights` buffer (per-channel weights based on pressure levels) is used in training loss, not in the MSE evaluator. The MSE evaluator only uses the spatial `weight` buffer.

### Top-Percent Extreme Value RMSE

- **Core**: `MSE` evaluator supports `top_percent` parameter to compute RMSE over the top X% of grid points ranked by |target|.
- **Offline script**: Supports `--top-percent 0.02` (for 2%) to replicate.

### Neighborhood Dilation

- **Core**: `F.max_pool2d(mask.float(), kernel_size=neighborhood_size, stride=1, padding=neighborhood_size // 2) > 0` — zero-padded, no wrap-around at grid boundaries.
- **Offline script**: Now uses `np.pad` with `constant_values=False` to match core's zero-padding behavior. Previously used `np.roll` which wrapped around at edges.

### Threshold Metric Counts

- **Core**: Accumulates raw counts (hit, false_alarm, miss) per frame_index, then computes final scores from aggregated counts.
- **Offline script**: Now accumulates raw counts across all init dates before computing TS/POD/FAR/FB, matching core's aggregation logic.

## Visualization Modes (Updated 2026-08-18)

The ``--mode`` parameter controls the output layout:

| Mode | Output | Use Case |
|------|--------|----------|
| ``compare`` (default) | 3-panel: Prediction / Observation / Error | Full comparison |
| ``pred-obs`` | 2-panel: Prediction / Observation | Quick visual comparison |
| ``pred`` | 1-panel: Prediction only | Single variable inspection |

Examples:

.. code-block:: bash

    # Single prediction map for precipitation
    python visualize_eval.py --pred-dir path/to/pred --channels tp --mode pred

    # Prediction vs observation for z500
    python visualize_eval.py --pred-dir path/to/pred --channels z500 --mode pred-obs

Output file naming:

- ``compare_tp_lead01.png`` (compare mode)
- ``pred-obs_tp_lead01.png`` (pred-obs mode)
- ``pred_tp_lead01.png`` (pred mode)

## Multi-Init Composite Error (Updated 2026-08-19)

The ``--mode composite`` option aggregates errors across multiple init dates:

- **Mean Bias**: Average of (pred - obs) across all init dates. Reveals systematic over/under-prediction.
- **RMSE**: Root mean squared error across init dates. Shows error magnitude independent of sign.

Usage:

.. code-block:: bash

    # Composite error using all init dates in pred-dir
    python visualize_eval.py --pred-dir path/to/pred --mode composite --channels tp,z500

    # Composite error for specific init dates
    python visualize_eval.py --pred-dir path/to/pred --mode composite --init-date 20200101,20200102

Output: ``composite_<channel>_lead<NN>.png`` (2-panel: Mean Bias / RMSE)

The ``--init-date`` parameter also works with other modes to select a specific init date
instead of defaulting to the first one found.

## Region Cropping (Updated 2026-08-18)

The ``--region`` parameter supports regional visualization:

- **Direct coordinates**: ``--region lon_min,lon_max,lat_min,lat_max`` (cartopy convention)
- **Region name**: The LLM resolves region names to coordinates before calling the script

Common region coordinates:

| Region | Coordinates (lon_min,lon_max,lat_min,lat_max) |
|--------|----------------------------------------------|
| China | ``70,137,17,55`` |
| East China | ``113,125,21.5,38.5`` |
| North China | ``97,126.5,34,54`` |
| South China | ``104,118.5,18,26.5`` |
| Japan | ``129,146,31,45`` |
| Korea | ``124,132,33,43`` |

When the user requests a region by name (e.g., "show me East China"), the LLM
determines the appropriate coordinates and passes them to the script.

## Frequency-Aware Labeling (Updated 2026-08-18)

Both evaluation scripts support `--freq` to specify the forecast frequency in hours:

- `--freq 24` (default): S2S/Land daily forecasts. Labels: `D1`, `D2`, `Day 1`, `Day 2`
- `--freq 6`: IWC 6-hourly forecasts. Labels: `6h`, `12h`, `18h`

This affects:
- Table column headers in `evaluate_pred.py`
- X-axis labels in `visualize_eval.py` curves
- Frame titles in compare maps and GIFs

## Pending Confirmations

- Multi-member (ensemble) output: `test()` calls `squeeze()`, which drops
  size-1 dims; how multi-member prediction tensors are saved (if any model
  uses `members > 1`) has not been verified on real inference.
- Physical units per channel follow `visualize_pred_gif.py`'s `VAR_META`
  (e.g. msl in Pa, tp in m, t2m in K); verify before publishing results.




