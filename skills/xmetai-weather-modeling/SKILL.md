---
name: xmetai-weather-modeling
description: Use when working on XMetAI-style weather or climate model repositories, including meteorological data requirement extraction, pre-download planning, multi-config download plans, format conversion chains, data preprocessing, format detection, NetCDF or Zarr conversion, preprocessing validation, LazyConfig configs, Zarr or NetCDF data, static fields, model implementation, training, evaluation, ONNX export, inference deployment, or tensor-shape debugging.
metadata:
  short-description: XMetAI weather model research and operations
---

# XMetAI Weather Modeling

## Start Here

1. Locate the active repo and read local rules: `AGENTS.md`, `RTK.md`, `CONTRIBUTING.md`, README.
2. Check local changes before editing.
3. Classify the task: model development, evaluation/inference, data analysis, data download planning, or debugging.
4. Load only the matching reference.

## Non-Negotiable Rules

- Zarr writes require explicit approval before any write, overwrite, append, delete, in-place merge, rechunk, or store mutation.
- Prefer read-only inspection and dry runs.
- Prefer download planning and dry runs before network or filesystem writes.
- Before a large download, confirm the data source, variables, time range, destination, estimated size, and overwrite policy.
- Never store API keys, tokens, passwords, or other credentials in manifests, logs, reports, or repository files.
- Follow repo conventions before adding abstractions.
- Never claim train/export/deploy/runtime validation unless it actually ran.
- Keep workflows portable across agent tools unless a runtime is requested.

## Reference Routing

- Model development: `references/model-contracts.md`
- Evaluation and inference: `references/inference-export-deploy.md`
- Evaluation planning (offline metrics, pred/obs contract, visualization): `references/evaluation-planning.md`
- Data analysis: `references/zarr-static-contracts.md`
- Data download planning: `references/data-download-planning.md`
- Data preprocessing: `references/data-preprocessing.md`
- Debugging: `references/shape-debugging.md`

## Data Routing

- Use data analysis when existing Zarr, NetCDF, or static data needs read-only inspection, comparison, or validation.
- Use data download planning when data is missing and the task is to extract requirements, identify a source, or produce a pre-download plan.
- Data download planning stops before network requests, file creation, conversion, statistics generation, Zarr writes, or training-readiness validation.
- Use data preprocessing when downloaded data exists and needs format detection, normalization, conversion, or post-conversion validation.
- Use data analysis when data already exists and needs schema, integrity, quality, statistics, or training-readiness checks.

## Script Routing

- `inspect_workspace.py`: repo summary.
- `inspect_zarr_schema.py`: read-only Zarr schema/sample.
- `inspect_static_nc.py`: read-only static NetCDF schema.
- `inspect_data_format.py`: read-only format detection and metadata summary.
- `check_config_contract.py`, `summarize_lazy_config.py`: config checks.
- `build_train_command.py`: print train/eval/export command only.
- `check_onnx_io.py`: ONNX IO summary.
- `zarr_write_guard.py`: mandatory guard for future write scripts.
- `convert_to_zarr.py`: dry-run conversion plan, then guarded NetCDF/Zarr to Zarr.
- `compute_sidecars.py`: per-channel mean/std/weight sidecar generation.
- `merge_normalize.py`: merge multiple Zarr stores and normalize them as one dataset.
- `evaluate_pred.py`: offline RMSE / threshold metrics (TS, POD, FAR, FB) / TCC (Temporal Correlation Coefficient) from `pred_*.nc` / `obs_*.nc`.
- `visualize_eval.py`: map compare/error/pred panels, multi-init composite error (mean bias + RMSE), RMSE/TS/TCC curves, threshold metric plots.
- `collect_experiment_report.py`: lightweight report.

## Validation Posture

Use the narrowest meaningful validation: config load/contract check, focused unit test, read-only data schema/sample, ONNX IO check, or line-by-line tensor trace.




