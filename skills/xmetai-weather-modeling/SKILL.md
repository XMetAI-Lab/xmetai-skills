---
name: xmetai-weather-modeling
description: Use when working on XMetAI-style weather or climate model repositories, including LazyConfig configs, Zarr or NetCDF meteorological data, static fields, model implementation, training, evaluation, ONNX export, inference deployment, or tensor-shape debugging.
metadata:
  short-description: XMetAI weather model research and operations
---

# XMetAI Weather Modeling

## Start Here

1. Locate the active repo and read local rules: `AGENTS.md`, `RTK.md`, `CONTRIBUTING.md`, README.
2. Check local changes before editing.
3. Classify the task: model development, evaluation/inference, data analysis, or debugging.
4. Load only the matching reference.

## Non-Negotiable Rules

- Zarr writes require explicit approval before any write, overwrite, append, delete, in-place merge, rechunk, or store mutation.
- Prefer read-only inspection and dry runs.
- Follow repo conventions before adding abstractions.
- Never claim train/export/deploy/runtime validation unless it actually ran.
- Keep workflows portable across agent tools unless a runtime is requested.

## Reference Routing

- Model development: `references/model-contracts.md`
- Evaluation and inference: `references/inference-export-deploy.md`
- Data analysis: `references/zarr-static-contracts.md`
- Debugging: `references/shape-debugging.md`

## Script Routing

- `inspect_workspace.py`: repo summary.
- `inspect_zarr_schema.py`: read-only Zarr schema/sample.
- `inspect_static_nc.py`: read-only static NetCDF schema.
- `check_config_contract.py`, `summarize_lazy_config.py`: config checks.
- `build_train_command.py`: print train/eval/export command only.
- `check_onnx_io.py`: ONNX IO summary.
- `zarr_write_guard.py`: mandatory guard for future write scripts.
- `collect_experiment_report.py`: lightweight report.

## Validation Posture

Use the narrowest meaningful validation: config load/contract check, focused unit test, read-only data schema/sample, ONNX IO check, or line-by-line tensor trace.
