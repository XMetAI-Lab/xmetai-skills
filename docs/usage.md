# Usage

This page is written for the documentation website. For installation and local setup, use the repository README.

Use `xmetai-weather-modeling` for LazyConfig, Zarr/NetCDF/static data, model code, training/eval, ONNX/deploy, or tensor-shape debugging.

## Workflow

1. Inspect repo rules and local changes.
2. Start from `skills/xmetai-weather-modeling/SKILL.md`.
3. Prefer read-only scripts and dry runs.
4. Ask before Zarr mutation.
5. Report only validation that actually ran.

## Task Routing

- Workspace and repo conventions: `references/workspace-contract.md`
- LazyConfig patterns: `references/lazyconfig-patterns.md`
- Zarr, NetCDF, and static data: `references/zarr-static-contracts.md`
- Model contracts and shape debugging: `references/model-contracts.md`, `references/shape-debugging.md`
- Training and evaluation operations: `references/training-eval-ops.md`
- ONNX export and deployment: `references/inference-export-deploy.md`
