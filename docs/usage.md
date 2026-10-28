# Usage

This page is written for the documentation website. For installation and local setup, use the repository README.

Start from [the skills router](../skills/SKILL.md) when the required capability is
unclear. Concrete skills are peers under `skills/`:

- [Dataset downloader](../skills/dataset-downloader/SKILL.md): download planning,
  configuration, preview, execution, and recovery through Datakit.
- [Evaluation reporting](../skills/evaluation-reporting/SKILL.md): figures and
  reports from completed evaluation products.
- [Weather modeling](../skills/xmetai-weather-modeling/SKILL.md): model development,
  evaluation/inference execution, data analysis, and debugging.

## Workflow

1. Inspect repository rules and local changes.
2. Load the selected skill's `SKILL.md` and verify its required inputs.
3. Prefer read-only scripts and dry runs.
4. Ask before Zarr mutation.
5. Report only validation that actually ran.

## Task Routing

- Model development: `references/model-contracts.md`
- Evaluation and inference: `references/inference-export-deploy.md`
- Data analysis: `references/zarr-static-contracts.md`
- Debugging: `references/shape-debugging.md`
