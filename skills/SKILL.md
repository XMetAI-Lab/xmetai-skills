---
name: skills
description: >
  Use when selecting among XMetAI skills, reviewing available capabilities,
  resolving ambiguous weather-data or modeling requests, or coordinating
  workflows spanning dataset downloads, weather modeling, and evaluation-result
  reporting. Route concrete requests to dataset-downloader, evaluation-reporting,
  or xmetai-weather-modeling; each destination owns its contracts and commands.
metadata:
  short-description: XMetAI skill selection and workflow routing
---

# XMetAI Skills Router

This entrypoint routes requests to the three peer skills. It does not implement
downloads, model operations, metric computation, or report generation.

## Procedure

1. Identify the required capability using the routing table.
2. Open the destination's `SKILL.md` and follow its input contract and workflow.
   Commands, column names, data rules, and outputs belong to that skill.
3. If the destination is ambiguous, ask one question that determines ownership,
   such as whether evaluation products already exist. Do not infer missing inputs.
4. If a destination is missing, empty, or not installed, report the missing
   capability and stop that workflow. Do not substitute remembered commands.
5. For multi-stage requests, load each applicable skill at the relevant stage.
   Preserve its authorization requirements; completing one stage does not
   authorize the next.

## Routing

| Request | Destination | Scope |
|---|---|---|
| Download ERA5, ERA5-Land, or ORAS5; select variables, dates, regions, or levels; configure credentials; preview, execute, or resume downloads | [dataset-downloader](./dataset-downloader/SKILL.md) | Requires the Datakit repository and invokes its existing downloaders |
| Generate figures, Markdown reports, model comparisons, typhoon verification, rainstorm analyses, or spatial maps from completed evaluation products | [evaluation-reporting](./evaluation-reporting/SKILL.md) | Consumes evaluation products; does not execute evaluation pipelines or compute their metrics |
| Inspect model configurations, Zarr/NetCDF/static fields, model contracts, or tensor shapes; run training or evaluation/inference; export ONNX or review deployment | [xmetai-weather-modeling](./xmetai-weather-modeling/SKILL.md) | Operates on the active model repository |
| Select a skill, inspect available capabilities, or coordinate an ambiguous multi-stage workflow | This router | Select destinations without duplicating their implementation |
| Compute metrics not provided by a selected workflow | The relevant evaluation pipeline repository | Do not treat evaluation-reporting as a metric-computation pipeline |

## Mandatory Boundaries

- Maintain member contracts only in their own documentation.
- Do not produce numeric results, figures, or reports through this router.
- Ask for clarification when multiple destinations remain plausible.
- Report unavailable destinations explicitly.
- When adding a skill, create its directory beside the existing skills and update
  this table and the repository README.

## Host Registration

Hosts discover registered directories, not arbitrary nested skill files. Register
each concrete skill separately so that its own trigger description is available.
The router at `skills/SKILL.md` is optional for specific requests.

From the parent directory of this checkout, a Codex registration can use:

```bash
export CODEX_HOME="${CODEX_HOME:-$HOME/.codex}"
mkdir -p "$CODEX_HOME/skills"
ln -s "$(pwd)/xmetai-skills/skills/dataset-downloader" "$CODEX_HOME/skills/dataset-downloader"
ln -s "$(pwd)/xmetai-skills/skills/evaluation-reporting" "$CODEX_HOME/skills/evaluation-reporting"
ln -s "$(pwd)/xmetai-skills/skills/xmetai-weather-modeling" "$CODEX_HOME/skills/xmetai-weather-modeling"
ln -s "$(pwd)/xmetai-skills/skills" "$CODEX_HOME/skills/skills"
```

Inspect existing registrations rather than overwriting them. VS Code / Copilot
can link concrete skills into `.github/skills/` or `~/.copilot/skills/`, using names
matching their frontmatter. Register the router under `skills` if required.
Registering only the router does not independently index its children. When both
are registered, concrete requests should select concrete skills; this router
handles skill selection and ambiguous requests.

## Validation

Use the repository's structure tests to verify destinations and metadata. Follow
each destination's validation instructions for scripts, data contracts, and output
regressions; router validation does not establish runtime success.
