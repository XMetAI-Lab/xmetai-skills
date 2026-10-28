# xmetai-skills

Agent skills for meteorological and oceanographic data downloads, XMetAI-style
weather model development, and evaluation-result reporting.

## Skills

| Skill | Entry | Purpose |
|---|---|---|
| `skills` | [Router](skills/SKILL.md) | Selects concrete skills for ambiguous requests and coordinates multi-stage workflows. |
| `dataset-downloader` | [Download workflow](skills/dataset-downloader/SKILL.md) | Plans, previews, executes, and resumes ERA5, ERA5-Land, and ORAS5 downloads using the Datakit repository. |
| `evaluation-reporting` | [Reporting workflow](skills/evaluation-reporting/SKILL.md) | Consumes completed evaluation products and produces figures and Markdown reports; does not execute evaluation pipelines or compute their metrics. |
| `xmetai-weather-modeling` | [Modeling workflow](skills/xmetai-weather-modeling/SKILL.md) | Inspects workspaces, LazyConfig, data and model contracts, and tensor shapes; supports training, evaluation/inference execution, ONNX export, and deployment review. |

Hosts select skills through `SKILL.md` frontmatter, particularly `name` and
`description`. Files under `agents/` provide optional metadata, not selection rules.

## Installation

### Register Agent Skills

For Codex, clone the repository and link each required skill directory:

```bash
export CODEX_HOME="${CODEX_HOME:-$HOME/.codex}"
git clone git@github.com:XMetAI-Lab/xmetai-skills.git
mkdir -p "$CODEX_HOME/skills"

ln -s "$(pwd)/xmetai-skills/skills/dataset-downloader" "$CODEX_HOME/skills/dataset-downloader"
ln -s "$(pwd)/xmetai-skills/skills/evaluation-reporting" "$CODEX_HOME/skills/evaluation-reporting"
ln -s "$(pwd)/xmetai-skills/skills/xmetai-weather-modeling" "$CODEX_HOME/skills/xmetai-weather-modeling"
# Optional selection router:
ln -s "$(pwd)/xmetai-skills/skills" "$CODEX_HOME/skills/skills"
```

For Claude Code, run from the repository root:

```bash
mkdir -p .claude/skills
cp -r skills/dataset-downloader .claude/skills/dataset-downloader
cp -r skills/evaluation-reporting .claude/skills/evaluation-reporting
cp -r skills/xmetai-weather-modeling .claude/skills/xmetai-weather-modeling
# Optional router; its relative destination links require the child directories:
mkdir -p .claude/skills/skills
cp skills/SKILL.md .claude/skills/skills/SKILL.md
ln -s ../dataset-downloader .claude/skills/skills/dataset-downloader
ln -s ../evaluation-reporting .claude/skills/skills/evaluation-reporting
ln -s ../xmetai-weather-modeling .claude/skills/skills/xmetai-weather-modeling
```

Inspect existing registrations before changing them; do not overwrite existing
skills. Hosts do not recursively register nested `SKILL.md` files. Register concrete
skills separately even when the router is installed.

For VS Code / Copilot, link the concrete directories into workspace
`.github/skills/` or user-level `~/.copilot/skills/`, preserving frontmatter names.
The optional router directory is named `skills`.

### Use a Sibling Checkout or Submodule

Keep this repository beside the target `xmetai-core` or `xmetai-datakit` checkout,
or add it as a submodule:

```bash
git submodule add git@github.com:XMetAI-Lab/xmetai-skills.git xmetai-skills
git submodule update --init --recursive
```

Load `xmetai-skills/skills/<skill>/SKILL.md` directly, or use
`xmetai-skills/skills/SKILL.md` for selection. Do not copy model or Datakit source
into this repository; skills inspect the active target workspace.

## Dependencies

The modeling helpers and download dispatcher use the Python standard library.
The download report renderer requires PyYAML. Download execution and preview use
dependencies maintained by the Datakit downloaders; see the
[download README](skills/dataset-downloader/README.md).

Evaluation reporting requires the scientific Python dependencies declared in its
`pyproject.toml`:

```bash
python -m pip install -r requirements.txt
# Optional spatial extraction and typhoon-map dependencies:
python -m pip install "./skills/evaluation-reporting[all]"
```

`requirements.txt` delegates dependency versions and extras to that project.
Package installation is not required for standalone entrypoint discovery:
scripts add their own `scripts/` directory to `sys.path`, but their runtime
dependencies must still be available.

## Usage

### Dataset Downloads

Use `dataset-downloader` to select data, generate task-specific configurations,
preview a plan, or execute and verify a confirmed download:

```bash
python skills/dataset-downloader/scripts/run_download.py \
  --repo-root /absolute/path/to/xmetai-datakit \
  --source cds /absolute/path/to/batch.yaml
```

The default is an offline preview, not a download. Execution requires `--execute`,
confirmation of the concrete plan and storage directory, and source-specific
authentication. Follow the [download contract](skills/dataset-downloader/SKILL.md).

### Model-Side Workflows

Use `xmetai-weather-modeling` for configuration review, Zarr/NetCDF/static-data and
model-contract inspection, tensor-shape debugging, training, evaluation/inference
execution, ONNX export, and deployment checks.

```bash
python skills/xmetai-weather-modeling/scripts/inspect_workspace.py --root /path/to/xmetai-core
python skills/xmetai-weather-modeling/scripts/check_config_contract.py --config /path/to/config.py
python skills/xmetai-weather-modeling/scripts/inspect_static_nc.py --input /path/to/static.nc
```

This skill supports executing evaluation jobs and reporting observed execution.
It is not a metric-computation implementation or an analysis-report generator.

### Evaluation Reporting

Use `evaluation-reporting` when evaluation products already exist:

```bash
python skills/evaluation-reporting/scripts/render_report.py \
  outputs/results/weather_ts_single_fgvp --result report/fgvp --model FGVP
```

For a TS-family multi-model comparison:

```bash
python skills/evaluation-reporting/scripts/render_report.py \
  outputs/results/weather_ts_single_fgvp --result report/fgvp_vs_fuxi --model FGVP \
  --compare "FuXi=outputs/results/weather_ts_single_fuxi/diagnostics/categorical_wide.csv"
```

| Task | Entrypoint |
|---|---|
| Multi-model deterministic or ensemble RMSE | `render_det_report.py` or `render_ens_report.py --model NAME=dir` |
| Rainstorm case analysis | `render_rainstorm_report.py --model NAME=dir --process 202503,202521` |
| Rainstorm case selection | `render_rainstorm_summary.py`, producing `summary_long.csv` |
| Server-side spatial extraction | `extract_spatial_distribution.py` or `extract_spatial_distribution_ens.py` |
| Spatial maps | `render_spatial_distribution.py --csv ... --model ...` |
| Typhoon statistics | `render_report.py` with `--leads`, `--fixed-leads`, or `--bootstrap` |

See the [reporting contract](skills/evaluation-reporting/SKILL.md) for capability-specific
files, required columns, missing-value handling, and output constraints.

## Verification

From the repository root:

```bash
python -m pytest tests -q
```

Structure tests verify the router, peer skills, references, metadata, naming, and
documentation language. Downloader tests cover dispatch, credential setup,
templates, and report rendering. Real-Datakit compatibility and live CDS tests
require explicit opt-in; ordinary tests do not submit downloads.

Reporting tests copy the skill to a temporary host directory and exercise its
entrypoints and runtime modules without package installation. They skip if the
scientific stack is unavailable. CI runs the suite and compiles modeling helpers;
reporting runtime coverage requires an environment containing its dependencies.

## Repository Layout

```text
skills/
  SKILL.md                            Cross-skill router
  dataset-downloader/
    SKILL.md  README.md
    agents/  assets/templates/  references/  scripts/
  evaluation-reporting/
    SKILL.md  README.md  pyproject.toml
    agents/  assets/templates/  references/
    scripts/
      render_*.py  extract_*.py        CLI entrypoints
      reporting/           Python runtime package
        dispatch.py
        core/  sources/  spectra/  reports/
  xmetai-weather-modeling/
    SKILL.md
    agents/  assets/templates/  references/  scripts/
docs/                                 Website documentation
tests/                                Structure and runtime tests
requirements.txt                      Evaluation-reporting installation
```

## Documentation

Installation and local setup are maintained in this README.
Website-facing pages are [usage](docs/usage.md),
[tool compatibility](docs/tool-compatibility.md), and the
[release checklist](docs/release.md).

## Safety Boundaries

- Obtain explicit approval in the current conversation before any Zarr mutation,
  including writes, appends, overwrites, deletes, in-place merges, or rechunking.
- Evaluation reporting reads inputs without modification. Outputs must use a new
  or empty directory and must not overwrite an existing report.
- Downloads require plan and destination confirmation. Credentials remain private;
  authentication setup and preview do not authorize execution.

## Adding a Skill

1. Create `skills/<name>/` with `SKILL.md`, `agents/`, `assets/`, `references/`,
   and `scripts/` as required.
2. Match frontmatter `name` to the directory and include specific triggers in
   `description`.
3. Keep operational contracts in the concrete skill and extended material in its
   references.
4. Update the top-level router, this README, and directly related tests.
5. Register the skill separately in each required host.
