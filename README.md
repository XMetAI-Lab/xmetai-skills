# xmetai-skills

Agent skills for XMetAI-style weather model work: model-side development workflows
and evaluation-result reporting.

## Skills

| Skill | Entry | Purpose |
|---|---|---|
| `xmetai-weather-modeling` | [`skills/xmetai-weather-modeling/SKILL.md`](skills/xmetai-weather-modeling/SKILL.md) | Workspace inspection, LazyConfig review, Zarr/NetCDF/static-data checks, model contracts, tensor-shape debugging, training and evaluation **execution**, ONNX export, deployment review. |
| `xmetai-datakit-skills` | [`skills/xmetai-datakit-skills/SKILL.md`](skills/xmetai-datakit-skills/SKILL.md) | Family router: dispatches family-level or ambiguous requests to member skills. |
| `reporting` | [`skills/xmetai-datakit-skills/reporting/SKILL.md`](skills/xmetai-datakit-skills/reporting/SKILL.md) | Consumes finished evaluation products (`manifest.json` + result tables) and produces figures plus a Markdown report. Metric computation is out of scope. |

`SKILL.md` is the only file a host loads, and its `name` + `description` are the only
selection surface. `agents/*.yaml` is optional metadata and does not affect selection.

## Install

### Option 1: Register as agent skills

**Codex** — link each skill directory into the Codex skills folder:

```bash
export CODEX_HOME="${CODEX_HOME:-$HOME/.codex}"
git clone git@github.com:XMetAI-Lab/xmetai-skills.git
mkdir -p "$CODEX_HOME/skills"

ln -s "$(pwd)/xmetai-skills/skills/xmetai-weather-modeling" "$CODEX_HOME/skills/xmetai-weather-modeling"
ln -s "$(pwd)/xmetai-skills/skills/xmetai-datakit-skills" "$CODEX_HOME/skills/xmetai-datakit-skills"
ln -s "$(pwd)/xmetai-skills/skills/xmetai-datakit-skills/reporting" "$CODEX_HOME/skills/reporting"
```

**Claude Code** — copy (or link) each skill into `.claude/skills/<name>/`:

```bash
mkdir -p .claude/skills
cp -r skills/xmetai-weather-modeling .claude/skills/xmetai-weather-modeling
cp -r skills/xmetai-datakit-skills/reporting .claude/skills/reporting
```

Hosts do **not** discover nested `SKILL.md` files. Registering the family container
alone does not register `reporting`; link every skill you want selectable.

### Option 2: Keep this repo beside `xmetai-core` or `xmetai-datakit`

For project-local development, keep this repository as a sibling checkout or submodule
next to the target repository:

```bash
git submodule add git@github.com:XMetAI-Lab/xmetai-skills.git xmetai-skills
git submodule update --init --recursive
```

Then ask the agent to load `xmetai-skills/skills/<skill>/SKILL.md` directly. Do not
vendor `xmetai-core` or `xmetai-datakit` into this repository; the skills inspect the
active workspace where the user is working.

## Dependencies

Only the reporting skill has Python dependencies; the modeling skill uses the standard
library only.

```bash
python -m pip install -r requirements.txt
# with spatial extraction (zarr / netCDF4) and typhoon maps (basemap):
python -m pip install "./skills/xmetai-datakit-skills/reporting[all]"
```

`requirements.txt` only points at the skill's `pyproject.toml`; versions and extras are
maintained there. Installing is optional: the reporting entrypoints add `scripts/` to
`sys.path` themselves, so copying the skill directory is enough to run it.

## Use

### Model-side workflows

Ask the agent to use `xmetai-weather-modeling` for:

- LazyConfig creation or review.
- Zarr, NetCDF, static-data, and dataset-contract inspection.
- Model contract review and tensor-shape debugging.
- Training, evaluation/inference execution, ONNX export, and deployment checks.

Typical read-only helper commands:

```bash
python skills/xmetai-weather-modeling/scripts/inspect_workspace.py /path/to/xmetai-core
python skills/xmetai-weather-modeling/scripts/check_config_contract.py /path/to/config.py
python skills/xmetai-weather-modeling/scripts/inspect_static_nc.py /path/to/static.nc
```

The modeling skill covers *running* an evaluation job and reporting what actually ran.
It does not compute metrics and does not produce analysis reports.

### Evaluation reporting

Ask the agent to use `reporting` when an evaluation result directory already exists:

```bash
python skills/xmetai-datakit-skills/reporting/scripts/render_report.py \
  outputs/results/weather_ts_single_fgvp --result report/fgvp --model FGVP
```

Multi-model comparison (TS family):

```bash
python skills/xmetai-datakit-skills/reporting/scripts/render_report.py \
  outputs/results/weather_ts_single_fgvp --result report/fgvp_vs_fuxi --model FGVP \
  --compare "FuXi=outputs/results/weather_ts_single_fuxi/diagnostics/categorical_wide.csv"
```

Other entrypoints:

| Task | Command |
|---|---|
| Multi-model RMSE (single / ensemble) | `render_det_report.py` / `render_ens_report.py --model NAME=dir` |
| Rainstorm case analysis | `render_rainstorm_report.py --model NAME=dir --process 202503,202521` |
| Pick rainstorm cases | `render_rainstorm_summary.py` → `summary_long.csv` |
| Spatial extraction (server) | `extract_spatial_distribution.py` / `extract_spatial_distribution_ens.py` |
| Spatial maps | `render_spatial_distribution.py --csv ... --model ...` |
| Typhoon statistics | `render_report.py` + `--leads` / `--fixed-leads` / `--bootstrap` |

The input contract (required files per capability, required columns, missing-value
handling) is documented in [`reporting/SKILL.md`](skills/xmetai-datakit-skills/reporting/SKILL.md).

## Verify

From the repository root:

```bash
python -m pytest tests -q
```

- `tests/test_skill_structure.py` checks the modeling skill's entrypoint, references,
  agent metadata, and that its scripts compile.
- `tests/test_reporting.py` copies `reporting` to a temporary directory and runs every
  entrypoint without installation; it skips when the scientific stack is unavailable.

CI compiles the modeling skill's scripts and runs the test suite; the reporting tests
only execute where the scientific stack is installed.

## Repository Layout

```text
skills/
  xmetai-weather-modeling/             Model-side workflow skill.
    SKILL.md
    agents/openai.yaml
    assets/templates/
    references/
    scripts/
  xmetai-datakit-skills/               Datakit skill family.
    SKILL.md                           Family router.
    reporting/                         Evaluation products → figures + REPORT.md.
      SKILL.md  README.md  pyproject.toml
      agents/
      assets/templates/
      references/
      scripts/
        render_*.py  extract_*.py      CLI entrypoints
        reporting/                     Runtime library
          dispatch.py                  Capability detection + dispatch
          core/                        Style, tables, writer, semantics
          sources/                     Contracts, ERA5, layouts, track geometry
          spectra/                     Metric base, spherical bands
          reports/                     TS / field / RMSE / rainstorm / typhoon / spatial
docs/                                  Website documentation pages.
tests/                                 Skill structure and standalone tests.
requirements.txt                       Environment setup (points at reporting/pyproject.toml).
```

## Documentation

`docs/` holds website documentation pages. Installation and local setup instructions
belong in this README so there is one canonical setup path.

- `docs/usage.md`: website-facing usage overview.
- `docs/tool-compatibility.md`: website-facing compatibility notes.
- `docs/release.md`: release checklist.

## Safety Boundary

- Zarr reads are normal inspection work. Writes, appends, overwrites, deletes,
  in-place merges, rechunks, or any store mutation require explicit user approval
  in the active conversation.
- The reporting skill only reads its inputs; report output must go to a new or
  empty directory and never overwrites an existing report.

## Adding a Skill

1. Create `<name>/` with `SKILL.md`, `agents/`, `assets/`, `references/`, and `scripts/`.
2. Put trigger phrases in `SKILL.md`'s `description`; keep `name` equal to the
   directory name.
3. Keep `SKILL.md` a router; long-form rules belong in `references/`.
4. Add the skill to the family router table (if it belongs to a family) and to this README.
5. Register it with each host you use.
