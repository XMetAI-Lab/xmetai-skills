# XMetAI Evaluation Reporting

Convert **existing evaluation products** into figures and a Markdown report.
Do not recompute evaluation metrics, infer missing data, or identify capabilities
from directory names.

[`SKILL.md`](./SKILL.md) is the authoritative source for capability contracts,
trigger scenarios, and mandatory rules. This README describes the package,
installation, and usage. The final skill location is `skills/evaluation-reporting/`;
the repository router is `skills/SKILL.md`, and downloading is handled separately
by `skills/dataset-downloader/`.

## Capabilities

| Workflow | Input | Output |
|---|---|---|
| Deterministic/ensemble precipitation TS | `diagnostics/categorical_wide.csv`; ensembles may also provide `probability_wide.csv` | Up to 11 SVG figures and six-chapter `REPORT.md` |
| Continuous-field RMSE, ACC, activity, and spectra | `scores.csv` | PNG figures and `REPORT.md` |
| Multi-model RMSE batch comparison | N standard product directories | Multi-model comparison report |
| Typhoon track/intensity | `typhoon/tc*.csv` + `*_meta.json` | Six-chapter report, statistics CSV/JSON, and per-storm track maps |
| Rainstorm cases | Both `{id}_24h/` and `{id}_total/` product segments | Report organized by event |
| Spatial distributions | Wide CSV produced by extraction scripts | Three-row deterministic or four-row ensemble SVG maps |

## Input Contract

The unified entry point accepts **standard product directories only**:

```text
<product-directory>/
  manifest.json                 # Required; resolved_config.pipeline identifies capability
  scores.csv                    # Required for continuous fields
  diagnostics/
    categorical_wide.csv        # Required for TS
    probability_wide.csv        # Optional for ensembles
  typhoon/                      # Required for typhoon verification
    tc<id>_<initialization>.csv
    tc<id>_<initialization>_meta.json
```

See **Input Contract: Standard Evaluation Product Directories** in
[`SKILL.md`](./SKILL.md) for required columns, missing-column handling, grade
normalization, and criteria for marking inputs as not directly comparable.
**Missing required columns cause explicit errors listing the missing fields;
defaults are not substituted.**

## Installation

Python >= 3.10 is required. From the repository root:

```bash
python -m pip install -r requirements.txt
# For spatial extraction from Zarr/NetCDF or typhoon maps:
python -m pip install "./skills/evaluation-reporting[all]"
```

The root `requirements.txt` delegates to this directory's `pyproject.toml` through
`-e ./skills/evaluation-reporting[test]`. Maintain dependency versions only in
`pyproject.toml`, not a duplicate requirements list.

**Entry points can also run without installing this package**, provided their
dependencies are available. They add their own `scripts/` directory to `sys.path`,
so the entire `evaluation-reporting/` directory can be copied to another location.
The internal Python package is `reporting`, with an underscore rather
than the hyphen used in the skill directory.

## Usage

From the repository root:

```bash
python skills/evaluation-reporting/scripts/render_report.py <product-directory> \
  --result report/<name> --model <model-name>
```

Multi-model TS comparison:

```bash
python skills/evaluation-reporting/scripts/render_report.py \
  outputs/results/weather_ts_single_fgvp \
  --result report/fgvp_vs_fuxi --model FGVP \
  --compare "FuXi=outputs/results/weather_ts_single_fuxi/diagnostics/categorical_wide.csv"
```

Other entry points, located under `skills/evaluation-reporting/scripts/`:

| Scenario | Command or workflow |
|---|---|
| Multi-model RMSE, single-member/ensemble | `render_det_report.py` / `render_ens_report.py --model NAME=DIRECTORY` |
| Rainstorm case analysis | `render_rainstorm_report.py --model NAME=DIRECTORY --process 202503,202521` |
| Rainstorm event selection | `render_rainstorm_summary.py` → `summary_long.csv` |
| Spatial extraction on the server | `extract_spatial_distribution.py` / `extract_spatial_distribution_ens.py` |
| Spatial rendering | `render_spatial_distribution.py --csv ... --model ...` |
| Typhoon statistics | `render_report.py` with `--leads`, `--fixed-leads`, and `--bootstrap` |

## Directory Structure

```text
skills/evaluation-reporting/
├── SKILL.md           # Triggers, input contract, capability routing, mandatory rules
├── README.md          # This document
├── pyproject.toml     # Standalone installation and dependencies
├── agents/            # Location for openai.yaml Codex metadata; currently empty
├── assets/templates/  # Six report formatting contracts
├── references/        # Nine documents on statistical definitions and interpretation
└── scripts/
    ├── _bootstrap.py  _spatial_common.py
    ├── render_*.py  extract_*.py     # Nine CLI entry points
    └── reporting/        # Bundled runtime
        ├── dispatch.py              # Capability detection and dispatch
        ├── core/                    # Styling, tables, writers, semantics
        ├── sources/                 # Contracts, ERA5, layouts, typhoon geometry
        ├── spectra/                 # Metric base classes and spherical bands
        └── reports/                 # Reporting families
```

## Tests

From the repository root:

```bash
python -m pytest tests -q
```

`tests/test_reporting.py` copies the skill into an isolated test directory to verify
that all entry points work outside the original repository without installing the
package. Tests skip automatically when scientific-computing dependencies are unavailable.

## Boundaries

- **Do not recompute evaluation metrics:** evaluation pipelines produce them;
  this skill consumes their products.
- **Do not synthesize missing tables:** a missing wide table is an error, not a
  reason to convert long tables into a substitute.
- **Do not identify capabilities from directory names:** use
  `manifest.json` → `resolved_config.pipeline`.
- Inputs are read-only. Output directories must not exist or must be empty.
