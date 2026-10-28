---
name: evaluation-reporting
description: >
  Use when analyzing existing meteorological evaluation products, generating figures
  or paper-style reports, comparing models or versions, diagnosing problems, or
  assessing module-change benefits and costs. Covers deterministic and ensemble
  precipitation TS, POD, FAR, BIAS and AROC/BSS; heavy precipitation and daily/event-total
  rainstorm case studies; continuous-field RMSE, ACC, activity, spectra and subset
  comparisons; typhoon error reports and tropical cyclone forecast verification,
  including track errors, AT/CT, intensity MAE/ME and paired ensemble comparisons;
  and monthly deterministic/ensemble spatial distributions and bias maps. Accept
  standard product directories with manifest.json and categorical_wide.csv, scores.csv
  or typhoon products. Route by product metadata, not directory names; respect the
  selected primary model. Produce quantitative figures, sequential captions, section
  summaries and evidence-based recommendations without rerunning evaluation.
metadata:
  version: 4.0
  author: Shen Zhewen (szw)
---

# XMetAI Evaluation Results: Figures and Reports

## Standalone Operation

Copy the entire `skills/evaluation-reporting/` directory to use this skill independently.
The `datakit` and `pipelines` packages are not required. The bundled runtime is
`scripts/reporting/`; entry points load that copy without searching for
the repository root. Python >= 3.10 is required.

```bash
python -m pip install "./skills/evaluation-reporting"
# For spatial extraction or typhoon maps:
python -m pip install "./skills/evaluation-reporting[all]"
```

Adjust installation paths to the actual copied location. The `spatial` and `typhoon`
extras can also be installed separately. This skill does not include an evaluation
workflow executor; inputs must already be evaluated products. The repository-level
skill router is `skills/SKILL.md`; downloading belongs to `skills/dataset-downloader/`,
not to this reporting workflow.

## Purpose

Convert **existing evaluation results into figures and a readable report**.
Depending on the reporting family, inputs are product directories or CSV files;
outputs include figures and `REPORT.md`. The report follows a paper's experimental
results structure rather than presenting an undifferentiated collection of data:
each section follows **introduction → table → figure → quantitative interpretation**.
For TS reports, each Chapter 4 section additionally ends with agent-written analysis
explaining the numerical results and their operational implications. Recommendations
must have supporting attribution evidence.

## Exclusions

- **Do not recompute evaluation metrics.** Consume existing evaluation products.
  Metric recomputation belongs to the evaluation workflow, not a separate implementation here.
- **Do not infer missing data.** Disclose missing columns, files, and lead times;
  never complete a report with defaults or estimates.
- **Do not invent conclusions.** Every reported number must be supported by input
  tables; omit unsupported judgments.
- **Do not determine capabilities from directory names.** Follow capability routing below.

---

## Trigger Scenarios

Use this skill for requests such as:

- “Generate figures or a report from these FGVP TS results.”
- “Create a report from `outputs/results/weather_ts_single_aifs`.”
- “Inspect these directories, combine compatible products, and compare the models.”
- “Use **FGVP as the primary model** and compare it with FuXi and AIFS.”
- “I replaced the TP parameterization; determine whether the change or module helped.”
- “Analyze this precipitation evaluation, including heavy precipitation and deficiencies.”
- “Compare the new model's TS with FuXi.”
- “Review how continuous-field RMSE increases with lead time.”
- “Is the ACC decay unusual? Is the activity ratio close to 1? Is the spectrum distorted?”
- “Compare **tropical spring RMSE**” or “Which model has higher Northern Hemisphere
  summer ACC?” Select a subset from existing long tables and add it to the report;
  **do not rerun evaluation**. Follow `references/rmse-batch-evaluation.md` §6.
- “Prepare a **precipitation case study**,” “Assess these 2025 rainstorm events,”
  or “Compare selected events using daily and event-total accumulation.”
  Use `render_rainstorm_report.py`; `--process` identifies the selected event IDs.
  First run `render_rainstorm_summary.py` and select from `summary_long.csv`;
  do not parse `screening.md` as a data source.
- “Add **spatial distribution maps**,” “Show the z500 bias pattern,” or “Show July
  and November mean fields.” Follow the spatial-distribution workflow and first
  confirm months, forecast archive root and deterministic/ensemble layout, and model name.

---

## Input Contract: Standard Evaluation Product Directories

The unified entry point, `render_report.py`, accepts **only standard evaluation
product directories**, not standalone CSV files.

```text
<product-directory>/
  manifest.json                 # Required; resolved_config.pipeline identifies capability
  scores.csv                    # Required for continuous fields; typhoon summary
  diagnostics/
    categorical_wide.csv        # Required for the primary TS model
    probability_wide.csv        # Optional for ensembles; omission removes probability data only
  typhoon/                      # Required for typhoon verification
    tc<id>_<initialization>.csv
    tc<id>_<initialization>_meta.json
```

| Capability | Required files |
|---|---|
| TS, deterministic or ensemble | `manifest.json` + `diagnostics/categorical_wide.csv` |
| Continuous fields | `manifest.json` + `scores.csv` |
| Typhoon | `manifest.json` + `typhoon/tc*.csv` + `*_meta.json` |

Required TS wide-table columns: `lead_h`, `grade`, and `TS`. Common optional columns:
`window_h`, `threshold_mm`, `hits`, `misses`, `false_alarms`, `n_pairs`, `POD`, `FAR`,
the miss-rate source column (`"\u6f0f\u62a5\u7387"` in JSON escape notation), and
`BIAS`. Preserve this source column name rather than renaming the input schema.

- Normalize inconsistent grade representations, such as `≥0.1` and `0.1`, before grouping.
- Align different grade sets on shared grades. Display missing entries as `—`, never zero.
- Stop if required columns are absent and explicitly identify the missing columns.

Each typhoon case's `_meta.json` must contain at least `tcid`, `init_utc`,
`forecast_type` (`det` or `ens`), and `tz_shift_h`. Ensemble metadata must additionally
declare `track_err_km` in `aggregation.B`.

> Flat directories containing only `ts_*.csv + *_meta.json` are **no longer accepted
> directly by `render_report.py`**. Produce `manifest.json` and
> `diagnostics/categorical_wide.csv` through the evaluation workflow, or provide
> a separate CSV entry point.

### Input and Output Locations

| Directory | Contents |
|---|---|
| `outputs/results/<directory>/` | Evaluation products: input manifest and CSV/diagnostics |
| `report/<name>/` | Deliverables: `REPORT.md` and figures, including `*.svg` |
| `outputs/.temp/` | Intermediate products, not deliverables |

---

## Classification: Identify the Product Family First

The naming convention is `weather_<capability>_<single|ens>_<model>`, with the model last:

```text
weather_ts_single_fgvp      # Primary model, with <capability>.json
weather_ts_single_fuxi      # Comparison model
weather_ts_single_aifs      # Comparison model
weather_ts_single_fengqing  # Comparison model
weather_ts_single_fgvp_tp   # Label identifying a changed variable
weather_ts_ens_fuxi         # Ensemble
weather_rmse_single_fgvp    # RMSE batch family
```

**Names are only supporting evidence**, not capability identifiers. Historical archives
may use other conventions, including `weather_ts_aifs_single`. Apply this priority order:

| Priority | Evidence | Application |
|---|---|---|
| 1 | `manifest.json`: `resolved_config.pipeline` | **Authoritative**: `weather_ts_det`, `weather_ts_ens`, `weather_field_scores`, `weather_typhoon_det`, `weather_typhoon_ens` |
| 2 | `protocol_id` + `manifest["metrics"]` | Fallback for historical products without `resolved_config` |
| 3 | Product structure | TS: `diagnostics/categorical_wide.csv`; continuous fields: `scores.csv`; typhoon: `typhoon/tc*.csv` + `_meta.json` |
| 4 | Directory name | Final corroboration only; never the deciding criterion |

### Classification Reference

| Observed products | Family | Route |
|---|---|---|
| `{id}_24h/{initialization-date}/` and `{id}_total/`, each segment containing `manifest.json` + `diagnostics/` | **Rainstorm cases**, two accumulation definitions | `render_rainstorm_report.py` → `rainstorm_case.md`. **Do not pass these to `render_report.py`**: an individual segment's `weather_ts_det` pipeline would run but incorrectly produce continuous-period sections. |
| Manifest pipeline `weather_ts_det` + `diagnostics/categorical_wide.csv` | **Deterministic TS** | `weather_ts_single.md` |
| Manifest pipeline `weather_ts_ens` + the same wide-table structure | **Ensemble TS** | `weather_ts_ens.md` |
| Ensemble TS plus `diagnostics/probability_wide.csv` or historical `aroc_bss*.csv`, in the primary or comparison product location | **Probability scores available** | Populate ensemble §4.7; see `precipitation-evaluation.md` §2.1. Check table presence and columns, not names alone or the `aroc_windows` configuration. Tables are the renderer's input. |
| Long-form `scores.csv` + `manifest.json` + `diagnostics/`, with `resolved_config.pipeline = weather_field_scores` | **RMSE batch family**, continuous fields | `_single` → `render_det_report.py`; `_ens` → `render_ens_report.py`. `_wave` remains the legacy spectral entry point `render_wave_report.py`; spherical bands are integrated into the first two reports. For one model, use `render_report.py`, which accepts the same products but only one directory. |
| `_meta.json` contains `babj`, `tcid`, or `forecast_type` | **Typhoon tracks** | Both `"det"` and `"ens"` use `weather_typhoon_single.md`; ensemble-mean reporting has no separate template. Use `render_report.py` with typhoon options. |
| Demonstration integration inventories, such as `mni_csv` | **Structural demonstrations or synthetic data** | Do not treat these as results unless explicitly requested. |

### Report the Inventory Before Execution

A supplied batch often mixes TS and RMSE families. Classify it and present an inventory:

```text
Nine directories under results/:
  Deterministic TS  weather_ts_single_fgvp / weather_ts_single_fuxi /
                    weather_ts_single_aifs / weather_ts_single_fengqing / weather_ts_single_fgvp_tp
  Ensemble TS       weather_ts_ens_fuxi (empty)
  RMSE batch        weather_rmse_single_fgvp / _fuxi / _ens_fuxi (all empty)
Which group should be reported?
```

**Identify empty directories explicitly.** Silently skipping them can falsely imply
that evaluation completed.

---

## Agent-Executed Workflow

**Execute the Python commands for the user** rather than expecting the user to run
them, except for server-side spatial extraction as specified below. Follow seven steps.

### Step 1: Classify

Identify product families and empty directories using the preceding rules.
**Present the inventory before proceeding.**

### Step 2: Select the Primary Model

- If the user names a primary model, such as FGVP, use it without substitution.
- Otherwise infer it in order, and **disclose the inference**:
  1. An unmistakable primary-model directory marker, such as the latest
     `weather_ts_single_fgvp` run.
  2. A directory containing `<capability>.json`, indicating explicit selection.
  3. If still ambiguous, ask the user.

The primary model determines every reference to “this model,” the direction of
difference-plot axes, and the recipient of recommendations. An incorrect selection
changes the interpretation of the entire report.

### Step 3: Verify Statistical Definitions Before Execution

> **For typhoon reports, ask about lead-time scope; do not assume it.** Before running,
> ask two questions in one concise request:
>
> 1. **Which lead times should the report include?** Recommend multiples of 24 h
>    (24, 48, 72, …), and provide actual available lead times with their `N_case`
>    and `N_storm`. If fixed-sample sensitivity is required, also confirm
>    `--fixed-leads`, which must be a subset of `--leads`.
> 2. **Who determines the discussion scope?** The user may specify `--primary-leads`
>    and, when needed, `--exploratory-leads`, or explicitly authorize agent selection,
>    for example, “You decide” or “Use your recommendation.”
>
> Select the scope yourself **only with explicit authorization**. Use only the user's
> stated operational window and per-lead sample counts: `N_case ≥ 10` and
> `N_storm ≥ 3` are primary guidance; explicitly requested leads with insufficient
> samples belong in exploratory discussion; remaining leads are case-study references.
> **Never select leads by error magnitude or visual appearance, or revise the window
> after plotting.** Document the actual parameters and selection rationale at delivery.
> The report's fixed statement that the user explicitly specified the discussion scope
> reflects the presence of `--primary-leads`, not the actual source of that decision.
>
> If the user neither specifies nor delegates scope, produce a provisional full-range
> sample at multiples of 24 h, disclose **“formal discussion window not yet selected,”**
> and do not designate any lead range as a formal conclusion.

| Check | Required handling when inconsistent |
|---|---|
| Same `window_h`? | **Do not combine in one figure**. Separate reports or explicitly disclose the discrepancy in the report. |
| Matching `lead_h` sets? | Apply the same restriction. |
| Matching `grade` sets? | Align on **shared grades**; show missing entries as `—`. `≥250` is commonly absent. |
| Consistent grade notation? | Normalize first, for example `≥0.1` versus `0.1`. |
| Same `tp_scale`, `var`, `interp`, and `tz_shift_h`? | Identify differences in source definitions and include a **comparability statement**. |
| Same coverage of initialization dates, events, and lead ranges? | **Do not subtract separately averaged model values with different denominators.** Use the common subset, or state both denominators in the table title. Otherwise Δ mixes coverage effects with model differences. |

> **Cross-model aggregate Δ values can have unequal denominators when coverage differs.**
> Cross-model means computed with pandas `.mean()` skip missing values by default;
> a model missing a segment silently receives a smaller denominator. An observed
> 43-event versus 33-event comparison increased Δ from +0.020 to +0.041.
> This can affect any family, although identical evaluation periods often conceal it.
> Check both denominators before communicating Δ.
>
> For **deterministic or ensemble continuous fields**, also ask whether spatial maps
> are wanted. If so, confirm initialization months, forecast archive root and
> `member_*/` versus single-file layout, and legend model name; see Spatial Distributions.

### Step 4: Execute

```bash
python skills/evaluation-reporting/scripts/render_report.py <primary-product-directory> \
  --result report/<report-name> --model <primary-model-label> \
  --compare "FuXi=outputs/results/weather_ts_single_fuxi/diagnostics/categorical_wide.csv" \
  --compare "AIFS=outputs/results/weather_ts_single_aifs/diagnostics/categorical_wide.csv"
```

- For TS, **comparison inputs are wide-table CSV files, not directories**.
  Repeat `--compare "NAME=CSV_PATH"` using `diagnostics/categorical_wide.csv`.
- The **primary input is a standard product directory** containing `manifest.json`
  and `diagnostics/categorical_wide.csv`. Flat `ts_*.csv + *_meta.json` directories
  without a manifest are no longer supported by the unified entry point.
- If the user describes a change, add `--change "..."`. It appears in the report
  header and initiates difference attribution; see `references/model-diff-analysis.md`.
- To change the model label used in the report title, pass, for example, `--model "FGVP-v2"`.

> **Input contract:** the unified entry point requires a standard product directory.
> Add the required manifest to flat products, or use a separate CSV entry point;
> no separate CSV entry point is currently provided.

**Rainstorm cases use a separate entry point.** It accepts the root of a year's
43 events, not a flat directory representing one evaluation under `outputs/results/`.

```bash
python skills/evaluation-reporting/scripts/render_rainstorm_report.py \
  --model FGVP=/workspace/szwCode/evaluation_results/rainstorm_ts_single_fgvp \
  --model AIFS=/workspace/szwCode/evaluation_results/rainstorm_ts_single_aifs \
  --model FuXi=/workspace/szwCode/evaluation_results/rainstorm_ts_single_fuxi \
  --model FengQing=/workspace/szwCode/evaluation_results/rainstorm_ts_single_fengqing \
  --model Tongyun=/workspace/szwCode/evaluation_results/rainstorm_ts_single_tongyun \
  --process 202503,202521 \
  --result /workspace/szwCode/evaluation_results/rainstorm_report
```

- `--model NAME=DIRECTORY` identifies a **product root** containing `{id}_24h/`
  and `{id}_total/`, not an individual event directory. The first model is primary.
  **There is only one Δ column**, always first model minus second model.
  Additional models receive side-by-side values only; win counts and recommendations
  also concern only the first two models. Produce separate reports for a pairwise matrix.
- **`--process` is required**, with comma-separated event IDs. First run
  `scripts/render_rainstorm_summary.py` to obtain both-definition summaries for all
  43 events, then select events for detailed analysis.

Select events from **`summary_long.csv`, not parsed Markdown in `screening.md`**.
The CSV is a long table suitable for pandas grouping; Markdown is for visual review.
Manual `split('|')` parsing can misalign columns. One such error shifted definition B
by one column and incorrectly yielded 33 wins and 0 losses instead of
22 wins, 3 losses, and 8 ties. Both files are written under `--result`.

Verify that comparison products exist for every selected event using
`ls <product-root>/{id}_24h` and `ls <product-root>/{id}_total`.
Otherwise the corresponding Chapter 3 block contains only missing Δ values.
Select events that rank prominently under **both definitions** and cover **both
directions of advantage**, including primary-model and comparison-model wins.

### Step 5: Add Chapter 4 Analysis Paragraphs for TS Reports

**This step applies only to TS reports.** Typhoon §4.1 and §4.2 already receive
renderer-generated **summaries**; §4.3 needs only 1–3 additional sentences interpreting
track shape when appropriate. For continuous fields and rainstorms, follow the
instructions at the beginning of their respective report skeletons.

The TS renderer writes only introduction → table → figure → `•` interpretation.
**Append an agent-written analysis paragraph to every `## 4.x` section and to the
opening Chapter 4 section itself**, after the last `•` line and before the next heading.
Follow the skeleton's instructions for end-of-section Chapter 4 analysis:

1. **Explain the metric:** what it measures, its favorable direction, and its
   relationship to neighboring metrics.
2. **Describe the pattern:** variation by threshold, variable, or lead, including
   concrete initial/final values, retention ratios, or decreases.
3. **Compare models:** identify the leader, margin, consistent cells, or reversals.
   For one model, compare thresholds or lead times instead.
4. **End with an operational implication or statistical-definition caveat.**

Mandatory constraints:

- Use **one paragraph of 3–6 sentences**, without bullets, directly attached to the
  section. Do not repeat the `•` interpretations: those report values, whereas
  analysis explains what they mean.
- Use only numbers in this report's tables and figures, or direct differences,
  ratios, and percentages calculated from them. **Do not introduce external numbers
  or independently recompute using a different definition.**
- Omit analysis when the entire section is unavailable for the batch or lacks its table.
- Check that every number is traceable to the corresponding table or figure caption,
  or is immediately derivable from those values.

### Step 6: Read the Report and Communicate the Results

Open the generated `REPORT.md`. Communicate **only 3–5 principal conclusions**,
each with specific numbers, and list the generated files. Do not restate the entire
report or introduce numbers absent from it.

**Before delivery, check numerical consistency across every table where A − B = Δ:**

```python
bad = frame[abs((frame["a"] - frame["b"]) - frame["delta"]) > 1e-6]
```

Nonempty results indicate column misalignment or mixed statistical definitions.
This validates both the report and any agent-written table-reading code, including
the Markdown-parsing failure described above. It **does not detect unequal
denominators**; verify coverage separately.

If the user described a module change, also answer **whether the changed module was
effective**, including costs: which thresholds improved and which deteriorated.
Use `references/model-diff-analysis.md`.

### Step 7: Disclose Failures

**Relay command errors verbatim** instead of bypassing them with an improvised manual
workflow. The table describes error meanings in English; actual runtime wording
may differ.

| Error meaning | Interpretation and required response |
|---|---|
| Report template for the capability is not implemented | Explain that figure/report generation is unavailable for this capability. |
| `categorical_wide.csv` does not exist | The wide-table writer was not enabled; request evaluation configuration changes and a rerun. |
| `manifest.json` does not exist | The supplied directory does not satisfy the standard product contract; existing flat result directories encounter this error. |
| Plotting requires `matplotlib` | Install the missing environment dependency; the historical environment command is `pip install -e .[viz]`. Use the appropriate installation location. |
| Neither `{id}_24h/` nor `{id}_total/` exists under the supplied directory | Supply the `OUTPUT_ROOT` of `rainstorm_ts_single_*`, not an individual event directory. |
| Event ID is not in the catalog | Correct `--process`; valid IDs are 202501–202543. See `rainstorm_catalog.py`. |
| At least two models are required, or model names are duplicated | Comparison reports require at least two inputs with unique names; duplicates misalign table columns. |

---

## Spatial Distributions: Optional Deterministic and Ensemble Continuous Fields

When producing a continuous-field report through `render_det_report.py`,
`render_ens_report.py`, or the `weather_field_scores` branch of `render_report.py`,
**ask about spatial maps after selecting the primary model**. If requested, confirm
each missing item:

1. **Initialization months**, as `YYYY-MM`; multiple months are supported.
2. **Forecast root and archive type**:
   deterministic `<ROOT>/<YYYYMMDD>/<NNN>.nc`;
   ensemble `<ROOT>/<YYYYMMDD>/member_<NNN>/<NNN>.nc`, for example 51 members
   `000`–`050`. `NNN` in filenames is the 6 h lead-step index:
   `004` = +24 h and `020` = +120 h. Member directories use the same zero-padded
   numbering format for member IDs.
3. **Legend model name**, such as `Tongyun`.
4. **Deterministic or ensemble report**, which determines extractor and three- versus
   four-row maps.

**Extraction runs on the server. Give these commands to the user to execute** because
the local environment cannot access server archives:

```bash
cd /workspace/szwCode/xmetai-evalation-v2
# Deterministic archive:
python skills/evaluation-reporting/scripts/extract_spatial_distribution.py \
  --root <forecast-root> --months 2025-07,2025-11 \
  --out /workspace/data/shenzw/spatial_monthly.csv
# Ensemble archive with member_NNN/ layout:
python skills/evaluation-reporting/scripts/extract_spatial_distribution_ens.py \
  --root <ensemble-forecast-root> --months 2025-07,2025-11 \
  --out /workspace/data/shenzw/spatial_monthly_ens.csv
```

The deterministic output is a wide CSV with 65,160 rows × 34 columns, approximately
9 MB. Ensemble extraction adds `{var}_spread_{YYYYMM}_h{lead}` for each combination.
Column patterns are:

```text
lat, lon, {var}_{YYYYMM}_h{lead}, [{var}_spread_{YYYYMM}_h{lead},] {var}_era5_{YYYYMM}_h{lead}
```

After the user downloads the CSV, render standalone maps; `spread` columns
automatically select four rows:

```bash
python skills/evaluation-reporting/scripts/render_spatial_distribution.py \
  --csv report/spatial_monthly.csv --out report/spatial --model Tongyun
```

Alternatively, pass **one model's CSV** to a report entry point to populate §4.6:

```bash
python skills/evaluation-reporting/scripts/render_det_report.py ... \
  --spatial-csv report/spatial_monthly.csv --spatial-model Tongyun
python skills/evaluation-reporting/scripts/render_ens_report.py ... \
  --spatial-csv report/spatial_monthly_ens.csv --spatial-model Tongyun
```

Without `--spatial-csv`, §4.6 retains its position and states that products are
unavailable for the batch; **subsequent figure numbers do not shift**.
An ensemble report rejects CSV files without `spread` columns, which are deterministic products.

**Statistical definitions must be included when communicating the maps:**

- **Monthly means use initialization month.** Average all initialization dates in the
  month separately for each lead. Never mix the four leads. Thus a July 2025 +120 h
  field includes valid times in early August by definition.
- **ERA5 uses the same valid times**, initialization date at 00Z plus lead, before
  averaging. The difference row is the **mean error field, or bias**, not a difference
  between two climatologies. Bias at +24 h is usually close to zero.
- **Ensemble spread:** first average initialization dates within each member to form
  monthly mean fields, then calculate the across-member **standard deviation with
  `ddof=0`**. Do not average instantaneous ensemble standard deviations.
- **Calculate wind speed before averaging:** compute `sqrt(u²+v²)` at each time and
  then average monthly.
- z500 output is in gpm. Sources in m²/s² are automatically divided by g; apply the
  same conversion to spread.

**Outputs:** one SVG per variable and month, with four columns for lead times.
Deterministic maps have model / ERA5 / model−ERA5 rows; ensemble maps have ensemble
mean / ERA5 / bias / spread, with a separate spread color scale.
**Produce SVG only, not PNG**, named `<lowercase-model>_<variable>_<YYYYMM>_vs_era5.svg`.

> A nearly white +24 h difference map does **not** establish a perfect model:
> monthly mean bias does not show random error. With approximately 30 samples per
> month, isolated grid-cell red/blue patterns are mostly noise. Interpret coherent
> structures larger than 20°.

---

## Capability Routing: Identify Capability Before Selecting a Template

Use `manifest.json` → `resolved_config.pipeline` as the authoritative criterion.
For historical products without `resolved_config`, use `protocol_id` together with
`manifest["metrics"]`. **These decisions are already implemented in code**:

```bash
python skills/evaluation-reporting/scripts/render_report.py <product-directory> --result reports/<name>
```

The entry point detects capabilities and dispatches to implemented renderers.
It explicitly rejects unknown capabilities or recognized capabilities without a
renderer. **Never substitute another family's template.** Standard typhoon products
are supported; use `render_report.py` typhoon options such as `--leads` and `--fixed-leads`.

| Pipeline | Coverage | Report template |
|---|---|---|
| `weather_ts_det` / `fdp_precip_ts` | Deterministic station precipitation categorical verification: TS, POD, FAR, frequency bias | Supported: [`assets/templates/weather_ts_single.md`](./assets/templates/weather_ts_single.md) |
| `weather_ts_ens` | Ensemble station precipitation categorical verification; TS of the ensemble-mean field, plus AROC/BSS probability scores | Supported: [`assets/templates/weather_ts_ens.md`](./assets/templates/weather_ts_ens.md) |
| `weather_field_scores` | Deterministic continuous fields: RMSE, ACC, activity, zonal spectra | Capability and renderer exist, but the formatting contract was removed from `assets/templates/`. |
| `weather_typhoon_det`, `forecast_type: det` | Deterministic typhoon track, AT/CT, intensity MAE/ME | Supported: [`weather_typhoon_single.md`](./assets/templates/weather_typhoon_single.md), with common statistics, plotting, and report generation |
| `weather_typhoon_ens`, `forecast_type: ens` | Scheme B ensemble-mean main results; exclude mean member errors | Same [`weather_typhoon_single.md`](./assets/templates/weather_typhoon_single.md), with member-count table and ensemble-definition wording |
| Other pipelines | CRPS, FSS, other probability-scoring capabilities | Not implemented; fail explicitly. This does not exclude the ensemble TS AROC/BSS section above. |

### Typhoon Branch

First read the [report workflow](./references/typhoon-report-workflow.md) and
[statistical definitions and field checks](./references/typhoon-verification.md).
Deterministic and ensemble-mean reports share one skeleton.

Follow **scope confirmation → CSV/metadata/tool validation → strict pairing by metric
and lead → statistics, figures, report, and machine-readable summaries → acceptance
and delivery**. Initialization cases receive equal weight by default.
Report MAE/ME where applicable and both `N_case` and `N_storm`.
Use storm-grouped bootstrap confidence intervals when required.
Do not rerun center diagnostics or modify source products. Validate metadata, times,
keys, and ensemble definitions at entry. Explicitly reject missing critical inputs;
missing observations and template placeholders are not results.
These rules apply only to typhoons and do not alter TS/RMSE statistical definitions.

```bash
python skills/evaluation-reporting/scripts/render_report.py \
  evaluation_results/weather_typhoon_ens_fgvp \
  --model FGVP \
  --compare "FuXi=evaluation_results/weather_typhoon_ens_fuxi" \
  --result reports/typhoon_ens_comparison \
  --leads 24 48 72 96 120 --fixed-leads 24 48 72 96 120
```

Single- and multi-model reports are supported. **Confirm lead-time scope first** using
Step 3; without explicit lead selection, the entry point uses all available data leads.
The default is 2,000 storm-grouped bootstrap iterations; `--bootstrap 0` disables them.
Outputs are `REPORT.md`, `summary.json`, `statistics.csv`, `differences.csv`, `track.svg`,
one `mae_<metric>.svg` and `me_<metric>.svg` per component/intensity metric, and one track
map per storm under `track_plots/`. The output directory must not exist or must be empty.

Typhoon `--compare` takes `NAME=PRODUCT_DIRECTORY`, unlike TS wide-table CSV comparisons.
Historical archives lacking `manifest.json` require a manifest first; bare case
directories are unsupported by the unified entry point.
For “use this model as primary” or “compare old and new versions,” pass
`--primary-model` and, when applicable, `--previous-model`; do not infer relationships
from model-list order alone. Specify discussion scope through `--primary-leads`
and `--exploratory-leads`. The report uses compact horizontal tables, paired primary-model
comparisons, and bias analysis; full statistics remain in CSV/JSON.
The six-chapter report is not a statistical inventory: summaries must distinguish
mean improvements from confidence-interval evidence.

### RMSE Batch Family

`weather_rmse_single_<model>` and `weather_rmse_ens_<model>` use dedicated multi-model
entry points **because of comparison scope, not a different product structure**.
They contain standard `scores.csv + manifest.json + diagnostics/` products and use
`weather_field_scores`. Consequently, `render_report.py` also recognizes them but
accepts only one product directory and produces a single-model report.

For multi-model comparison, use `scripts/render_det_report.py` for single-member
products or `scripts/render_ens_report.py` for ensembles, with at least two
`--model NAME=DIRECTORY` arguments. `render_wave_report.py` remains a **legacy spectral
analysis entry point**, consumes the same products, and has no separate skeleton.
The skeletons are [`weather_rmse_single.md`](./assets/templates/weather_rmse_single.md),
including §4.5 spherical-band power, and
[`weather_rmse_ens.md`](./assets/templates/weather_rmse_ens.md), including §4.8
spherical-band power. Both include optional §4.6 spatial distributions through `--spatial-csv`.

**Do not interchange deterministic and ensemble entry points.** Their product
directories have identical shapes; `render_det_report.py` can accept ensemble
products without an error but will produce single-member chapter definitions.
See `references/rmse-batch-evaluation.md` for archive structure, required matching
parameters, and derived metrics: relative RMSE, FA bias, and spectral log RMS,
using **ln for ensembles and log10 for single-member reports**.

### TS and Rainstorm Families

The three TS pipelines produce the same table structure and share `ts_report.py`.
Two skeletons exist only because ensembles add AROC/BSS, not because pipelines differ:
**templates are organized by reporting family, not pipeline**.

**Rainstorm cases are the sole exception: the same pipeline uses a different renderer.**
Each segment also has `weather_ts_det` and the TS table schema, but a full evaluation
contains 43 events × two definitions, totaling 255 independent product segments.
The ordinary `ts_report.py` assumes one continuous period and one wide table.
Rainstorm reports therefore use `rainstorm_report.py` + `rainstorm_plots.py`,
organized **by event rather than lead time**, through `scripts/render_rainstorm_report.py`
with at least two `--model NAME=DIRECTORY` arguments and mandatory `--process`.

The **two accumulation definitions are not mutually comparable**. Present them in
separate sections and compare models only within the same definition:

- **Definition A, daily 24 h:** `{id}_24h/{initialization-date}/`, one 08–08 window
  per day; event base rate varies with daily observations.
- **Definition B, event-total accumulation:** `{id}_total/`, one forecast accumulated
  over the complete event; base rates are substantially higher.

TS may differ by several times between definitions. In event 202503, ≥25 mm TS
was 0.236 for pooled definition A and 0.313 for event-total accumulation.
Comparing their magnitudes across definitions is not meaningful.

**Definition A event-level scores use pooled-count recomputation**: sum hits,
misses, and false alarms across initialization dates, then calculate TS from those
counts. Do not take the arithmetic mean of daily TS ratios, which would equally
weight a zero-TS day involving only a few stations and a day involving thousands
of events. Both renderer `daily_pooled` and summary-script `pooled_ts` use this definition;
update them together. Contingency-table counts exist only in
`diagnostics/categorical_wide.csv`, **not `scores.csv`**.

The bundled catalog's source of truth for event ID → dates/severity is
`scripts/reporting/reports/rainstorm_catalog.py`. The summary script and
renderer import it; do not duplicate the catalog elsewhere. The bundled catalog is
an independent copy of the main repository's module; evaluation configurations retain
the main-repository import, so catalog changes require explicit synchronization.

Rainstorm case reports **do not include precipitation spatial distributions**:
aggregated station contingency statistics contain no station-level or gridded rainfall.

---

## Output Contract

### A. Precipitation Categorical Verification: TS Family

```text
report/<model>_<batch>/
  overall_heatmap.svg       Figure 1     Overall metric heatmap; columns are metrics, normalized within each column/batch
  metrics_at_lead.svg       Figure 2     Reference-lead TS/POD/FAR/BIAS grouped bars by grade
  ts_vs_lead.svg            Figure 3     TS by grade and lead; small multiples with adaptive y-axis
  pod_vs_lead.svg           Figure 4     POD by grade and lead; y-axis 0–1
  far_vs_lead.svg           Figure 5     FAR by grade and lead; y-axis 0–1
  bias_vs_lead.svg          Figure 6     BIAS by grade and lead; dashed reference at 1
  ts_boxplot.svg            Figure 7     Cross-lead TS distribution by model
  performance_diagram.svg   Figure 8     POD versus 1−FAR; models share one plot
  aroc_vs_lead.svg          Figure 9     AROC; ensembles with probability tables only
  bss_vs_lead.svg           Figure 10    BSS; ensembles with probability tables only
  lead_dot.svg              Figure 9/11  Composite TS by lead; Figure 11 for ensembles
  region_ts_vs_lead.svg     Figure L1    TS by latitude band, when region rows exist
  REPORT.md
```

All figures are **SVG**; `fig.savefig(path)` determines format from the suffix.
Filenames have no window/lead prefixes: the report and captions state the window
and reference lead. Multi-model curves show **absolute values with adaptive y-axes**
(`zoom_y=True` for `ts_vs_lead`); forcing 0–1 obscures differences between similar
models. Tables in §4.1 and §5 retain absolute numerical values.
**All figure text is English**, including axes, titles, legends, and grade labels.
Use a single-line legend at the bottom, `(a)`–`(f)` subplot labels in upper corners,
and ASCII model names such as `FengQing`; non-ASCII names may render as missing glyphs.
Separate difference plots were removed because the curves already show leaders and margins.

`REPORT.md` contains **six chapters and an appendix**, using `# 1.` through `# 6.`.
Sections follow introduction → table → figure → interpretation.
Each Chapter 4 section also needs agent-written analysis after the interpretations
(Step 5). **Tables are unnumbered** and introduced by bold titles.
**Figures have sequential, in-place captions**, rendered here as `Figure N:`.
Quantitative interpretations use one `• ` line each.

```text
# 1. Summary                    Object/definitions → one quantitative diagnostic paragraph → operational assessment → BIAS explanation
# 2. Samples and Data Quality   Coverage table + coverage/lead/grade consistency bullets + note that the wide table lacks dates
# 3. Metrics and Interpretation Metric-definition table + two statistical-definition notes
# 4. Overall Results            Overall metrics + Figure 1 + overall interpretations + Figure 2 reference-lead bars
  4.1 TS                        All-lead mean TS + primary contingency counts + lead-decay table + Figure 3 + interpretations
  4.2 POD                       POD table + Figure 4 + interpretations
  4.3 FAR                       FAR table + Figure 5 + interpretations
  4.4 BIAS                      BIAS table + Figure 6 + interpretations
  4.5 Cross-Lead Stability       Boxplot statistics + statistical-definition note + Figure 7 + interpretations
  4.6 Performance Diagram        Reference-lead POD/FAR table + Figure 8 + interpretations
  4.7 Probability Scores        Ensembles only: table + Figure 9 AROC + Figure 10 BSS + interpretations; reserve section and figure numbers if absent
# 5. Results by Lead            Composite TS by lead + Figure 9, or Figure 11 for ensembles + interpretations
# 6. Conclusions and Recommendations  Numbered recommendations + definitions and caveats
# Appendix: Additional Diagnostics    Optional Appendix L, separately numbered Figure L1; entire block may be absent
```

The ensemble skeleton, `weather_ts_ens`, adds three metric-definition rows in §3
(AROC, BS, BSS) and §4.7 probability scores; the other chapters are identical.
**Reserve Figures 9 and 10 for AROC and BSS**, making the ensemble §5 dot plot
Figure 11 and the deterministic dot plot Figure 9.
Search the primary model and each comparison input's product location for
`diagnostics/probability_wide.csv`, then historical `aroc_bss*.csv`.
Show probability results only when those tables exist, but **always retain the
section slot and figure numbers**. If missing, state “not produced for this batch”;
do not renumber later figures.

TS reports have **no figure/table index** because figures and captions appear in place.
Continuous-field reports instead collect figures in a final Figures section and do
not duplicate them in the body.

> **No automated test currently enforces the TS skeleton.** There is no
> `test_ts_report.py` or literal template-scanning test in `tests/`.
> [`weather_ts_single.md`](./assets/templates/weather_ts_single.md) and
> [`weather_ts_ens.md`](./assets/templates/weather_ts_ens.md) are complete reports
> with `{…}` numerical placeholders. All other text, introductory wording, headers,
> separators, emphasis, and definition notes must match final output literally.
> When changing renderer structure, figure/table order, or fixed wording, update
> **both skeletons in the same change**, rerender real products, and compare each
> section manually. CI does not enforce this correspondence.

### B. Deterministic Continuous Fields

For `weather_field_scores`, under `--out`:

```text
rmse_vs_lead.png                     One line per variable; facets by unit, such as z500 m^2/s^2 and t2m K
acc_vs_lead.png                      ACC decay with lead
activity_ratio_vs_lead.png           Reference line at y=1
spectrum_power_ratio_vs_lead.png     Reference line at y=1
rmse_heatmap.png                     Variable × lead, showing multiples of the first-lead RMSE, independent of units
spectrum_curve_<variable>_<lead>h.png Optional; log y-axis, forecast and observation curves
REPORT.md
```

Historically, `tests/unit/visualization/test_field.py` ran the renderer and compared
the report with `assets/templates/field_scores.md`. **That skeleton was deleted**,
so the formatting source of truth is absent; failures cannot simply be attributed
to renderer changes. The final **Figures** section exists only when figures were
produced. Without figures, omit its heading and number and make Definitions and
Caveats Chapter 6, without a numbering gap.

### C. Rainstorm Cases

For `render_rainstorm_report.py`, under `--result`:

```text
figures/process_overview_daily.png  Definition A pooled TS for all 43 events, grouped bars in descending primary-model order
figures/process_overview.png        Event-total TS for all 43 events, grouped bars in descending primary-model order
figures/<id>_daily_ts.png            Daily 24 h TS for one event; facets by grade, one line per model
figures/<id>_threshold_ts.png        Event-total TS for one event; grouped bars by grade
figures/caliber_scatter.png         Event scatter: x = definition A pooled TS, y = definition B event-total TS
REPORT.md
```

The report is organized **by event**, not lead:

```text
1. Executive Summary          Object → change → fixed explanation of both definitions and two calculated base rates → quantitative diagnostic paragraph → operational assessment
2. Event Inventory and Selection
                              Tables 1/2: all 43 events under both definitions
                              + Figures 1/2: all-event definition A/B bars → summary
3. Event-by-Event Analysis     One block per selected event, with four subsections:
  3.x.1 Event Overview         Initialization-day counts, window lengths, valid pairs under each definition
  3.x.2 Definition A: Daily 24 h
                              Raw daily-value table + figure → summary using pooled values
  3.x.3 Definition B: Event Total
                              Table + figure → summary
  3.x.4 Event Summary          Whether model ranks agree across definitions
4. Cross-Event Comparison      Equal-weight means for nine metrics, using only events with valid values for every model in each row + scatter → summary
5. Recommendations            Each linked to a specific number or event ID
6. Definitions and Caveats    Ten items, including same-event cross-model aggregation and absence of precipitation spatial maps
```

The number of Chapter 3 blocks equals the number of IDs in `--process`.
**Table and figure numbering continues across blocks without resetting.**
[`rainstorm_case.md`](./assets/templates/rainstorm_case.md) illustrates one block with
`3.{x}` and `Table {N}` placeholders; do not hardcode `3.1` or `Table 3`.

The tests for `assets/templates/` do not cover the rainstorm skeleton.
Only the real-data integration test,
`tests/integration/test_rainstorm_ts_realdata.py`, exists; **there is no literal
skeleton comparison test**. Keep the skeleton manually aligned with static wording
in `rainstorm_report.py`, updating both when changing report prose.

### D. Spatial Distributions

For optional `render_spatial_distribution.py` maps, under `--out`:

```text
<lowercase-model>_z500_<YYYYMM>_vs_era5.svg
<lowercase-model>_ws850_<YYYYMM>_vs_era5.svg
```

Produce one **SVG only** per variable/month. Deterministic rows are model / ERA5 /
model−ERA5; ensembles add spread, defined as the standard deviation of member
monthly mean fields, with a separate `YlOrRd` scale. Columns are +24, +48, +72,
and +120 h. The first two rows share an absolute-value scale; differences use
zero-centered `RdBu_r` with half-width set to the 99.5th percentile of absolute differences.

---

## Mandatory Rules

1. **Numbers must come from input tables.** Explicitly label derived quantities,
   for example miss rate = `1−POD`, as derived.
2. **Equal-weight averaging is not sample weighting.** Retain the statement that
   the report's all-lead mean weights lead times equally.
3. **`BIAS` is frequency bias**, forecast event count divided by observed event count.
   It is not mean error.
4. **Missing values are not zero.** `TS = NaN` means no valid pairs or no events,
   unlike a score of 0.000; distinguish these cases.
5. **Grades with too few `n_pairs` or no events do not support relative-strength
   conclusions.** For example, zero hits support “no skill,” not “the worst skill.”
6. **Cross-model comparisons require common definitions.** Disclose sources, periods,
   and lead ranges; explicitly mark inconsistent definitions as “not directly comparable.”
7. **Disclose missing Chinese-language fonts.** If the plotting script reports missing
   fonts for such text, relay the warning and recommend Noto Sans CJK.
8. **State failures explicitly.** Missing columns, absent result tables, baseline-only
   products without a new model, and unimplemented templates must not produce an
   empty report silently or be handled with another family's template.
9. **Do not impose absolute continuous-field quality classes.** RMSE, ACC, activity,
   and spectra have no universally accepted pass thresholds. Do not claim, for
   example, “ACC ≥ 0.6 passes,” or apply red/amber/green ratings. Compare within
   the data and discuss decay, deviation from 1, and monotonicity.
10. **`n_valid` has metric-dependent meaning.** For `rmse`, `acc`, and `activity_*`,
    it counts grid points: global 0.25° fields contain 1,038,240. For
    `spectrum_power_ratio`, it counts averaged two-dimensional fields: one for a
    single initialization and lead. It is not a sample-size measure; do not use
    it to assess credibility or enter it in a sample-size column.
11. **Units follow the source.** In `*_phys` layouts, z500 RMSE is in `m^2/s^2`,
    **without division by g**; 1 m²/s² ≈ 0.102 m. Specific humidity q uses `g/kg`;
    dimensionless quantities use `1`. Do not directly compare raw values across
    variables. Facet RMSE plots by unit and use multiples of first-lead RMSE in heatmaps.
12. **TS reports must resemble paper experimental-results chapters, not data inventories.**
    Every figure needs a sequential in-place caption, and every table needs an
    unnumbered bold title. End each analytical subsection with quantitative `• `
    interpretations, one per line. Each statement must contain specific numbers or
    be falsifiable from the data. If evidence is insufficient, say so instead of
    using unsupported descriptions such as “good performance.”
    `ts_report.py`'s `analysis_*` functions and `analysis_probability` generate these
    interpretations from **already computed statistics**. They must not reread tables
    or reimplement metrics, which would desynchronize tables, figures, and prose.
    No automated tests enforce these requirements; update skeletons with renderer changes.
13. **Attribution requires a control.** Without a change description, report differences
    but do not attribute them. Before claiming that change X improved results, state
    which metric should move in which direction if X is effective. See
    `references/model-diff-analysis.md` for thresholds and cost checks.
14. **Treat differences below 0.005 as ties**, without naming a leader.
    Use **competition ranking**, assigning tied results the same rank.
15. **Recommendations must follow attribution evidence.** Every recommendation must
    refer to specific preceding numerical results. Omit unsupported generic suggestions,
    such as increasing model size or replacing attention mechanisms.

---

## Common Situations

| Situation | Required handling |
|---|---|
| `<name>.csv` does not match the directory name | **Glob `*.csv`** rather than constructing filenames from directory names. |
| A batch mixes TS and RMSE | Classify, report the inventory, and ask which group to report; do not combine all families. |
| A directory is empty | **Identify it explicitly**, not silently skip it. |
| Two directories appear to contain the same model or duplicate runs | Compare `n_pairs` cell by cell. Identical counts indicate duplicate runs of the same evaluation; do not include both in comparison plots, which would produce coincident boxes. |
| Grade notation mixes `≥0.1` and `0.1` | Normalize before alignment; otherwise they become separate grade sets. |
| Grade coverage ends at `≥250` for some models and `≥100` for others | Align shared grades, use `—` for missing values, **never zero**. |
| No primary model was selected | Follow name marker → `<capability>.json` → ask the user; disclose any inference. |
| A change description is supplied | Follow `references/model-diff-analysis.md`: expected signature → observed signature → costs. |
| No 24 h lead exists | Use the smallest available lead, not a hardcoded 24 h, and disclose it. |
| Only 6 h accumulation windows exist | Typical thresholds are `0.1/13/25`, unlike 24 h thresholds; do not mix definitions. |
| Only one model exists and no baseline is available | Produce a single-model report and omit comparison chapters; retain boxplots, one box per grade. |
| Only long-form `scores.csv` exists, without the wide table | **Do not synthesize it.** Add `writers=["csv_long", "categorical_wide"]` to evaluation configuration and rerun. Mixing definitions is worse than failing. |
| Ensemble forecasts | There are two products: deterministic-definition TS of the **ensemble-mean field**, and probability scores AROC/BSS. Use `weather_ts_ens.md`; **do not combine both score types in one figure**. See `precipitation-evaluation.md` §3.1. |
| Ensemble evaluation omitted `--ref` | `BS_ref` is entirely missing and BSS cannot be calculated. State “no climatological reference”; do not substitute another window. |
| A continuous-field variable lacks a metric | Check per-variable routing in manifest `metric_options`, where each metric has its own `variables` list. Truly absent results have no row, not an empty `value`. |
| Continuous-field `level`, `region`, `threshold`, and `window_h` are all empty in `scores.csv` | This is the default `grid_valid_time` definition using `variable`, `sample_unit`, and `unit`, **not missing data**. |
| Assessing normal total spectral power but distorted distribution | `spectrum_power_ratio` sums over all wavenumbers and cannot reveal this. Use `--spectrum-variable` and `--spectrum-lead` to inspect high-wavenumber spectral curves. |
| A requested slice such as tropical spring RMSE | Follow `references/rmse-batch-evaluation.md` §6: filter long tables and recompute summaries using existing definitions, **without rerunning evaluation**. Seasons use initialization time, with DJF grouped as meteorological winter. Add the requested prose manually; the renderer does not interpret such requests. |
| No explicit season/latitude-band column appears available | Derive season from `init_time` (§6.2); latitude bands already use `region`, with an empty value meaning global. Neither requires evaluation configuration changes. |
| Rainstorm analysis is requested without event selection | Run `render_rainstorm_summary.py` for 43 events × two definitions × N models and ranking guidance. Select from **`summary_long.csv`**. **`--process` is mandatory**; reporting all 43 creates an unnecessarily extensive document. |
| Product roots have different event coverage | Cross-model Δ otherwise has unequal denominators, with an observed twofold effect. The rainstorm renderer now uses only events valid for every model and states event counts in table titles. Disclose that restriction rather than presenting Δ without coverage context. |
| The same event has TS values of 0.21 and 0.61 | They may represent daily pooled versus event-total accumulation. Different base rates make them **not mutually comparable**. Confirm the desired definition or report both in separate sections. |
| A user requests pooled hits rather than a mean of daily ratios | Use pooled counts: TS is a ratio and averaging ratios is distorted by small-sample days. Keep renderer `daily_pooled` and summary `pooled_ts` definitions identical. Read `diagnostics/categorical_wide.csv`, not `scores.csv`, for counts. |
| A rainstorm report lacks precipitation-area maps | This is intentional: aggregate hits/misses/false alarms/n_pairs/TS contain no station or gridded rainfall. Maps require original forecast NetCDF and station messages. |
| Events 202519 and 202520 have identical values | The catalog lists two aspects of the same period separately. **Evaluate separately and do not merge.** Identical values are expected; the catalog module docstring records this. |
| Continuous-field spatial maps are requested | Follow Spatial Distributions: confirm initialization months, archive root and deterministic/ensemble type, and model name. Run the appropriate extractor on the server, download the CSV, then render SVG locally or pass `--spatial-csv` to the report entry point. |

---

## References

### Report Formatting Contracts

- Precipitation categorical verification:
  `assets/templates/weather_ts_single.md` for the two deterministic pipelines;
  `assets/templates/weather_ts_ens.md` for ensembles with probability scores.
- RMSE batches: `assets/templates/weather_rmse_single.md`, including §4.5 spherical
  bands; `assets/templates/weather_rmse_ens.md`, including §4.8 spherical bands.
- Typhoon track/intensity: `assets/templates/weather_typhoon_single.md`, a six-chapter
  structure **shared by deterministic and ensemble-mean reports**. Ensemble main
  results use scheme B with a member-count table and definition statement, not
  ensemble-specific metrics or a separate template.
- Rainstorm cases: `assets/templates/rainstorm_case.md`, two definitions × multiple
  models, organized by event.

These are the six existing templates. The deterministic continuous-field
`weather_field_scores` skeleton and the templates directory's `README.md` were deleted.
The capability, renderer, and `references/field-evaluation.md` remain, but the
formatting contract lacks a source of truth.

### Figure Styling

- Colors, figure widths, and font sizes: `references/visual-style.md`.
  Code source of truth: `scripts/reporting/core/style.py`.

### Numerical Interpretation

- Metric definitions and interpretation: `references/evaluation-metrics.md`.
- Precipitation definitions: `references/precipitation-evaluation.md`.
- Precipitation diagnostics, sample thresholds, skill classes, bias/error patterns,
  and recommendation triggers: `references/diagnostic-rules.md`.
- Continuous-field metric semantics: `references/field-evaluation.md`.
- Typhoon workflow and acceptance:
  [references/typhoon-report-workflow.md](./references/typhoon-report-workflow.md).
- Typhoon fields, ensemble definitions, strict pairing, and uncertainty:
  [references/typhoon-verification.md](./references/typhoon-verification.md).
- RMSE batch archive structure, derived metrics, and interpretation:
  `references/rmse-batch-evaluation.md`; `_single` and `_ens` are current families,
  `_wave` is the legacy spectral entry point. **§6 covers requested subset comparisons**,
  such as tropical spring RMSE.
- **Model-difference attribution**, effectiveness, costs, and diagnostic signatures:
  `references/model-diff-analysis.md`.

---

## Maintenance

- **Rendering logic is bundled** in `scripts/reporting/`:
  `precipitation_plots.py` + `ts_report.py` for TS;
  `field_plots.py` + `field_report.py` for continuous fields;
  `det_report.py` for deterministic RMSE batches;
  `rainstorm_plots.py` + `rainstorm_report.py` for rainstorm cases;
  `report.py` for capability detection and dispatch.
  Typhoon statistics/rendering use `typhoon_report.py` + `typhoon_plots.py`.
  `scripts/render_report.py` is only an entry point.
  Change the bundled runtime when changing capabilities and run standalone tests.
  References document rule-to-code mappings; **code takes precedence if they conflict**.
- **The bundled rainstorm catalog source of truth** is
  `scripts/reporting/reports/rainstorm_catalog.py`, containing 43
  ID → dates/severity entries. The summary script and renderer share it.
  It is an independent copy of the main-repository implementation; evaluation
  configurations still use that original module. Explicitly synchronize both copies.
- **Keep `scripts/render_report.py` minimal**:
  add `scripts/` to `sys.path` → call `build_report` → print outputs.
  **Do not add classification or workflow assembly there**, which would diverge
  from runtime routing.
- **`assets/templates/` contains formatting contracts**, each a complete report with
  `{…}` numerical placeholders, not an explanatory document or template engine.
  They do not participate in rendering and have no automated literal-match tests.
  Update skeletons with renderer structure/order/fixed-wording changes, then rerun
  real products and compare section by section.
- **Chapter 4 analysis placeholders represent a mandatory manual step outside
  rendering**. The renderer emits only `•` interpretations; write the additional
  analysis using Step 5. Preserve an analysis placeholder in every `## 4.x` section,
  including any newly added subsection.
- **Known limitations and implementation details:**
  1. The unified entry point no longer accepts flat `ts_*.csv + *_meta.json`
     directories. Supply `manifest.json` + `diagnostics/categorical_wide.csv`.
  2. **Ensemble date definitions have two modes with different numerical results.**
     `render_ens_report.py` requires at least two common dates by default and names
     failures explicitly. With `--own-dates`, **each model uses all its own
     initialization dates**, while §6 paired tests still use common dates.
     For substantially different coverage, this avoids truncating all models to the
     shortest archive. One five-model example had AIFS 27, FGVP 228, FGVP_old 348,
     FuXi 349, and S2B 29 days, but only 26 common dates, below the 30-day sample
     threshold; the report labels this “directional reference.”
     **The cost of `--own-dates` is unequal periods within one table, not a fair
     comparison.** The renderer discloses this in §1, §2, §4, §7, and the opening
     appendix paragraph, and adds a “dates used in this report” column to §2.
     The opening table of
     [`weather_rmse_ens.md`](./assets/templates/weather_rmse_ens.md) documents exact
     wording differences between the two modes.
  3. **AROC/BSS rendering already exists.** `ts_report.py` reads
     `diagnostics/probability_wide.csv` for ensemble §4.7, including its table,
     Figures 9/10, and interpretations. Report-side `report.py` searches primary and
     comparison locations for `probability_wide.csv`, then historical `aroc_bss*.csv`.
     Older `weather_ts_ens_aifs` archives with two adjacent bare CSV files use the
     latter convention. Found probability tables join the comparison; models without
     them contribute TS only. Three tested v2 ensemble archives had the new table,
     whereas `ts_multi_*` had neither table.
     Add the first section's period disclosure manually; inferred day counts are
     hints only, as illustrated by `_probe_ts`.
     **No product field records the configured ensemble member count**; a number
     computed in a long table is not the run parameter. Supply `--members N`;
     without it, the report displays `—`.
  4. **Spherical-band power rendering already exists**, in deterministic §4.5 and
     ensemble §4.8, using helpers in `wave_report`. Three expanded
     `spherical_bands` rows come from the long table, distinguished by **`group`**.
     Every participating product must contain these rows: `probe_bands` uses
     `all(...)`. One missing input sends the entire block through `SPHERICAL_ABSENT`,
     displaying “not produced for this batch.”
     Mixed archives are particularly susceptible: in one five-model batch only
     v2-produced FGVP/SPW had spherical-band rows; three converted intermediate
     archives did not, leaving all three related sections empty. Complete those
     evaluations or accept absence. See `references/rmse-batch-evaluation.md` §5,
     item 6.
     The legacy `_wave` entry point also supports the two `--own-dates` modes:
     spectra use each model's full dates, while paired tests always use common dates.
  5. **Typhoon reports support current case CSVs and historical archives explicitly
     declaring scheme B.** Deterministic reports have six chapters and per-storm
     track maps, requiring basemap; the `viz` dependency group declares it.
     Member-detail spread, probability skill, and additional scheme A reports are
     not included. Verify historical algorithms and units before reporting.
  6. **Rainstorm skeletons have no literal comparison tests.** Only real-data
     integration tests exist. Manually align `rainstorm_case.md` with static prose
     in `rainstorm_report.py`; actual rendering and skeleton comparison are currently
     the only acceptance method. A local validation passed the same products under
     two model names, yielding all `+0.000` differences while verifying rendering
     paths and continuous numbering.
     `render_rainstorm_summary.py` also lacks
     `stream.reconfigure(encoding="utf-8")`; Windows GBK consoles may display
     language-specific text incorrectly. **UTF-8 output files are unaffected.**
- `agents/` is currently empty.
- **`skills/evaluation-reporting/` is the source**; the registered copy is
  `.claude/skills/evaluation-reporting/`. Windows symbolic links require administrator
  privileges, so deployment uses copying. Refresh that copy after source changes.
- **Maintainer:** Shen Zhewen (szw).

### 2026-10-26: Standalone Copy

- Copied the reporting runtime and minimal internal dependencies into this skill's
  `scripts/reporting/` without moving the original implementation.
- All entry points load the skill-local runtime and remain executable when the
  complete directory is copied to another project or registration location.
- Standalone dependencies are in `pyproject.toml`. `tests/test_reporting.py` covers
  isolated imports, all script entry points, and real report generation.
  Run these tests after updating the copy.

### 2026-10-23: Maintenance Record Before the Split

- **Reporting-stack consolidation:** the former `skills/visualization` implementation
  was merged into the former `pipelines/evaluation/reporting` location.
  Historical `xmetai_evaluation.*` dependencies became `datakit.*` and
  `pipelines.evaluation.reporting.*`. These are historical locations, not the current
  standalone package, which is `reporting`.
- **Single entry point:** agents use `scripts/render_report.py`, calling
  `build_report()`. Typhoon options are forwarded unchanged:
  `--leads`, `--fixed-leads`, `--bootstrap`, `--seed`, `--small-cases`,
  `--small-storms`, `--forecast-type`, and `--verify-legacy-members`.
  `generate_typhoon_report.py`, package-level `cli.py`, and `__main__.py` were removed.
  The former `python -m pipelines.evaluation.reporting` command is no longer supported.
- **Input contract:** standard product directories only; bare case directories
  without `manifest.json` are unsupported.
- **Validation:** the Tongyun deterministic typhoon report had six chapters and
  36 figures, including 27 `track_plots` figures, complete links, and no placeholders.
  A literal comparison against the former dedicated entry point differed only in
  output paths and commands.
- **Outstanding items recorded at that time:** the `all` extra had not incorporated
  `cartopy`/`basemap`; report-contract tests had not been added.
