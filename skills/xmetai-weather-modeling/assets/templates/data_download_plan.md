# Data Download Plan

> **Bottom line**: <what to download, from which source, approximate size, where to place it. One or two sentences.>

## Download List

### Main table: data requirements

One row per dataset and purpose, deduplicated across configs. For training, split rows by the config's train and validation/test time ranges; for inference, use the history window before the inference time. Add one row for static fields and statistics sidecars only when the model contract requires them (for example non-empty `maskid`/`static_fields` or required `const`); omit the static-field row when the model does not need it. Statistics sidecars (`mean`/`std`/`weight`) are always required for Zarr grid models.

| Purpose | Dataset | Variables / channels | Frequency | Time coverage | Grid | Est. size |
|---|---|---|---|---|---|---|
| Train | | | | | | |
| Validation / test | | | | | | |
| Inference (optional) | | | | | | |
| Static fields / sidecars | <separate download or computed> | mask / land_mask / const / mean / std / weight | n/a | one-time | model grid | small |

In each **Time coverage** cell, distinguish `requested model range`, `prepared model-frame range`, and `raw source range` when they differ. Label extra frames caused by the executing dataset indexer as runtime implementation buffers. For temporally aggregated variables, state the window and whether the output timestamp labels its start or end; split variable groups into separate rows when their raw ranges differ.

Time split notes (training): record the train and validation/test ranges from `train_times`/`test_times` (or training-script defaults) and note any overlap. For inference, record the inference time plus the required history window (`hist_frames`), maximum target lead, and any endpoint frame required by the actual sequence-index loop; `mean`/`std` sidecars are still required.

Static field acquisition notes: ERA5 invariants come from `reanalysis-era5-single-levels` with a single time point; ERA5-Land orography/land-sea mask come from the ERA5-Land documentation attachments; sidecars are computed from the prepared dataset. The static field grid must match the training grid.

### Download & conversion (one row per dataset)

Name the source dataset explicitly (CDS catalogue name or author-provided data) so the plan stays executable without re-deriving the download or conversion details.

| Source dataset | Download format | Conversion step | Target format |
|---|---|---|---|
| | | | |

Temporal aggregation note: for every resampled, accumulated, averaged, differenced, minimized, or maximized variable, record the source temporal representation, timestamp meaning, cadence, target operator/window, output label convention, incomplete-window policy, and raw boundary required to create the prepared range. Do not infer this rule from a special variable name.

Precipitation unit note: additionally record the source's original unit and conversion factor to **mm accumulated values**. ERA5/ERA5-Land deliver `tp` in metres; rate forms (`kg m-2 s-1`, `mm/h`, `mm/s`) require conversion using their interval. Keep unit conversion distinct from the generic time-window calculation.

## Data Source

- Provider / dataset name / acquisition method:
- License and authentication (never include credentials):
- Variable-name mapping (only when non-obvious):
- Contact / URL:

## Pending Confirmation

Only items that block the download or the conversion.

- [ ] ...
- [ ] ...

## Next Step

- Recommended action:
- Blocking confirmations:
