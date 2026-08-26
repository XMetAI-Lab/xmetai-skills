# Data Download Planning

Use this reference only for extracting meteorological data requirements and producing a plan before any download or data write begins. It governs the planning stage; it does not prohibit a later, explicitly confirmed download.

## Scope

This route answers: **What data does the selected config require, where may it come from, and what must be confirmed before downloading it?**

During this planning stage, do not download, create, convert, aggregate, regrid, normalize, append, overwrite, or otherwise modify data. Do not generate statistics or claim that a dataset is ready for training.

After the user explicitly confirms the plan and asks to proceed, this planning stage is complete. Return to the main skill routing: the model may use its native capabilities to write a suitable downloader, read local configuration, use locally configured credentials without exposing or copying them, and execute only the confirmed download. No downloader needs to be bundled with this skill. Do not treat the planning-stage stop rule as a permanent ban on execution.

Use **Data analysis** instead when data already exists and the task is to inspect schema, integrity, time gaps, values, statistics, static fields, or training readiness.

## Workflow

1. Inspect the target repository and list candidate entry configs (for example, `configs/*.py`).
2. If more than one candidate exists, present them with their key differences (dataset path, resolution, frequency, time range, variable count) and ask the user which model(s) the plan should cover. Multiple selections are allowed; proceed in multi-config mode when the user selects more than one.
3. For each selected entry config, trace its reused base configs and dataset definitions.
4. Extract the dataset contract: variables, levels, time range, frequency, spatial extent, resolution, format, static fields, and statistics sidecars.
5. Classify each requirement as confirmed, optional, or pending confirmation.
6. Identify the documented or proposed data source, license, authentication method, and variable-name mapping.
7. Record the destination, estimated size when known, and existing-file policy.
8. Produce a Markdown, YAML, or JSON pre-download plan.
9. Stop before any network request or filesystem write and ask the user to confirm the download list.
10. If the user later confirms the list and asks to proceed, exit this planning route. Execute the confirmed download with the model's native capabilities; do not continue applying this reference's no-write restriction to that execution turn.

Confirmation covers only the planned source, variables, time range, destination, estimated size, authentication mechanism, and existing-file policy. If execution would materially change any of them, update the plan and obtain confirmation again. Download completion does not authorize conversion, normalization, statistics generation, Zarr mutation, or training-readiness claims.

## Requirement Extraction

Classify the entry config as training or inference first:

- **Training configs** (for example `*_base.py`, `*_vit.py`, `*_lora.py`): extract the full training and validation/test coverage from `train_times`/`test_times` (or the training-script defaults) and record the split explicitly in the plan. Note overlaps when the config places the same period in both train and test.
- **Inference configs** (for example `*_infer.py`): they reuse the base dataset contract (channels, frequency, grid) but only need the history window before each inference time. Record the inference window instead of the full training coverage. The exported ONNX interface takes physical values in the current models, but the core inference wrapper may receive normalized inputs — confirm the export/inference code before preparing inference data. `mean`/`std` sidecars remain required, `weight` is not needed, and targets are not downloaded.

Check:

- Dataset class and configured data paths.
- Input and output variable names and channel counts.
- Historical and forecast frames.
- Training and evaluation time coverage required from the data.
- Time frequency and sample interval.
- Spatial extent, grid, and resolution when specified.
- Expected Zarr or NetCDF variables and coordinates.
- Static fields, masks, and `mean`, `std`, or `weight` sidecars expected by the config.

Do not infer missing scientific requirements without evidence. Mark unresolved fields as pending confirmation.

### Evidence Classification

- **Confirmed requirement**: the selected config or its executing dataset path enforces the item. Cite the enforcing file, field, or code path.
- **Optional requirement**: the code handles the item only when present or provides a fallback when absent.
- **Pending confirmation**: the repository does not establish the exact value, ordering, units, preprocessing rule, source, or license.
- Docstrings, comments, and README statements are documentation evidence, not code-path evidence. When they conflict with the executing code path (for example a comment claiming the model normalizes inputs in `forward` while it only does so in `export_onnx`), the code path wins. Scientific conventions such as the normalization storage convention stay `Pending confirmation` unless verified from the executing path.

Do not describe optional or pending items as mandatory downloads. Variable lists used only for weighting, masking, accumulation handling, logging, or evaluation are not complete training-variable lists unless the selected config explicitly enforces them.

### Config Isolation

Before extracting requirements, list the candidate entry configs and confirm with the user which model(s) the plan covers. If the user selects one config, resolve only that entry config and trace only its imported base configs and dataset definitions.

Treat similarly named configs as separate contracts when their data path, grid, frequency, time range, variables, static channels, or preprocessing differ. If a family name such as `ViT` matches multiple configs, present the variants and ask which entry config is intended before producing a definitive plan.

Do not merge requirements from S2S, land, high-resolution, ensemble, or other variants into a single dataset contract. In multi-config mode, keep each config's contract separate and only deduplicate shared datasets in the aggregated list.

### Multi-Config Mode

When the user selects more than one entry config, produce a multi-config plan instead of a single-config plan.

- Extract requirements per config and keep each config's contract in its own block.
- List a shared dataset once when two or more configs use the same dataset path, resolution, frequency, and variable set; note every config that uses it.
- Never merge conflicting contracts: when configs differ in grid, frequency, time range, or variables, list separate dataset entries even if they share a source (for example, a 1.5° daily S2S dataset and a 0.25° 6-hourly ensemble dataset are separate entries).
- Aggregate size estimates after deduplication, and report both the per-config total and the deduplicated grand total.
- Report preflight status per config; shared items may be confirmed once and referenced by the configs that share them.
- Output as a single plan document with per-config blocks plus a deduplicated summary, or as one plan file per config plus a summary manifest.

### Claims Requiring Direct Verification

- Confirm the main dataset open path separately from sidecar-loading helpers before claiming consolidated or non-consolidated Zarr support.
- Label statistics and static files as required only when the selected runtime path enforces them; otherwise describe them as optional or pending confirmation.
- Do not claim that resizing, cropping, regridding, aggregation, normalization, or unit conversion is implemented unless the executing code path was traced.
- A README instruction to contact the author establishes the documented acquisition method, but does not prove that data is private, restricted, or unavailable elsewhere.
- When channel names come from a Zarr coordinate dynamically, report only explicitly enforced channels as confirmed unless this reference provides an authoritative default manifest for the selected dataset family. For S2S C76, use the default manifest below when the user has not requested a custom channel configuration; do not mark its list or order as pending merely because the runtime reads the Zarr coordinate dynamically.

### Default S2S C76 Channel Manifest

The default S2S channel configuration is the 76-channel order verified against the reference `cla.zarr` climate dataset and implemented by the preprocessing tools. Use it for S2S download plans unless the selected config explicitly enforces a different contract or the user requests a custom channel list/order.

Pressure-level channels are variable-major. Within each variable, pressure decreases from 1000 to 50 hPa. The complete ordered manifest is:

```yaml
channel_profile: s2s_c76_default
channels:
  - z1000
  - z925
  - z850
  - z700
  - z600
  - z500
  - z400
  - z300
  - z250
  - z200
  - z150
  - z100
  - z50
  - t1000
  - t925
  - t850
  - t700
  - t600
  - t500
  - t400
  - t300
  - t250
  - t200
  - t150
  - t100
  - t50
  - u1000
  - u925
  - u850
  - u700
  - u600
  - u500
  - u400
  - u300
  - u250
  - u200
  - u150
  - u100
  - u50
  - v1000
  - v925
  - v850
  - v700
  - v600
  - v500
  - v400
  - v300
  - v250
  - v200
  - v150
  - v100
  - v50
  - q1000
  - q925
  - q850
  - q700
  - q600
  - q500
  - q400
  - q300
  - q250
  - q200
  - q150
  - q100
  - q50
  - t2m
  - d2m
  - sst
  - ttr
  - 10u
  - 10v
  - 100u
  - 100v
  - msl
  - tcwv
  - tp
```

In every S2S pre-download plan:

- State that `s2s_c76_default` will be used when the user has not supplied a channel override.
- Show the complete ordered manifest to the user before download, not only `5 variables x 13 levels + 11 surface channels`.
- Tell the user that they may keep the default or provide a custom channel list and exact order.
- Treat an explicit user order as authoritative and preserve it through download mapping and conversion.
- Warn that changing the channel count or order changes the data/model contract and may make existing checkpoints, normalization sidecars, or reference datasets incompatible.
- Do not block a download solely to reconfirm the default C76 order. Other unresolved requirements, such as source, units, accumulation windows, time coverage, destination, or overwrite policy, may still block execution.

## Pre-Download Plan

Record:

- Selected config(s) and traced base configs (per config in multi-config mode).
- Dataset and proposed source.
- Confirmed variables and levels.
- Optional requirements.
- Pending confirmations.
- Shared datasets and deduplication notes (multi-config mode).
- Start and end time.
- Time coverage split: training vs validation/test ranges when the config defines them, or the inference history window for inference configs.
- Temporal frequency.
- Spatial extent and resolution.
- Download format, target format, and conversion step (see Format Conversion Chain).
- Static fields and statistics sidecars expected by the config.
- Normalization convention: the core convention stores normalized values in Zarr (model forward does not re-normalize); record that the prepared dataset follows it. Precipitation channels use **mm accumulated values** (see Format Conversion Chain). Inference input form depends on the model's export/inference code and should be confirmed per model.
- Destination directory, following the directory layout defined in data-preprocessing.md (download staging, converted Zarr, sidecars).
- Estimated file count and size when known.
- Authentication method without credentials.
- Existing-file and overwrite policy.

Prefer a machine-readable YAML or JSON manifest when the plan will be reused. Creating a manifest file is itself a filesystem write; print it in the response unless the user explicitly asks to save it.

## Format Conversion Chain

A download plan must record the full chain from the downloaded format to the format the selected config consumes, so the plan stays actionable without rediscovering tooling later.

Map the chain for every planned dataset:

- **Download format**: the format the provider offers (for example, GRIB, NetCDF, CINRAD binary, NPZ).
- **Reading toolchain**: the library that reads that format (for example, xarray for NetCDF, cfgrib or eccodes for GRIB, the core radar reader for CINRAD binaries).
- **Target format**: the format the selected config consumes (Zarr in this model library, with `mean`, `std`, and `weight` sidecars).
- **Conversion step**: the concrete transformation (for example, `NetCDF -> Zarr via xarray.to_zarr`, including variable rename, time alignment, and coordinate normalization).

Rules:

- When a provider offers multiple download formats, prefer the one that matches the available conversion toolchain and record the choice and its reason.
- Record conversion dependencies explicitly; do not assume a reader library is available at execution time.
- The plan marks the chain as feasible or pending; executing the conversion belongs to Data preprocessing, not to this route.
- CDS delivery form is dataset-specific and must not be assumed: `reanalysis-era5-land` returns a ZIP archive by default even for single-variable requests, so record `download_format: unarchived` in the request and keep unzipping as a fallback step; `reanalysis-era5-single-levels` returns a plain file by default.
- On the current CDS backend, ERA5 requests should use `date` plus `product_type: reanalysis`; the `year`/`month`/`day` form fails with `Duplicate value for month`. Record the exact request fields so the chain stays actionable without rediscovery.
- Precipitation channels are normalized to **mm accumulated values** regardless of source. Sources differ: ERA5/ERA5-Land deliver `tp` in metres (step-accumulated, ×1000 for mm), some products deliver rates (`kg m-2 s-1`, `mm/h`, `mm/s`, needing × accumulation seconds), and others already deliver mm accumulations with a specific window. Record the source's original unit and window for every precipitation variable, the conversion factor, and the target accumulation window from the model contract (for example daily totals for S2S, 6-hourly for IWC, hourly for ERA5-Land-based models).

## Static Fields Acquisition

Static fields (orography, land-sea mask, soil type) and statistics sidecars are separate from the time-series download:

- **ERA5**: request the invariant variables (`geopotential`, `land_sea_mask`, `soil_type`) from `reanalysis-era5-single-levels` with a single time point; they are time-invariant. Request them on the same grid and area as the training data.
- **ERA5-Land**: the orography and land-sea mask are not in the CDS catalogue. Download `geo_1279l4_0.1x0.1.grib2_v4_unpack.nc` and `lsm_1279l4_0.1x0.1.grb_v4_unpack.nc` from the ERA5-Land data documentation (Parameter listings attachments). Other invariants (soil, vegetation) come from ERA5/IFS documentation or external mirrors.
- The static field grid must match the training data grid, or be resampled to it.
- Statistics sidecars (`mean`, `std`, `weight`) are computed from the prepared dataset, not downloaded.

## Source Retrieval Guidance

When a download-list item has no confirmed source, do not hard-code a dataset-specific channel. Instead:

- Mark the item as source-unconfirmed in the plan.
- Search the web for possible channels for that data (official site, CDS, GitHub, Kaggle, academic mirrors, and so on), preferring official sources.
- Output a channel list per item, each entry naming: the channel, authority (official / mirror / community), licence and authentication requirements, expected format and approximate size, and how to obtain it (link, API, or command).
- Stop at the channel list; do not download. The user picks a channel and runs the download, then the agent verifies the files (format detection) before conversion.
- Mark unverified channels as pending confirmation; never claim a channel is official without checking.

## Preflight Status

Before declaring the plan complete, report whether these items are confirmed:

- Exact entry config.
- Data source and license.
- Variable and level mapping.
- Time range, frequency, extent, and resolution.
- Destination and expected storage.
- Existing-file policy.
- Secure authentication mechanism.

Never include API keys, tokens, passwords, or other credentials in plans, commands, logs, or repository files.

## Required Output

Use `assets/templates/data_download_plan.md` as the standard output structure. Fill only fields supported by repository evidence, write `Pending confirmation` when a value is unresolved, and print the completed plan in the response unless the user explicitly asks to save it.

Keep the printed plan short and readable:

- Lead with a one-sentence bottom line: what to download, from where, approximate size, and where to place it.
- Show the download list as two compact tables: a main requirements table (purpose, dataset, variables, frequency, time range, grid, estimated size) and a download & conversion table (source dataset, download format, conversion step, target format). Avoid a single wide table with more than about seven columns.
- For training, split the download rows by the config's train and validation/test time ranges; for inference, list only the inference history window and the required sidecars.
- Name the source dataset explicitly in every download row (CDS catalogue name or author-provided data) and include the conversion step from source to target, so the plan is executable without re-deriving the download or conversion details.
- Include static fields in the download list only when the model contract requires them (for example non-empty `maskid`, non-empty `static_fields`, or required `const`); omit the static-field row for models that do not need them. Statistics sidecars (`mean`/`std`/`weight`) are always required for Zarr grid models and should be listed with their source (computed from the prepared dataset).
- Print download scripts and steps/config files in the response instead of writing them to the workspace; creating such files is itself a filesystem write and requires the user to ask for it explicitly.
- Do not state unverified scientific conclusions (for example the normalization storage convention) in the bottom line or the download list; put them in Pending Confirmation unless verified from the executing code path.
- Keep evidence and config-internal details (code paths, helper names, constants) out of the printed plan. Evidence is collected during extraction but shown only when the user asks for justification.
- Do not print optional requirements or a preflight table as separate sections; fold "not needed" items into one line and keep pending items to those that block the download or the conversion.
- State the recommended next step and the blocking confirmations.

The template covers:

```text
Bottom line
Download list (deduplicated; per-config blocks when contracts differ)
Data source
Pending confirmation (blocking items only)
Next step
```

For the planning response, end by stating that no download, conversion, file creation, or data modification was performed and ask the user to confirm the listed download before execution. This statement applies only to the planning response. Once the user confirms and asks to proceed, exit this route rather than repeating it. Do not report file integrity, schema validation, or training readiness under the planning route; hand those tasks to Data analysis after data exists.
