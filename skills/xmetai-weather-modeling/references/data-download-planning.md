# Data Download Planning

Use this reference only for extracting meteorological data requirements and producing a plan before any download or data write begins.

## Scope

This route answers: **What data does the selected config require, where may it come from, and what must be confirmed before downloading it?**

Do not download, create, convert, aggregate, regrid, normalize, append, overwrite, or otherwise modify data. Do not generate statistics or claim that a dataset is ready for training.

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
9. Stop before any network request or filesystem write.

## Requirement Extraction

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
- When channel names come from a Zarr coordinate dynamically, report only explicitly enforced channels as confirmed. Keep the complete list and order pending until an authoritative schema or manifest is available.

## Pre-Download Plan

Record:

- Selected config(s) and traced base configs (per config in multi-config mode).
- Dataset and proposed source.
- Confirmed variables and levels.
- Optional requirements.
- Pending confirmations.
- Shared datasets and deduplication notes (multi-config mode).
- Start and end time.
- Temporal frequency.
- Spatial extent and resolution.
- Download format, target format, and conversion step (see Format Conversion Chain).
- Static fields and statistics sidecars expected by the config.
- Destination directory.
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

The template covers:

```text
Selected config(s)
Confirmed requirements (one block per config)
Shared datasets and deduplication (multi-config)
Optional requirements
Pending confirmation
Data source
Format conversion chain
Destination and estimated size
Preflight status (per config)
Next step
```

End by stating that no download, conversion, file creation, or data modification was performed. Do not report file integrity, schema validation, or training readiness under this route; hand those tasks to Data analysis after data exists.
