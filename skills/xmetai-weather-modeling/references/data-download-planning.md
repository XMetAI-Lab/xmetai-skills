# Data Download Planning

Use this reference only for extracting meteorological data requirements and producing a plan before any download or data write begins.

## Scope

This route answers: **What data does the selected config require, where may it come from, and what must be confirmed before downloading it?**

Do not download, create, convert, aggregate, regrid, normalize, append, overwrite, or otherwise modify data. Do not generate statistics or claim that a dataset is ready for training.

Use **Data analysis** instead when data already exists and the task is to inspect schema, integrity, time gaps, values, statistics, static fields, or training readiness.

## Workflow

1. Inspect the target repository and identify the exact entry config requested by the user.
2. Trace its reused base configs and dataset definitions.
3. Extract the dataset contract: variables, levels, time range, frequency, spatial extent, resolution, format, static fields, and statistics sidecars.
4. Classify each requirement as confirmed, optional, or pending confirmation.
5. Identify the documented or proposed data source, license, authentication method, and variable-name mapping.
6. Record the destination, estimated size when known, and existing-file policy.
7. Produce a Markdown, YAML, or JSON pre-download plan.
8. Stop before any network request or filesystem write.

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

Resolve one entry config and trace only its imported base configs and dataset definitions. Treat similarly named configs as separate contracts when their data path, grid, frequency, time range, variables, static channels, or preprocessing differ.

If a family name such as `ViT` matches multiple configs, list the variants briefly and ask which entry config is intended before producing a definitive plan. Do not merge requirements from S2S, land, high-resolution, ensemble, or other variants.

### Claims Requiring Direct Verification

- Confirm the main dataset open path separately from sidecar-loading helpers before claiming consolidated or non-consolidated Zarr support.
- Label statistics and static files as required only when the selected runtime path enforces them; otherwise describe them as optional or pending confirmation.
- Do not claim that resizing, cropping, regridding, aggregation, normalization, or unit conversion is implemented unless the executing code path was traced.
- A README instruction to contact the author establishes the documented acquisition method, but does not prove that data is private, restricted, or unavailable elsewhere.
- When channel names come from a Zarr coordinate dynamically, report only explicitly enforced channels as confirmed. Keep the complete list and order pending until an authoritative schema or manifest is available.

## Pre-Download Plan

Record:

- Selected config and traced base configs.
- Dataset and proposed source.
- Confirmed variables and levels.
- Optional requirements.
- Pending confirmations.
- Start and end time.
- Temporal frequency.
- Spatial extent and resolution.
- Expected format and layout.
- Static fields and statistics sidecars expected by the config.
- Destination directory.
- Estimated file count and size when known.
- Authentication method without credentials.
- Existing-file and overwrite policy.

Prefer a machine-readable YAML or JSON manifest when the plan will be reused. Creating a manifest file is itself a filesystem write; print it in the response unless the user explicitly asks to save it.

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
Selected config
Confirmed requirements
Optional requirements
Pending confirmation
Data source
Destination and estimated size
Preflight status
Next step
```

End by stating that no download, conversion, file creation, or data modification was performed. Do not report file integrity, schema validation, or training readiness under this route; hand those tasks to Data analysis after data exists.
