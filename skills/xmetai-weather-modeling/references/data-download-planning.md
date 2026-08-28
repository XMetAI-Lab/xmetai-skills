# Data Download Planning

Use this reference to produce a pre-download plan. It does not govern the later confirmed execution or post-download validation stages.

## Scope

This route answers: **What data does the selected config require, where may it come from, and what must be confirmed before downloading it?**

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

Confirmation covers only the planned source, variables, raw time ranges, destination, estimated size, authentication mechanism, and existing-file policy. Material changes require an updated plan. Download completion does not authorize conversion, Zarr mutation, or training-readiness claims.

## Requirement Extraction

Classify the entry config as training or inference first:

- **Training configs** (for example `*_base.py`, `*_vit.py`, `*_lora.py`): extract the full training and validation/test coverage from `train_times`/`test_times` (or the training-script defaults) and record the split explicitly in the plan. Note overlaps when the config places the same period in both train and test.
- **Inference configs** (for example `*_infer.py`): they reuse the base dataset contract (channels, frequency, grid) but only need the history window before each inference time. Record the inference window instead of the full training coverage. The exported ONNX interface takes physical values in the current models, but the core inference wrapper may receive normalized inputs — confirm the export/inference code before preparing inference data. `mean`/`std` sidecars remain required, `weight` is not needed, and targets are not downloaded.

Check:

- Dataset class and configured data paths.
- Input and output variable names and channel counts.
- Historical and forecast frames.
- Training and evaluation time coverage required from the data.
- The executing dataset indexer's first/last admissible sequence, including strict endpoint exclusions or other implementation-only buffer frames.
- Time frequency and sample interval.
- Per-variable source time semantics and aggregation coverage: source interval, source timestamp meaning, target window, target label, and required leading/trailing source boundary.
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

### Time Coverage Derivation

Do not derive the download range only as `initialization time + maximum forecast lead`. Produce three explicit ranges:

1. **Requested model range**: the history, initialization, and physical target timestamps the user intends to train or evaluate.
2. **Prepared model-frame range**: the timestamps that must exist after conversion so the executing dataset class actually emits every requested sample.
3. **Source download range**: the raw timestamps needed per variable group to create that prepared range.

Trace the executing dataset indexer rather than assuming an ideal inclusive range. Inspect how it computes `total_frames`, maps a sequence index to initialization time, checks continuity, and bounds its loop. A strict bound such as `i < len(times) - total_frames` or `range(len(times) - total_frames)` excludes a sequence that ends exactly on the last available frame; if the code is not being fixed, include and label the additional trailing frame as a **runtime implementation buffer**, not as a physical forecast target. Apply a loader-required buffer to every model channel that must share the prepared time axis.

For every variable that is resampled, rolled, differenced, accumulated, averaged, minimized, or maximized over time, derive its raw boundary from configuration and source metadata, not from its name. Record:

- source temporal representation: instantaneous, interval accumulation, interval mean/rate, or running accumulation since a forecast start/reset;
- source timestamp meaning: instant, interval start, or interval end;
- source interval and expected timestamp cadence;
- target operator and window length;
- target label convention and label hour;
- incomplete-window policy;
- exact leading/trailing source coverage needed for the prepared output range.

For interval-ending hourly samples aggregated over `W` hours and labelled at the window end, prepared outputs `[S, E]` require raw samples `[S - (W - 1) hours, E]`. For the same inputs labelled at the window start, outputs `[S, E]` require `[S + 1 hour, E + W hours]`. These formulas are examples, not universal conventions: adjust them when the provider labels intervals differently. If the provider already supplies the exact target window with matching labels, no aggregation boundary extension is needed. Running accumulations must be de-accumulated with their reset/forecast-step semantics before applying a new window; never sum them as if they were independent intervals.

Keep variable groups separate when their source ranges differ. Instantaneous daily fields, hourly accumulated fields, and static fields may therefore have different download rows. After deriving each group, verify that their converted time-axis intersection still covers the complete prepared model-frame range. Never use an ambiguous phrase such as "the 10 October daily value" without also stating the window and whether the timestamp labels its start or end.

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

Use the required output template. In addition to source, variables, grid, frequency, formats, destination, size, authentication, and overwrite policy, record the three time ranges from **Time Coverage Derivation** per variable group and separate confirmed, optional, and blocking-pending items. In multi-config mode, preserve per-config contracts and deduplicate only identical source datasets. Print reusable YAML/JSON unless the user explicitly asks to save it.

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
- Apply the Time Coverage Derivation rules to every temporally transformed variable, not only named precipitation or radiation channels. Precipitation channels additionally normalize to **mm accumulated values** regardless of source. Sources differ: ERA5/ERA5-Land deliver `tp` in metres (step-accumulated, ×1000 for mm), some products deliver rates (`kg m-2 s-1`, `mm/h`, `mm/s`, needing × accumulation seconds), and others already deliver mm accumulations with a specific window. Record the source's original unit and window, conversion factor, target window, and time-label convention for each affected variable.

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

## Required Output

Use `assets/templates/data_download_plan.md` as the standard output structure. Fill only fields supported by repository evidence, write `Pending confirmation` when a value is unresolved, and print the completed plan in the response unless the user explicitly asks to save it.

Keep the printed plan short:

- Lead with what to download, source, size, and destination.
- Use the template's requirements and conversion tables; split rows when configs or raw time ranges differ.
- Show prepared versus raw time coverage and explain every boundary extension.
- Include only model-required static fields; list Zarr sidecars as computed outputs.
- Keep evidence/internal helpers out unless requested, and show only blocking pending items.
- Print scripts/configs instead of saving them unless the user asks for files.

End the planning response by stating that nothing was written and asking for confirmation. After confirmation, exit this route. File integrity and training readiness belong to Data analysis after download/conversion.
