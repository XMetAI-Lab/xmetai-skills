# Data Preprocessing

Use this reference after download to detect formats and execute guarded NetCDF/Zarr/GRIB conversion, normalization sidecars, and optional static-field conversion. It neither downloads data nor claims training readiness; Data analysis owns the final judgment.

## Scope

No write occurs without explicit approval.

## Workflow

1. Run `inspect_data_format.py --path <file-or-dir>` to detect formats and read metadata (dims, variables, units).
2. Derive the steps config from the download plan's format conversion chain and the selected config's contract (variables, frequency, time range).
3. Dry-run `convert_to_zarr.py` and show the before/after plan to the user.
4. After explicit user approval, run with `--allow-write --ack-risk I-understand-this-mutates-zarr`; the Zarr write guard executes before any mutation.
5. Report the conversion summary, then hand the output to Data analysis for schema and training-readiness checks.

A `normalize` step writes `mean`/`std`/`weight` beside the output Zarr. Prepare required static fields separately as `const.nc`.

## Format Support

| Input | Detection | Conversion | Status |
|---|---|---|---|
| NetCDF (classic CDF / NetCDF-4 HDF5) | magic `CDF` / HDF5 | xarray -> Zarr | supported |
| Zarr store | `.zgroup` / `zarr.json` | normalize -> Zarr | supported |
| GRIB | magic `GRIB` | cfgrib/eccodes (pip-installable on Windows) | supported when cfgrib installed; `decode-pending` otherwise (multi-variable ERA5-Land GRIB: see Format Support Notes) |
| CINRAD `.bin` / NPZ | extension / magic | needs the core radar reader | pending |

## Format Support Notes

CDS may deliver a ZIP archive instead of a bare data file:

- `reanalysis-era5-land` returns a ZIP archive by default, even for a single-variable, single-day request. The archive contains one NetCDF or GRIB member. Add `download_format: unarchived` to the download request to receive a bare file; keep ZIP detection as a fallback because the option may not always be honoured.
- When a delivery is a ZIP, the extension and magic bytes disagree: `inspect_data_format.py` reports `mismatch` with `container: zip` and the member names. Extract the member before running the conversion chain.

GRIB decoding has an additional limit:

- Multi-variable `reanalysis-era5-land` GRIB files mix GRIB edition 1 (`2t`, `tp`) and edition 2 (`sde`) messages, so cfgrib cannot open the whole file as one dataset (`DatasetBuildError: multiple values for key 'edition'`). `convert_to_zarr.py` handles this automatically: it reads each variable with `filter_by_keys` and merges them. The merged dataset keeps the forecast-step structure (`time`, `step`, `lat`, `lon`); NetCDF remains the preferred download format when the target contract needs a plain daily or hourly `time` axis.

## Scripts

### inspect_data_format.py

Read-only format detection and metadata summary.

```text
python inspect_data_format.py --path <file-or-directory> [--json]
python inspect_data_format.py --path <file-or-directory> --config expected.json
```

Statuses: `recognized`, `recognized-by-magic`, `mismatch`, `decode-pending`, `decode-error`, `unsupported`. Use `inspect_zarr_schema.py` for deep Zarr checks and `inspect_static_nc.py` for static NetCDF; do not repeat the same shallow inspection.

### convert_to_zarr.py

Dry-run conversion plan first; guarded write on approval.

```text
python convert_to_zarr.py --input in.nc --output out.zarr [--steps-config steps.json]
python convert_to_zarr.py --input-glob "era5_*.nc" --output out.zarr [--input-chunks time=4]
python convert_to_zarr.py --input-glob "era5_*.nc" --output out.zarr --batch-time 31
python convert_to_zarr.py --input-glob "era5_*.nc" --output out.zarr --input-period-batch 1 --input-overlap-periods 1
python convert_to_zarr.py --input-glob "era5_*.nc" --output out.zarr --batch-time 31 --resume
python convert_to_zarr.py --input in.nc --output out.zarr --allow-write --ack-risk I-understand-this-mutates-zarr [--overwrite]
python convert_to_zarr.py --input static.nc --output const.nc --output-format static-netcdf --steps-config static.json
```

Inputs are NetCDF, Zarr, or supported GRIB. Writes require explicit approval flags; Zarr also runs the write guard. Static mode writes `const(channel, lat, lon)` and existing outputs require `--overwrite`.

`convert_to_zarr.py` is the stable CLI and compatibility facade. Its reusable implementation is split under `scripts/conversion/` by responsibility: input opening, grid handling, temporal aggregation, configured transforms, bounded batching, streaming statistics, resume state, Zarr/sidecar writers, and static fields. Add dataset-specific behavior to the narrowest module while keeping the CLI and declarative steps contract stable.

Repeat `--input` or use `--input-glob` for a homogeneous NetCDF collection. Multi-file inputs are opened lazily with `open_mfdataset(combine="by_coords")`; file opening is serial for Windows netCDF4/HDF5 safety, while downstream Dask reductions and writes remain chunked. Dry-run inspects metadata and the planned transforms only; it deliberately defers full-data normalization statistics until the guarded write phase. Use `--input-chunks` and `--output-chunks` to control memory and I/O. The default output layout keeps one time step per chunk and complete channel/spatial dimensions, matching full-field weather-model reads without loading the whole time series.

NetCDF-4/HDF5 inputs use `h5netcdf` when it is installed, including the one-file-at-a-time catalog pass and later multi-file batches. This avoids observed Windows netCDF4/HDF5 failures when a file is closed after metadata inspection and reopened for statistics. Classic CDF inputs retain the compatible netCDF4/xarray fallback; do not force `h5netcdf` for every `.nc` extension.

For multiple NetCDF inputs, the converter first catalogs files one at a time by source time coverage, then opens only a bounded number of chronological periods. `--input-period-batch` defaults to one period and `--input-overlap-periods` defaults to one neighboring period on each side. The overlap preserves transformations such as daily accumulation across month/year boundaries; duplicate prepared times are removed before statistics and writes. Files sharing the same source time coverage remain in one period so pressure-level and surface variable files can still be merged. The dry-run previews only the first bounded group instead of opening the complete collection. The conversion state records the maximum number of simultaneously grouped input files observed.

Zarr output is written incrementally along `time`; `--batch-time` sets the number of prepared output frames per append batch and defaults to 31. After each successful batch, `<output>.conversion.json` is atomically updated. Use `--resume` after an interrupted run: the converter reads the actual Zarr time coordinate and continues only when it is the exact prefix reproduced by the bounded file batches, while rejecting gaps, overlaps, reordered times, changed transformation/chunk/batching settings, and incompatible variables, dimensions, or coordinates. The state file is an audit record rather than the source of write progress, so a crash between a successful append and the state update does not duplicate data. Resume remains a Zarr mutation and requires the normal write approval flags.

When `normalize` is enabled for a multi-file conversion, each prepared file batch computes per-channel `count`, `mean`, and `M2`; the batches are combined with the Chan/Welford formula and persisted in the conversion state before the normalized write pass. This bounds the statistics task graph and handles NaNs with a separate valid count per channel. Resume of normalized output requires the identical input set so previously written values keep the same global statistics; extending a normalized store requires a separate statistics/renormalization workflow.

An interrupted statistics pass may leave the conversion state before any Zarr store exists. `--resume` accepts that state, recomputes the bounded statistics, and creates the store instead of attempting to open a missing group. A batch with zero valid samples for one channel contributes `count=0, M2=0` and does not poison later Chan/Welford merges. Before writing normalized data, the converter rejects any channel whose final count is zero or whose mean, M2, or standard deviation is non-finite.

cfgrib index files (`.idx`) are not written: all GRIB reads pass `indexpath=""`. The `merge_to_data` step combines data variables into a single `data` variable with a `level`/`channel` coordinate, matching the model library layout.

New CDS ERA5 downloads name the time coordinate `valid_time` instead of `time`; add a rename step (`{"rename": {"valid_time": "time"}}`) to normalize to the model library convention.

### compute_sidecars.py

Compute sidecars from physical, unit-converted and log-transformed values before normalization.

```text
python compute_sidecars.py --input out.zarr --output-dir <dir>
python compute_sidecars.py --input out.zarr --output-dir <dir> --chunks time=4 --allow-write [--overwrite]
```

Writes channel-indexed `mean.nc` / `std.nc` and latitude-indexed `weight.nc`. Statistics use one lazy, chunk-bounded full-data reduction over unit-converted and log-transformed values; zero variance is stored as `std=0`, which core treats as 1. Spatial weight is `cos(abs(lat))`, matching core's `(1,H,1)` broadcast convention. Core derives its separate level-scaled `channel_weights` buffer from channel names; it must not be stored as `weight.nc`.

## Steps Config

JSON or YAML. Unknown steps abort the plan.

```json
{
  "steps": [
    {"rename": {"total_cloud_cover": "clt"}},
    {"keep_vars": ["z", "t", "q"]},
    {"time": {"start": "2023-06-01", "end": "2023-06-02"}},
    {"resample": {"freq": "6h", "operator": "mean"}},
    {"daily_aggregation": {"window_hours": 24, "label_hour": 0, "incomplete": "error", "variables": {"tp": {"operator": "sum", "factor": 1000, "units": "mm"}, "ttr": {"operator": "sum", "factor": 0.000011574074074074073, "units": "W m-2"}}}},
    {"regrid": {"target": "s2s_1.5deg", "method": "linear", "variable_methods": {"lsm": "nearest"}}},
    {"merge_static": {"order": ["z", "lsm"], "coord": "channel", "name": "const"}},
    {"units": {"q": 1000}},
    {"log1p": ["tp"]},
    {"split_levels": {"vars": ["z", "t", "u", "v", "q"], "level_coord": "pressure_level"}},
    {"merge_to_data": {"coord": "level"}},
    {"normalize": true}
  ]
}
```

`merge_to_data` concatenates variables into `data`. Without `order`, known channels follow the canonical C76 relative order (five variables at 1000→50 hPa, then `t2m,d2m,sst,ttr,10u,10v,100u,100v,msl,tcwv,tp`); subsets remain valid and unknown channels are appended. Explicit `order` is authoritative. `split_levels` expands pressure-level variables, `units` applies factors such as `q` ×1000, and `log1p` clips at zero first. Use `daily_aggregation` rather than separate generic factors when conversion also requires a daily window.

`daily_aggregation` provides variable-independent hourly-to-daily aggregation. Input timestamps are interpreted as interval ends. `window_hours` controls the rolling window (1–24), `label_hour` selects the UTC hour used for each daily output label, and every entry under `variables` selects `sum`, `mean`, `min`, or `max` plus an optional numeric `factor`, `offset`, and output `units`. Multiple custom variables can be aggregated in one step; for example `ssr` uses the same `sum` and `1/86400` factor as `ttr` when converting accumulated J m-2 to daily-mean W m-2. Variables not listed under `variables` are retained at the resulting daily timestamps, and their original order is preserved. Arrays remain lazy.

For the default S2S convention, a value labelled at 00 UTC uses the 24 samples from 01 UTC of the previous day through 00 UTC of the labelled day. The example sums `tp` and converts metres to **mm per 24 hours**, while `ttr` is summed and multiplied by `1/86400` to produce **W m-2**. `incomplete` defaults to `error`, rejecting missing hours and leading/trailing partial windows; `drop` explicitly discards partial windows. Download the preceding day's 01–23 UTC boundary hours when the first requested output day must be retained. Do not apply additional unit factors to variables already converted by this step. The older `s2s_daily_accumulation` step remains as a backward-compatible `tp/ttr` shorthand, but new configurations should use `daily_aggregation`.

Precipitation unit convention: precipitation channels are normalized to **mm accumulated values** before `log1p`/`normalize`. ERA5 and ERA5-Land deliver `tp` in metres (step-accumulated). A direct conversion at the source accumulation interval uses ×1000; S2S daily data should instead use `daily_aggregation`, which performs the 24-hour sum and conversion together. Rate-form precipitation (`kg m-2 s-1`, `mm/h` averages, or `mm/s`) must first be multiplied by the accumulation length in seconds. The accumulation window must follow the selected model contract (for example daily totals for S2S, 6-hourly for IWC, hourly for ERA5-Land-based models) and must be recorded in the download plan so a given data source's original unit and window can be verified.

Output Zarr stores always use `lat`/`lon` as the grid coordinate names; CDS inputs carrying `latitude`/`longitude` are renamed automatically. This matches the core dataset classes, which access `ds.lat`/`ds.lon` directly (for example `GraphCastDataset` and the `MultiZarrDataset` bbox path).

`regrid` converts rectilinear ERA5 grids to a named model grid before channel merging and normalization. The `s2s_1.5deg` target is fixed to descending latitude `90, 88.5, ..., -90` (121 points) and longitude `0, 1.5, ..., 358.5` (240 points), matching the `cla.zarr` spatial contract. Source longitudes in `-180..180` are normalized to `0..360`; source latitude and longitude coordinates must be one-dimensional and unique. Standard ERA5 0.25-degree data contains every target point, so this aligned case uses exact lazy indexing without SciPy. Non-aligned source grids use xarray interpolation and require SciPy. `method` defaults to `linear` for continuous fields; use `variable_methods` with `nearest` for categorical fields such as land masks or soil type. Run this step before `merge_to_data` and `normalize`. Conservative remapping is not provided, so precipitation remapping must not be described as area-conservative.

`merge_static` is used only with `--output-format static-netcdf`. It selects static variables in the configured order, requires every `time` or `valid_time` dimension to contain exactly one value, removes that singleton dimension, and writes `const(channel, lat, lon)`. It rejects duplicate or missing variables, additional dimensions, an empty output, and normalization. Choose variable-specific regridding before this step (`linear` for continuous terrain/geopotential and `nearest` for categorical masks). The number and order of output channels must follow the selected model's `const_chans` contract; models with `const_chans=0` do not need this conversion.

`normalize` stores `(x - mean) / std` and writes matching sidecars. Source NaNs remain in Zarr; core replaces them at load time. Land/ocean channel corrections belong to core's separately derived `channel_weights`; they do not modify the latitude-based `weight.nc`.

`flatten_step` merges the `time` and `step` dimensions of a GRIB-derived dataset into a single `time` axis using the `valid_time` coordinate, drops all-NaN combinations (for example ERA5-Land short-forecast GRIB frames outside the requested window), and reorders to `(time, level, lat, lon)`. Use it after `merge_to_data` for GRIB inputs that carry a `step` dimension.

`merge_normalize.py` verifies time/grid alignment, merges channel stores, computes shared statistics, and writes one normalized Zarr plus sidecars without loading the full dataset. Its default order follows the same C76-relative rule; `--order` must list every input channel exactly once. Do not normalize component stores first.

## Normalization Convention

Training consumes normalized Zarr values directly: accumulated precipitation is converted to mm, log-transformed when required, then all channels use `(x - mean) / std`. Sidecars invert this for evaluation. Current exported ONNX graphs generally expose physical values, while the core cascade wrapper may receive normalized inputs; verify the selected model's export/inference path.

## Directory Layout

Keep downloaded source data, conversion products, and sidecars in separate, predictable locations so the plan stays executable and inspection is unambiguous:

- Download staging: the raw files delivered by the provider (NetCDF, GRIB, ZIP archives) go to a staging directory, not into the final data directory.
- Converted product: one Zarr store per dataset, placed at the destination recorded in the download plan.
- Sidecars and static fields: `mean`, `std`, `weight`, `mask`, `land_mask`, `const` files live in the same directory as the Zarr store (`.nc`, `.npy`, or Zarr variables).
- Naming: use the dataset name from the download plan (for example `s2s.1950-2024.c76`); do not mix multiple test or comparison datasets in one folder.

Keep credentials out of steps configs and logs. Unknown scientific requirements remain pending instead of being guessed.

