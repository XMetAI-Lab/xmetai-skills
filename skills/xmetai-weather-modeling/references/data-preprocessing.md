# Data Preprocessing

Use this reference when downloaded data exists and needs to be parsed, normalized, converted, or validated before training. This route executes the format chain planned under Data download planning and stops before any training-readiness claim.

## Scope

This route answers: **What format is the downloaded data, how is it normalized, and is the converted result consistent with the selected config?**

It covers:

- Format detection: extension and magic bytes first, then metadata read.
- Declarative conversion steps: variable rename, variable selection, time selection, resampling, channel merge.
- Optional static-field conversion to a core-compatible `const.nc` when the selected model requires it.
- Guarded conversion to Zarr.
- Post-conversion validation: hand the converted store to Data analysis, where `inspect_zarr_schema.py` describes the store and the agent judges training readiness against the model contract.

It does **not**:

- Download data or touch the network. Download planning produces the plan; execution happens in the user's environment.
- Claim training readiness. Hand that to Data analysis after the data exists.
- Write without approval. Conversion is dry-run by default; real writes require explicit flags and the Zarr write guard.

Use **Data download planning** when data is missing and only a plan is requested. Use **Data analysis** when data already exists and needs read-only schema, integrity, statistics, or training-readiness checks.

## Workflow

1. Run `inspect_data_format.py --path <file-or-dir>` to detect formats and read metadata (dims, variables, units).
2. Derive the steps config from the download plan's format conversion chain and the selected config's contract (variables, frequency, time range).
3. Dry-run `convert_to_zarr.py` and show the before/after plan to the user.
4. After explicit user approval, run with `--allow-write --ack-risk I-understand-this-mutates-zarr`; the Zarr write guard executes before any mutation.
5. Hand the converted store to Data analysis: run `inspect_zarr_schema.py` for the store description (dims, variables, coordinates, chunks, statistics) and let the agent judge training readiness against the model contract.
6. Report the conversion summary. Training-readiness validation belongs to Data analysis.

Conversion completes data readiness for the main data: a `normalize` step writes `mean`/`std`/`weight` sidecars into the output directory. Static fields are prepared separately (download plan provides the source; `inspect_static_nc.py` checks their content) and their presence is verified by the agent at the Data analysis stage, not by the conversion script.

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

Statuses: `recognized`, `recognized-by-magic`, `mismatch` (extension and magic disagree), `decode-pending` (GRIB without cfgrib), `decode-error` (GRIB decode failure), `unsupported`.

Use `inspect_data_format.py` to identify the format and read shallow metadata. For deep structure checks on a known Zarr store (chunks, dtype, sample statistics), use `inspect_zarr_schema.py`; for static NetCDF field files, use `inspect_static_nc.py`. Do not run them redundantly for the same shallow metadata.

### convert_to_zarr.py

Dry-run conversion plan first; guarded write on approval.

```text
python convert_to_zarr.py --input in.nc --output out.zarr [--steps-config steps.json]
python convert_to_zarr.py --input-glob "era5_*.nc" --output out.zarr [--input-chunks time=4]
python convert_to_zarr.py --input in.nc --output out.zarr --allow-write --ack-risk I-understand-this-mutates-zarr [--overwrite]
python convert_to_zarr.py --input static.nc --output const.nc --output-format static-netcdf --steps-config static.json
```

The write path requires explicit approval flags; Zarr output additionally runs the Zarr write guard. Inputs: NetCDF (classic/NetCDF-4), Zarr, or GRIB (multi-variable mixed-edition GRIB is read per variable automatically). Outputs are a main Zarr store or, in static mode, one `const.nc` file.

Static conversion uses the same dry-run and explicit write acknowledgement. With `--output-format static-netcdf`, a `merge_static` step writes a single core-compatible `const(channel, lat, lon)` DataArray instead of a Zarr store. Existing files are never replaced unless `--overwrite` is supplied.

Repeat `--input` or use `--input-glob` for a homogeneous NetCDF collection. Multi-file inputs are opened lazily with `open_mfdataset(combine="by_coords")`; file opening is serial for Windows netCDF4/HDF5 safety, while downstream Dask reductions and writes remain chunked. Dry-run inspects metadata and the planned transforms only; it deliberately defers full-data normalization statistics until the guarded write phase. Use `--input-chunks` and `--output-chunks` to control memory and I/O. The default output layout keeps one time step per chunk and complete channel/spatial dimensions, matching full-field weather-model reads without loading the whole time series.

cfgrib index files (`.idx`) are not written: all GRIB reads pass `indexpath=""`. The `merge_to_data` step combines data variables into a single `data` variable with a `level`/`channel` coordinate, matching the model library layout.

New CDS ERA5 downloads name the time coordinate `valid_time` instead of `time`; add a rename step (`{"rename": {"valid_time": "time"}}`) to normalize to the model library convention.

### compute_sidecars.py

Compute the per-channel `mean` / `std` / `weight` sidecars for a prepared Zarr store.

```text
python compute_sidecars.py --input out.zarr --output-dir <dir>
python compute_sidecars.py --input out.zarr --output-dir <dir> --chunks time=4 --allow-write [--overwrite]
```

Writes `mean.nc` / `std.nc` / `weight.nc` (1-D per-channel NetCDF arrays); the core reader prefers `.nc` over `.npy` over Zarr variables and reshapes them per channel. `weight` follows the core convention (level-scaled, optional land/ocean corrections via `--land-names` / `--ocean-names`, normalized to a maximum of 1).

Channel statistics are reduced as one lazy computation graph instead of separately materializing every channel's mean and standard deviation. The operation remains a full-data scan, but its memory use is bounded by the configured chunks.

Compute sidecars from the unit-converted, log-transformed data before normalization. A channel with zero variance is written as `std=0`; the core normalizer treats zero std as 1.

## Steps Config

JSON or YAML. Unknown steps abort the plan.

```json
{
  "steps": [
    {"rename": {"total_cloud_cover": "clt"}},
    {"keep_vars": ["z", "t", "q"]},
    {"time": {"start": "2023-06-01", "end": "2023-06-02"}},
    {"resample": {"freq": "6h", "operator": "mean"}},
    {"regrid": {"target": "s2s_1.5deg", "method": "linear", "variable_methods": {"lsm": "nearest"}}},
    {"merge_static": {"order": ["z", "lsm"], "coord": "channel", "name": "const"}},
    {"units": {"q": 1000, "tp": 1000, "ttr": 1 / 3600}},
    {"log1p": ["tp"]},
    {"split_levels": {"vars": ["z", "t", "u", "v", "q"], "level_coord": "pressure_level"}},
    {"merge_to_data": {"coord": "level", "order": ["z1000", "z925", "z850", "z700", "z600", "z500", "z400", "z300", "z250", "z200", "z150", "z100", "z50", "t1000", "t925", "t850", "t700", "t600", "t500", "t400", "t300", "t250", "t200", "t150", "t100", "t50", "u1000", "u925", "u850", "u700", "u600", "u500", "u400", "u300", "u250", "u200", "u150", "u100", "u50", "v1000", "v925", "v850", "v700", "v600", "v500", "v400", "v300", "v250", "v200", "v150", "v100", "v50", "q1000", "q925", "q850", "q700", "q600", "q500", "q400", "q300", "q250", "q200", "q150", "q100", "q50"]}},
    {"normalize": true}
  ]
}
```

`merge_to_data` concatenates the listed variables along the channel coordinate and renames the result to `data`. When `order` is omitted, channels present in the canonical `D:\cla.zarr` 76-channel contract are automatically ordered as 13 pressure levels from 1000 to 50 hPa for each of `z`, `t`, `u`, `v`, `q`, followed by `t2m,d2m,sst,ttr,10u,10v,100u,100v,msl,tcwv,tp`. A single channel or subset keeps its relative position in that contract; non-contract channels remain available and are appended in their original order. An explicit `order` remains authoritative. The result is transposed to `(time, level, lat, lon)` when a `time` dimension exists. `split_levels` expands a variable with a level dimension (for example CDS pressure levels delivered as `z/t/u/v/q` with a `pressure_level` dimension) into one variable per level, so it can feed `merge_to_data`. `units` multiplies variables by the given factors (for example `q` ×1000, `tp` ×1000, `ttr` ÷3600); `log1p` applies `log1p(clip(min=0))` to the listed variables (for example `tp`).

Precipitation unit convention: precipitation channels are normalized to **mm accumulated values** before `log1p`/`normalize`. ERA5 and ERA5-Land deliver `tp` in metres (step-accumulated), so the steps config multiplies it by 1000 (`"tp": 1000`); rate-form precipitation (`kg m-2 s-1`, `mm/h` averages, or `mm/s`) must first be multiplied by the accumulation length in seconds. The accumulation window must follow the selected model contract (for example daily totals for S2S, 6-hourly for IWC, hourly for ERA5-Land-based models) and must be recorded in the download plan so a given data source's original unit and window can be verified.

Output Zarr stores always use `lat`/`lon` as the grid coordinate names; CDS inputs carrying `latitude`/`longitude` are renamed automatically. This matches the core dataset classes, which access `ds.lat`/`ds.lon` directly (for example `GraphCastDataset` and the `MultiZarrDataset` bbox path).

`regrid` converts rectilinear ERA5 grids to a named model grid before channel merging and normalization. The `s2s_1.5deg` target is fixed to descending latitude `90, 88.5, ..., -90` (121 points) and longitude `0, 1.5, ..., 358.5` (240 points), matching the `cla.zarr` spatial contract. Source longitudes in `-180..180` are normalized to `0..360`; source latitude and longitude coordinates must be one-dimensional and unique. Standard ERA5 0.25-degree data contains every target point, so this aligned case uses exact lazy indexing without SciPy. Non-aligned source grids use xarray interpolation and require SciPy. `method` defaults to `linear` for continuous fields; use `variable_methods` with `nearest` for categorical fields such as land masks or soil type. Run this step before `merge_to_data` and `normalize`. Conservative remapping is not provided, so precipitation remapping must not be described as area-conservative.

`merge_static` is used only with `--output-format static-netcdf`. It selects static variables in the configured order, requires every `time` or `valid_time` dimension to contain exactly one value, removes that singleton dimension, and writes `const(channel, lat, lon)`. It rejects duplicate or missing variables, additional dimensions, an empty output, and normalization. Choose variable-specific regridding before this step (`linear` for continuous terrain/geopotential and `nearest` for categorical masks). The number and order of output channels must follow the selected model's `const_chans` contract; models with `const_chans=0` do not need this conversion.

`normalize` computes per-channel `mean`/`std` (and level-scaled `weight`, with optional land/ocean corrections via `{"normalize": {"land_names": [...], "ocean_names": [...]}}`) from the prepared data, stores `(x - mean) / std` values in the Zarr, and writes `mean.nc` / `std.nc` / `weight.nc` next to the output Zarr. Source NaN values are preserved (for example ERA5-Land non-land pixels); the core dataset classes replace them with 0 via `torch.nan_to_num`. The same weight convention (level-scaled plus optional land/ocean) is used by `compute_sidecars.py` and `merge_normalize.py`.

`flatten_step` merges the `time` and `step` dimensions of a GRIB-derived dataset into a single `time` axis using the `valid_time` coordinate, drops all-NaN combinations (for example ERA5-Land short-forecast GRIB frames outside the requested window), and reorders to `(time, level, lat, lon)`. Use it after `merge_to_data` for GRIB inputs that carry a `step` dimension.

When a model dataset is built from multiple converted Zarr stores (for example 65 pressure-level channels plus 11 single-level channels), merge and normalize them as one dataset with `merge_normalize.py`: it verifies time/lat/lon alignment, concatenates the channel dimension (channel count is determined by the inputs), defaults to the same `cla.zarr` relative channel order, computes per-channel statistics over the merged data, and writes the normalized Zarr plus sidecars. `--order` overrides the default but must list every input channel exactly once. Do not normalize each input store separately before merging.

`merge_normalize.py` remains lazy through the final `to_zarr` call and never loads the complete merged dataset into RAM. Its `--output-chunks` option defaults to one time step per chunk with full channel/spatial dimensions; tune the time chunk only after measuring the intended storage and training access pattern.

## Normalization Convention

Store normalized values in Zarr: precipitation-like channels such as `tp` are first converted to **mm accumulated values**, then log-transformed with `log1p` (clip min 0, consistent with the core reader's `clamp(0, 7)` in `data_util.unnormalize`); other channels are scaled with `(x - mean) / std` using per-channel statistics. The companion `mean`/`std`/`weight` sidecars record those statistics. Training feeds the Zarr directly (the model forward pass does not re-normalize); evaluation, export, and inference invert the sidecars back to physical values (`inv_normalize`). The core repository states this convention explicitly: "Training datasets are normalized Zarr stores with companion mean/std/weight.npy" and "the last channel is log-transformed precipitation".

## Training vs Inference Data Forms

- **Training** consumes a normalized Zarr store: values are stored as `(x - mean) / std` (with `tp` log-transformed first), and the model forward pass does not re-normalize.
- **Exported ONNX** (`export_onnx` in the current ViT/Afnonet/GraphCast/GroupVAE/Puyun models): normalization and inverse normalization are baked into the exported graph, so the ONNX model interface takes physical values and returns physical values. The core inference wrapper (`onnx_infer.py`) currently receives normalized inputs in its cascade entry and converts internally. Do not assume this pattern holds for every model: check the export/inference code of the model in use before preparing inference data.

## Directory Layout

Keep downloaded source data, conversion products, and sidecars in separate, predictable locations so the plan stays executable and inspection is unambiguous:

- Download staging: the raw files delivered by the provider (NetCDF, GRIB, ZIP archives) go to a staging directory, not into the final data directory.
- Converted product: one Zarr store per dataset, placed at the destination recorded in the download plan.
- Sidecars and static fields: `mean`, `std`, `weight`, `mask`, `land_mask`, `const` files live in the same directory as the Zarr store (`.nc`, `.npy`, or Zarr variables).
- Naming: use the dataset name from the download plan (for example `s2s.1950-2024.c76`); do not mix multiple test or comparison datasets in one folder.

## Boundary Rules

- Read-only inspection and dry runs come first; no script under this route writes without explicit approval.
- Zarr mutation requires `zarr_write_guard.py` and the user-approved flags.
- Never claim training readiness under this route; Data analysis owns that after data exists.
- Keep credentials out of steps configs, plans, logs, and repository files.
- When a dataset is not in the data source reference and the user cannot confirm its facts, record the fields as pending confirmation instead of guessing.

