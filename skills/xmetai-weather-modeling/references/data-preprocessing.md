# Data Preprocessing

Use this reference when downloaded data exists and needs to be parsed, normalized, converted, or validated before training. This route executes the format chain planned under Data download planning and stops before any training-readiness claim.

## Scope

This route answers: **What format is the downloaded data, how is it normalized, and is the converted result consistent with the selected config?**

It covers:

- Format detection: extension and magic bytes first, then metadata read.
- Declarative conversion steps: variable rename, variable selection, time selection, resampling, channel merge.
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
python convert_to_zarr.py --input in.nc --output out.zarr --allow-write --ack-risk I-understand-this-mutates-zarr [--overwrite]
```

The write path requires the Zarr write guard and explicit approval flags. Inputs: NetCDF (classic/NetCDF-4), Zarr, or GRIB (multi-variable mixed-edition GRIB is read per variable automatically). Output: Zarr store only.

cfgrib index files (`.idx`) are not written: all GRIB reads pass `indexpath=""`. The `merge_to_data` step combines data variables into a single `data` variable with a `level`/`channel` coordinate, matching the model library layout.

New CDS ERA5 downloads name the time coordinate `valid_time` instead of `time`; add a rename step (`{"rename": {"valid_time": "time"}}`) to normalize to the model library convention.

### compute_sidecars.py

Compute the per-channel `mean` / `std` / `weight` sidecars for a prepared Zarr store.

```text
python compute_sidecars.py --input out.zarr --output-dir <dir>
python compute_sidecars.py --input out.zarr --output-dir <dir> --allow-write [--overwrite]
```

Writes `mean.nc` / `std.nc` / `weight.nc` (1-D per-channel NetCDF arrays); the core reader prefers `.nc` over `.npy` over Zarr variables and reshapes them per channel. `weight` follows the core convention (level-scaled, optional land/ocean corrections via `--land-names` / `--ocean-names`, normalized to a maximum of 1).

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
    {"units": {"q": 1000, "ttr": 1 / 3600}},
    {"log1p": ["tp"]},
    {"split_levels": {"vars": ["z", "t", "u", "v", "q"], "level_coord": "pressure_level"}},
    {"merge_to_data": {"coord": "level", "order": ["z500", "t850", "q925"]}},
    {"normalize": true}
  ]
}
```

`merge_to_data` concatenates the listed variables along the channel coordinate and renames the result to `data`; omit `order` to use the current variable order. The result is transposed to `(time, level, lat, lon)` when a `time` dimension exists. `split_levels` expands a variable with a level dimension (for example CDS pressure levels delivered as `z/t/u/v/q` with a `pressure_level` dimension) into one variable per level, so it can feed `merge_to_data`. `units` multiplies variables by the given factors (for example `q` ×1000, `ttr` ÷3600); `log1p` applies `log1p(clip(min=0))` to the listed variables (for example `tp`).

`normalize` computes per-channel `mean`/`std` (and level-scaled `weight`, with optional land/ocean corrections via `{"normalize": {"land_names": [...], "ocean_names": [...]}}`) from the prepared data, stores `(x - mean) / std` values in the Zarr, and writes `mean.nc` / `std.nc` / `weight.nc` next to the output Zarr. Source NaN values are preserved (for example ERA5-Land non-land pixels); the core dataset classes replace them with 0 via `torch.nan_to_num`. The same weight convention (level-scaled plus optional land/ocean) is used by `compute_sidecars.py` and `merge_normalize.py`.

`flatten_step` merges the `time` and `step` dimensions of a GRIB-derived dataset into a single `time` axis using the `valid_time` coordinate, drops all-NaN combinations (for example ERA5-Land short-forecast GRIB frames outside the requested window), and reorders to `(time, level, lat, lon)`. Use it after `merge_to_data` for GRIB inputs that carry a `step` dimension.

When a model dataset is built from multiple converted Zarr stores (for example 65 pressure-level channels plus 11 single-level channels), merge and normalize them as one dataset with `merge_normalize.py`: it verifies time/lat/lon alignment, concatenates the channel dimension (channel count is determined by the inputs), computes per-channel statistics over the merged data, and writes the normalized Zarr plus sidecars. Do not normalize each input store separately before merging.

## Normalization Convention

Store normalized values in Zarr: precipitation-like channels such as `tp` are log-transformed with `log1p`, other channels are scaled with `(x - mean) / std` using per-channel statistics. The companion `mean`/`std`/`weight` sidecars record those statistics. Training feeds the Zarr directly (the model forward pass does not re-normalize); evaluation, export, and inference invert the sidecars back to physical values (`inv_normalize`). The core repository states this convention explicitly: "Training datasets are normalized Zarr stores with companion mean/std/weight.npy" and "the last channel is log-transformed precipitation".

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
