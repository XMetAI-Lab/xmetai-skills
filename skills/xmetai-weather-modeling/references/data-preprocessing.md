# Data Preprocessing

Use this reference when downloaded data exists and needs to be parsed, normalized, converted, or validated before training. This route executes the format chain planned under Data download planning and stops before any training-readiness claim.

## Scope

This route answers: **What format is the downloaded data, how is it normalized, and is the converted result consistent with the selected config?**

It covers:

- Format detection: extension and magic bytes first, then metadata read.
- Declarative conversion steps: variable rename, variable selection, time selection, resampling.
- Guarded conversion to Zarr.
- Post-conversion validation: variables, units, and time continuity.

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
5. Run `preprocess_validate.py --path <output>` with expected variables, units, and frequency.
6. Report the conversion summary and the validation result. Do not report training readiness under this route.

## Format Support

| Input | Detection | Conversion | Status |
|---|---|---|---|
| NetCDF (classic CDF / NetCDF-4 HDF5) | magic `CDF` / HDF5 | xarray -> Zarr | supported |
| Zarr store | `.zgroup` / `zarr.json` | normalize -> Zarr | supported |
| GRIB | magic `GRIB` | cfgrib/eccodes (pip-installable on Windows) | supported when cfgrib installed; `decode-pending` otherwise |
| CINRAD `.bin` / NPZ | extension / magic | needs the core radar reader | pending |

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

The write path requires the Zarr write guard and explicit approval flags. Inputs: NetCDF (classic/NetCDF-4) or Zarr. Output: Zarr store only.

### preprocess_validate.py

Validate a prepared dataset without writing.

```text
python preprocess_validate.py --path out.zarr --config expected.json [--json]
python preprocess_validate.py --path out.zarr --variables z,t,q --freq 6h
```

Checks: expected variables present, units match, time monotonic and continuous at the expected frequency with the requested coverage. Exit code 0 when valid, 1 when invalid.

New CDS ERA5 downloads name the time coordinate `valid_time` instead of `time`. `preprocess_validate.py` accepts either name; to normalize a store to the model library convention, add a rename step (`{"rename": {"valid_time": "time"}}`) during conversion.

## Steps Config

JSON or YAML. Unknown steps abort the plan.

```json
{
  "steps": [
    {"rename": {"total_cloud_cover": "clt"}},
    {"keep_vars": ["z", "t", "q"]},
    {"time": {"start": "2023-06-01", "end": "2023-06-02"}},
    {"resample": {"freq": "6h", "operator": "mean"}}
  ]
}
```

## Validation Config

```json
{
  "variables": ["z", "t", "q"],
  "units": {"z": "m2 s-2", "t": "K"},
  "freq": "6h",
  "time": {"start": "2023-06-01", "end": "2023-06-03"}
}
```

## Boundary Rules

- Read-only inspection and dry runs come first; no script under this route writes without explicit approval.
- Zarr mutation requires `zarr_write_guard.py` and the user-approved flags.
- Never claim training readiness under this route; Data analysis owns that after data exists.
- Keep credentials out of steps configs, plans, logs, and repository files.
- When a dataset is not in the data source reference and the user cannot confirm its facts, record the fields as pending confirmation instead of guessing.
