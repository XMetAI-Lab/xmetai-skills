# Zarr And Static Contracts

Zarr checks:

- Opens consolidated or non-consolidated.
- Expected variable exists, usually `data`.
- Dims match config, usually `time`, `level|channel`, `lat`, `lon`.
- Time is sorted and matches `freq`.
- Channel count matches `in_chans`/`out_chans`.
- `mean`/`std`/`weight` shapes broadcast correctly.
- Precip or accumulated transforms match train and inference.

Static NetCDF checks:

- Names, shapes, dtypes, NaN/Inf, coordinates.
- Same grid as model input.
- Continuous and categorical normalization is intentional.
- Categorical representation is documented.
- Channel order is preserved when checkpoint/deploy depends on it.

Read-only scripts: `inspect_zarr_schema.py`, `inspect_static_nc.py`.
