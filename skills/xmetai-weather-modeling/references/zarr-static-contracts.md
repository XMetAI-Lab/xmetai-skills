# Data Analysis

Use this reference for Zarr, NetCDF, static data, schema inspection, data safety, statistics, masks, and variable distribution decisions.

## Safety

Read-only is normally OK: metadata, dims/coords/vars/chunks/time, tiny samples, stats files, static metadata, reports outside stores.

Ask first for any Zarr mutation: `to_zarr`, `save_zarr`, `mode="w"`, `mode="a"`, `append_dim`, `region`, overwrite, delete, in-place merge, in-place rechunk, or training stats/static edits.

Approval request must name inputs, output, operation type, output existence, and whether any input is modified.

Write-script guard:

- Require `--input`, `--output`, `--allow-write`, `--ack-risk`.
- Reject output equal to input unless `--allow-in-place` is explicitly approved.
- Default to dry-run when possible.
- Use `scripts/zarr_write_guard.py`.

## Zarr Schema

Zarr checks:

- Opens consolidated or non-consolidated.
- Expected variable exists, usually `data`.
- Dims match config, usually `time`, `level|channel`, `lat`, `lon`.
- Grid coordinates are named `lat`/`lon`; converted stores rename CDS `latitude`/`longitude` automatically.
- Time is sorted and matches `freq`.
- Channel count matches `in_chans`/`out_chans`.
- `mean`/`std`/`weight` shapes broadcast correctly.
- Precip or accumulated transforms match train and inference.

## Static NetCDF

Static NetCDF checks:

- Names, shapes, dtypes, NaN/Inf, coordinates.
- Same grid as model input.
- Continuous and categorical normalization is intentional.
- Categorical representation is documented.
- Channel order is preserved when checkpoint/deploy depends on it.

## Variable Distribution

- Treat data distribution as a design input. Each meteorological variable may need a different target transform, loss weighting, clipping policy, mask policy, or probabilistic head.
- Continuous near-Gaussian fields: verify normalization statistics, bias-sensitive metrics, and smoothness penalties only if physically justified.
- Heavy-tailed or sparse fields such as precipitation: consider occurrence/intensity decomposition, log or power transforms, thresholds, CRPS/quantile/objective choices, and rare-event weighting.
- Bounded fields: enforce or validate bounds after inverse transform; avoid losses that reward out-of-range outputs.
- Categorical or land/sea/static-derived targets: keep masks and class semantics explicit; do not treat them as ordinary continuous channels without a reason.
- Coupled variables: check whether independent channel losses violate known physical relationships, and document any multi-variable constraints.

Read-only scripts: `inspect_zarr_schema.py`, `inspect_static_nc.py`.
