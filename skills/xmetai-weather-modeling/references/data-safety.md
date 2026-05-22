# Data Safety

Read-only is normally OK: metadata, dims/coords/vars/chunks/time, tiny samples, stats files, static metadata, reports outside stores.

Ask first for any Zarr mutation: `to_zarr`, `save_zarr`, `mode="w"`, `mode="a"`, `append_dim`, `region`, overwrite, delete, in-place merge, in-place rechunk, or training stats/static edits.

Approval request must name inputs, output, operation type, output existence, and whether any input is modified.

Write-script guard:

- Require `--input`, `--output`, `--allow-write`, `--ack-risk`.
- Reject output equal to input unless `--allow-in-place` is explicitly approved.
- Default to dry-run when possible.
- Use `scripts/zarr_write_guard.py`.
