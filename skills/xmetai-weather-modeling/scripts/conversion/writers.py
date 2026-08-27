"""Incremental Zarr writes and normalization sidecars."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

try:
    import xarray as xr
except ImportError:  # pragma: no cover
    xr = None

from .state import _time_values, validate_resume_prefix, write_conversion_state
from .statistics import normalize_ds


def write_sidecars(out_dir, names, mean, std, lat, weight, coord_name):
    """Write channel mean/std and a separate latitude-weight sidecar."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dim = coord_name or "channel"
    for key, values in (("mean", mean), ("std", std)):
        da = xr.DataArray(values, dims=[dim], coords={dim: names}, attrs={"long_name": key})
        da.to_netcdf(out_dir / f"{key}.nc")
    xr.DataArray(
        weight,
        dims=["lat"],
        coords={"lat": lat},
        attrs={"long_name": "cosine latitude area weight", "formula": "cos(abs(lat))"},
    ).to_netcdf(out_dir / "weight.nc")


def write_incremental_zarr(
    ds,
    output_path: Path,
    *,
    batch_time: int,
    resume: bool,
    overwrite: bool = False,
    state_path: Path,
    state: dict[str, Any],
) -> None:
    """Write time batches, deriving recovery progress from the store itself."""
    if batch_time <= 0:
        raise SystemExit("--batch-time must be a positive integer")
    total = _time_values(ds).size
    start = 0
    if resume:
        if output_path.exists():
            try:
                existing = xr.open_zarr(output_path, consolidated=False)
                try:
                    start = validate_resume_prefix(ds, existing)
                finally:
                    existing.close()
            except SystemExit:
                raise
            except Exception as exc:
                raise SystemExit(f"resume refused: existing Zarr cannot be opened safely: {exc}") from exc
        else:
            # A guarded run may fail during statistics before the store exists.
            # In that state resume means recompute statistics and create the store.
            start = 0
    elif output_path.exists() and not overwrite:
        raise SystemExit(f"output already exists: {output_path}; use --resume or approved --overwrite")

    state.update({"status": "running", "total_time_steps": int(total), "completed_time_steps": start})
    write_conversion_state(state_path, state)
    try:
        for offset in range(start, total, batch_time):
            stop = min(offset + batch_time, total)
            batch = ds.isel(time=slice(offset, stop))
            if offset == 0:
                batch.to_zarr(str(output_path), mode="w" if overwrite else "w-", consolidated=False)
            else:
                batch.to_zarr(str(output_path), mode="a", append_dim="time", consolidated=False)
            state["completed_time_steps"] = stop
            state["last_time"] = str(np.asarray(ds.time.values)[stop - 1])
            write_conversion_state(state_path, state)
            print(f"batch: {offset}:{stop} / {total}")
        try:
            import zarr

            zarr.consolidate_metadata(str(output_path))
        except Exception as exc:
            raise SystemExit(f"data batches were written but metadata consolidation failed: {exc}") from exc
    except BaseException as exc:
        state["status"] = "failed"
        state["error"] = f"{type(exc).__name__}: {exc}"
        write_conversion_state(state_path, state)
        raise
    state.update({"status": "completed", "completed_time_steps": int(total)})
    state.pop("error", None)
    write_conversion_state(state_path, state)


def validate_append_contract(planned, existing) -> None:
    """Validate non-time schema before appending a prepared file batch."""
    if set(existing.data_vars) != set(planned.data_vars):
        raise SystemExit("resume refused: existing and planned data variables differ")
    for name in existing.data_vars:
        if existing[name].dims != planned[name].dims:
            raise SystemExit(f"resume refused: dimensions differ for variable {name!r}")
        for dim in existing[name].dims:
            if dim != "time" and existing.sizes[dim] != planned.sizes[dim]:
                raise SystemExit(f"resume refused: dimension {dim!r} differs for variable {name!r}")
    for coord in set(existing.coords) & set(planned.coords):
        if "time" not in existing[coord].dims and not np.array_equal(
            existing[coord].values, planned[coord].values
        ):
            raise SystemExit(f"resume refused: coordinate {coord!r} differs")


def write_streaming_file_batches(
    batch_factory,
    output_path: Path,
    *,
    batch_time: int,
    resume: bool,
    overwrite: bool,
    output_chunks,
    state_path: Path,
    state: dict[str, Any],
    normalization=None,
) -> None:
    """Append transformed file batches while verifying the complete resume prefix."""
    existing = None
    existing_times = np.asarray([], dtype="datetime64[ns]")
    cursor = 0
    if resume and output_path.exists():
        try:
            existing = xr.open_zarr(output_path, consolidated=False)
            existing_times = _time_values(existing)
        except Exception as exc:
            state.update({"status": "failed", "phase": "write", "error": f"{type(exc).__name__}: {exc}"})
            write_conversion_state(state_path, state)
            raise SystemExit(f"resume refused: existing Zarr cannot be opened safely: {exc}") from exc
    created = existing is not None
    completed = int(existing_times.size)
    state.update({"status": "running", "phase": "write", "completed_time_steps": completed})
    write_conversion_state(state_path, state)
    try:
        for prepared, open_count in batch_factory():
            if normalization is not None:
                names, mean, std, coord = normalization
                prepared = normalize_ds(prepared, mean, std, coord, names)
            if output_chunks:
                applicable = {name: size for name, size in output_chunks.items() if name in prepared.dims}
                prepared = prepared.chunk(applicable)
            times = _time_values(prepared)
            if existing is not None:
                validate_append_contract(prepared, existing)
            prefix_count = min(max(existing_times.size - cursor, 0), times.size)
            if prefix_count:
                if not np.array_equal(times[:prefix_count], existing_times[cursor : cursor + prefix_count]):
                    raise SystemExit(
                        "resume refused: existing Zarr times are not the exact prefix produced by file batches"
                    )
                cursor += prefix_count
                prepared = prepared.isel(time=slice(prefix_count, None))
            if prepared.sizes.get("time", 0) == 0:
                continue
            if cursor < existing_times.size:
                raise SystemExit("resume refused: planned file batches skip existing output times")
            for offset in range(0, prepared.sizes["time"], batch_time):
                stop = min(offset + batch_time, prepared.sizes["time"])
                piece = prepared.isel(time=slice(offset, stop))
                if not created:
                    piece.to_zarr(
                        str(output_path), mode="w" if overwrite else "w-", consolidated=False
                    )
                    created = True
                else:
                    piece.to_zarr(str(output_path), mode="a", append_dim="time", consolidated=False)
                completed += piece.sizes["time"]
                state.update(
                    {
                        "completed_time_steps": completed,
                        "last_time": str(np.asarray(piece.time.values)[-1]),
                        "max_open_input_files": max(state.get("max_open_input_files", 0), open_count),
                    }
                )
                write_conversion_state(state_path, state)
                print(f"stream batch: +{piece.sizes['time']} -> {completed} (input files open: {open_count})")
        if resume and cursor != existing_times.size:
            raise SystemExit("resume refused: planned input ended before the existing Zarr prefix")
        if not created:
            raise SystemExit("streaming conversion produced no output time steps")
        import zarr

        zarr.consolidate_metadata(str(output_path))
    except BaseException as exc:
        state.update({"status": "failed", "error": f"{type(exc).__name__}: {exc}"})
        write_conversion_state(state_path, state)
        raise
    finally:
        if existing is not None:
            existing.close()
    state.update({"status": "completed", "phase": "complete", "completed_time_steps": completed})
    state.pop("error", None)
    write_conversion_state(state_path, state)
