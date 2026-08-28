#!/usr/bin/env python3
"""Convert meteorological NetCDF/Zarr input into a normalized Zarr store.

The script is dry-run by default: it prints the conversion plan (input
metadata, applied steps, target layout) without writing anything. To perform
the write, pass ``--allow-write`` and ``--ack-risk``; the Zarr write guard is
then executed before any mutation.

Usage:

    python convert_to_zarr.py --input era5_sample.nc --output out.zarr
    python convert_to_zarr.py --input-glob "era5_*.nc" --output out.zarr
    python convert_to_zarr.py --input era5_sample.nc --output out.zarr --steps-config steps.json
    python convert_to_zarr.py --input era5_sample.nc --output out.zarr --allow-write --ack-risk I-understand-this-mutates-zarr

Steps config (JSON or YAML):

    {
      "steps": [
        {"rename": {"total_cloud_cover": "clt"}},
        {"keep_vars": ["z", "t", "q"]},
        {"time": {"start": "2023-06-01", "end": "2023-06-02"}},
        {"resample": {"freq": "6h", "operator": "mean"}},
        {"daily_aggregation": {
            "window_hours": 24, "label_hour": 0, "incomplete": "error",
            "variables": {
                "tp": {"operator": "sum", "factor": 1000, "units": "mm"},
                "ttr": {"operator": "sum", "factor": 1.1574074074074073e-5,
                        "units": "W m-2"}
            }}},
        {"regrid": {"target": "s2s_1.5deg", "method": "linear",
                    "variable_methods": {"lsm": "nearest"}}},
        {"merge_static": {"order": ["z", "lsm"], "coord": "channel",
                          "name": "const"}},
        {"units": {"q": 1000}},
        {"log1p": ["tp"]},
        {"split_levels": {"vars": ["z", "t", "u", "v", "q"],
                          "level_coord": "pressure_level",
                          "levels": [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]}},
        {"merge_to_data": {"coord": "level", "order": ["z1000", "z925", "z850", "z700", "z600", "z500", "z400", "z300", "z250", "z200", "z150", "z100", "z50", "t1000", "t925", "t850", "t700", "t600", "t500", "t400", "t300", "t250", "t200", "t150", "t100", "t50", "u1000", "u925", "u850", "u700", "u600", "u500", "u400", "u300", "u250", "u200", "u150", "u100", "u50", "v1000", "v925", "v850", "v700", "v600", "v500", "v400", "v300", "v250", "v200", "v150", "v100", "v50", "q1000", "q925", "q850", "q700", "q600", "q500", "q400", "q300", "q250", "q200", "q150", "q100", "q50"]}},
        {"normalize": true}
      ]
    }

``merge_to_data`` combines data variables into a single ``data`` variable
with the given channel coordinate (``level`` or ``channel``), matching the
model library layout. ``split_levels`` expands a variable with a level
dimension (for example CDS pressure levels) into one variable per level
using ``name_template`` (default ``{var}{level}``), so it can feed
``merge_to_data``. ``units`` multiplies variables by the given factors and
``log1p`` applies ``log1p(clip(min=0))`` to the listed variables. GRIB inputs
that mix GRIB editions are read per variable and merged automatically;
cfgrib index files are not written (``indexpath=""``). ``normalize`` computes
per-channel mean/std and cosine-latitude spatial weights from the prepared data,
scales the stored values with ``(x - mean) / std``, and writes
``mean.nc`` / ``std.nc`` / ``weight.nc`` next to the output Zarr. Unknown
steps abort the plan. The script never writes without the guard.

Static fields use the same transforms, followed by ``merge_static`` and
``--output-format static-netcdf``.  Every time-like dimension on a static
variable must have length one; it is removed before writing a core-compatible
``const(channel, lat, lon)`` DataArray.

Precipitation unit convention: precipitation channels are normalized to
**mm accumulated values** before ``log1p``/``normalize``. ERA5 and ERA5-Land
deliver ``tp`` in metres (step-accumulated), so the steps config multiplies
it by 1000 (``"tp": 1000``). For hourly ERA5 accumulations feeding S2S, use
``daily_aggregation`` instead: it applies a configurable interval-ending
window, aggregation operator, factor, offset, and output unit per variable.
The example sums the 24 samples through 00 UTC, converts ``tp`` from metres
to mm/day, and converts summed ``ttr`` energy from J/m2 to daily-mean W/m2.
The same step can aggregate custom variables such as ``ssr``. Do not apply
another ``units`` conversion to those variables afterwards. Rate-form precipitation
(``kg m-2 s-1`` or ``mm/h`` averages) must first be multiplied by the
accumulation length in seconds. The accumulation window must follow the
selected model contract (for example daily totals for S2S, 6-hourly for IWC).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# The tests and some downstream tools load this file directly with
# importlib.util.spec_from_file_location, where the script directory is not
# automatically present on sys.path. Keep the sibling package discoverable in
# both direct CLI and import-by-path usage.
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

# Compatibility facade: implementations include cfgrib shortName/filter_by_keys
# decoding with indexpath disabled, plus split_levels, merge_to_data, log1p,
# normalize, flatten_step, and write_sidecars transforms used by existing callers.
from conversion.batching import (
    catalog_time_periods, input_file_batches, period_batches,
    prepared_file_batches, steps_for_file_batch,
)
from conversion.grid import (
    REGRID_METHODS, TARGET_GRIDS, _prepare_source_grid, _s2s_target_coords,
    canonicalize_latlon, regrid_dataset,
)
from conversion.inputs import (
    _open_grib_by_shortnames, format_label, is_zarr_dir, netcdf_engine_for,
    open_input, open_inputs, parse_chunks, resolve_inputs,
)
from conversion.state import (
    GUARD_ACK, _time_values, conversion_fingerprint, conversion_state_path,
    describe, run_guard, validate_resume_prefix, write_conversion_state,
)
from conversion.static import guard_static_output, static_dataarray
from conversion.statistics import (
    channel_moments, compute_channel_stats, finalize_channel_moments,
    latitude_weights, merge_channel_moments, normalize_ds,
    streaming_channel_stats,
)
from conversion.temporal import daily_aggregation, s2s_daily_accumulation
from conversion.transforms import (
    CLA_76_CHANNELS, CLA_76_CHANNEL_SET, LEVEL_RE, apply_steps,
    default_channel_order, load_steps_config,
)
from conversion.writers import (
    validate_append_contract, write_incremental_zarr, write_sidecars,
    write_streaming_file_batches,
)

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", action="append", default=[], help="input path; repeat for multiple NetCDF files")
    parser.add_argument("--input-glob", default=None, help="glob for a homogeneous NetCDF collection")
    parser.add_argument("--output", required=True, help="target Zarr store or static NetCDF path")
    parser.add_argument(
        "--output-format",
        choices=("zarr", "static-netcdf"),
        default="zarr",
        help="write the main Zarr store (default) or a core-compatible const.nc",
    )
    parser.add_argument("--steps-config", default=None, help="JSON/YAML steps config")
    parser.add_argument("--input-chunks", default="time=4", help="lazy input chunks, e.g. time=4")
    parser.add_argument(
        "--output-chunks",
        default="time=1,level=-1,channel=-1,lat=-1,lon=-1",
        help="output chunks; -1 means the complete dimension",
    )
    parser.add_argument(
        "--batch-time",
        type=int,
        default=31,
        help="number of output time steps per Zarr append batch (default: 31)",
    )
    parser.add_argument(
        "--input-period-batch",
        type=int,
        default=1,
        help="number of distinct source time periods opened per batch (default: 1)",
    )
    parser.add_argument(
        "--input-overlap-periods",
        type=int,
        default=1,
        help="neighboring source periods included for cross-boundary transforms (default: 1)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="continue an interrupted incremental write when existing times are an exact prefix",
    )
    parser.add_argument("--allow-write", action="store_true", help="confirm the write after guard approval")
    parser.add_argument("--overwrite", action="store_true", help="confirm replacing an existing output store")
    parser.add_argument("--ack-risk", default=None, help=f"must equal: {GUARD_ACK}")
    args = parser.parse_args(argv)

    input_paths = resolve_inputs(args.input, args.input_glob)
    input_path = input_paths[0]
    output_path = Path(args.output).expanduser()
    input_chunks = parse_chunks(args.input_chunks)
    output_chunks = parse_chunks(args.output_chunks)

    if args.resume and args.overwrite:
        raise SystemExit("--resume and --overwrite are mutually exclusive")
    if args.resume and args.output_format != "zarr":
        raise SystemExit("--resume is supported only for Zarr output")
    if args.batch_time <= 0:
        raise SystemExit("--batch-time must be a positive integer")
    if args.input_period_batch <= 0 or args.input_overlap_periods < 0:
        raise SystemExit("input period batch must be positive and overlap must be non-negative")

    steps = load_steps_config(args.steps_config)
    do_normalize = any("normalize" in step for step in steps)
    do_merge_static = any("merge_static" in step for step in steps)
    if args.output_format == "static-netcdf":
        if not do_merge_static:
            raise SystemExit("static-netcdf output requires a merge_static step")
        if do_normalize:
            raise SystemExit("static-netcdf output does not support normalize")
    elif do_merge_static:
        raise SystemExit("merge_static requires --output-format static-netcdf")
    stream_mode = (
        args.output_format == "zarr"
        and len(input_paths) > 1
        and all(format_label(path) == "netcdf" for path in input_paths)
    )
    try:
        if stream_mode:
            periods = catalog_time_periods(input_paths)
            preview_paths, _, _ = next(period_batches(periods, args.input_period_batch, args.input_overlap_periods))
            ds = open_inputs(preview_paths, input_chunks)
        else:
            ds = open_inputs(input_paths, input_chunks)
    except Exception as exc:
        if format_label(input_path) == "grib":
            raise SystemExit(
                f"GRIB decode failed: {type(exc).__name__}: {exc}\n"
                "Hint: multi-variable GRIB may mix GRIB editions; the automatic "
                "per-variable fallback also failed. Prefer NetCDF for multi-variable "
                "downloads or split the GRIB by variable "
                "(see references/data-preprocessing.md)."
            ) from exc
        raise
    try:
        before = describe(ds)
        ds = apply_steps(ds, steps_for_file_batch(steps) if stream_mode else steps)
        ds = canonicalize_latlon(ds)
        if args.output_format == "static-netcdf":
            static_dataarray(ds)
        after = describe(ds)
    finally:
        ds.close()

    print(f"inputs: {len(input_paths)} ({input_paths[0]}{f' ... {input_paths[-1]}' if len(input_paths) > 1 else ''})")
    print(f"format: {format_label(input_path)}")
    print(f"before: {before['dims']} vars={before['variables']}")
    if steps:
        print(f"steps : {len(steps)} (see config: {args.steps_config})")
    print(f"after : {after['dims']} vars={after['variables']}")
    print(f"output: {output_path}")
    print(f"output format: {args.output_format}")
    print(f"chunks: input={input_chunks or 'backend'} output={output_chunks or 'backend'}")
    if args.output_format == "zarr":
        print(f"incremental: batch_time={args.batch_time} resume={args.resume}")
    if stream_mode:
        print(
            f"file streaming: periods_per_batch={args.input_period_batch} "
            f"overlap_periods={args.input_overlap_periods}; preview only opened {len(preview_paths)} files"
        )
    if do_normalize:
        print("normalize: statistics deferred until the guarded write phase")

    if not args.allow_write:
        print("\nDRY-RUN: nothing was written.")
        print("To execute, pass --allow-write --ack-risk I-understand-this-mutates-zarr")
        return 0

    if args.ack_risk != GUARD_ACK:
        raise SystemExit(f"Refusing write: --ack-risk must equal {GUARD_ACK}")

    if args.output_format == "zarr":
        run_guard(input_paths, output_path, args.overwrite, args.resume)
    else:
        guard_static_output(input_paths, output_path, args.overwrite)

    execution = {
        "input_period_batch": args.input_period_batch,
        "input_overlap_periods": args.input_overlap_periods,
    }
    fingerprint = conversion_fingerprint(input_paths, steps, input_chunks, output_chunks, execution)
    state_path = conversion_state_path(output_path)
    if args.resume:
        if not state_path.exists():
            raise SystemExit(f"resume refused: conversion state is missing: {state_path}")
        previous = json.loads(state_path.read_text(encoding="utf-8"))
        if previous.get("contract_sha256") != fingerprint["contract_sha256"]:
            raise SystemExit("resume refused: transformation, chunk, or batching configuration changed")
        if do_normalize and previous.get("inputs") != fingerprint["inputs"]:
            raise SystemExit(
                "resume refused: normalized incremental output requires the identical input set "
                "because its global mean/std must not change"
            )
    state = {**fingerprint, "output": str(output_path.resolve())}

    if stream_mode:
        def batch_factory():
            return prepared_file_batches(
                input_paths,
                period_batch_size=args.input_period_batch,
                overlap_periods=args.input_overlap_periods,
                chunks=input_chunks,
                steps=steps,
            )

        normalization = None
        if do_normalize:
            normalization = streaming_channel_stats(batch_factory, state_path, state)
            names, mean, std, coord = normalization
            preview, _ = next(batch_factory())
            lat, weight = latitude_weights(preview)
        write_streaming_file_batches(
            batch_factory,
            output_path,
            batch_time=args.batch_time,
            resume=args.resume,
            overwrite=args.overwrite,
            output_chunks=output_chunks,
            state_path=state_path,
            state=state,
            normalization=normalization,
        )
        if do_normalize:
            write_sidecars(output_path, names, mean, std, lat, weight, coord or "channel")
            print(f"sidecars: mean.nc / std.nc / weight.nc -> {output_path}")
        print(f"\nWROTE {output_path}")
        return 0

    # Guard passed; reopen the single dataset for the actual write.
    ds = open_inputs(input_paths, input_chunks)
    try:
        ds = apply_steps(ds, steps)
        ds = canonicalize_latlon(ds)
        if do_normalize:
            names, mean, std, coord = compute_channel_stats(ds)
            lat, weight = latitude_weights(ds)
            ds = normalize_ds(ds, mean, std, coord, names)
        if output_chunks and args.output_format == "zarr":
            applicable_chunks = {name: size for name, size in output_chunks.items() if name in ds.dims}
            ds = ds.chunk(applicable_chunks)
        if args.output_format == "zarr":
            write_incremental_zarr(
                ds,
                output_path,
                batch_time=args.batch_time,
                resume=args.resume,
                overwrite=args.overwrite,
                state_path=state_path,
                state=state,
            )
            if do_normalize:
                write_sidecars(output_path, names, mean, std, lat, weight, coord or "channel")
                print(f"sidecars: mean.nc / std.nc / weight.nc -> {output_path}")
        else:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            static_dataarray(ds).to_netcdf(output_path, mode="w")
    finally:
        ds.close()
    print(f"\nWROTE {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
