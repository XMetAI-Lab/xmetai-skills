#!/usr/bin/env python3
"""Merge multiple converted Zarr stores and normalize them as one dataset.

Takes one or more prepared Zarr stores (each a single ``data`` variable with a
``level``/``channel`` coordinate, not yet normalized), verifies that their
time/lat/lon axes align, concatenates the channel dimension, computes
per-channel mean/std over the merged dataset, and writes a normalized Zarr
plus ``mean.nc`` / ``std.nc`` / ``weight.nc`` sidecars. The channel count is
determined by the inputs, so any combination of datasets can be merged without
hard-coding dimensions. Normalization uses the same channel-level statistics
as ``convert_to_zarr.py``, so the merged dataset is normalized as a whole,
not per input store.

Usage:

    python merge_normalize.py --stores a.zarr b.zarr --output merged.zarr
    python merge_normalize.py --stores a.zarr b.zarr --output merged.zarr --allow-write --ack-risk I-understand-this-mutates-zarr

Read-only by default; writes only with ``--allow-write`` and the Zarr write guard.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import xarray as xr

from convert_to_zarr import GUARD_ACK, canonicalize_latlon, channel_weights, compute_channel_stats, normalize_ds, write_sidecars


def open_store(path: Path):
    try:
        return xr.open_zarr(path, consolidated=True)
    except Exception:
        return xr.open_zarr(path, consolidated=False)


def channel_names_of(ds) -> tuple[list[str], str]:
    coord = next((c for c in ("level", "channel") if c in ds.coords), "")
    if not coord or "data" not in ds.data_vars:
        raise SystemExit(f"store {ds.encoding.get('source', '?')} is not a single-data channel store")
    return [str(v) for v in ds.coords[coord].values], coord


def run_guard(inputs: list[Path], output: Path, overwrite: bool) -> None:
    guard = Path(__file__).resolve().parent / "zarr_write_guard.py"
    cmd = [
        sys.executable,
        str(guard),
        "--operation",
        "normalize",
        "--allow-write",
        "--ack-risk",
        GUARD_ACK,
        "--output",
        str(output),
    ]
    for p in inputs:
        cmd += ["--input", str(p)]
    if overwrite:
        cmd.append("--overwrite")
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        detail = (exc.stdout or "") + (exc.stderr or "")
        raise SystemExit(f"Zarr write guard refused merge-normalize:\n{detail}") from exc


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--stores", nargs="+", required=True, help="input Zarr stores (each single-data + channel coord)")
    parser.add_argument("--output", required=True, help="target normalized Zarr store path")
    parser.add_argument("--order", default=None, help="comma-separated target channel order (default: input order)")
    parser.add_argument("--land-names", default="", help="comma-separated land channel names for weight correction")
    parser.add_argument("--ocean-names", default="", help="comma-separated ocean channel names for weight correction")
    parser.add_argument("--allow-write", action="store_true", help="confirm the write after guard approval")
    parser.add_argument("--overwrite", action="store_true", help="confirm replacing an existing output store")
    parser.add_argument("--ack-risk", default=None, help=f"must equal: {GUARD_ACK}")
    args = parser.parse_args(argv)

    stores = [Path(s).expanduser() for s in args.stores]
    output = Path(args.output).expanduser()
    for p in stores:
        if not p.exists():
            raise SystemExit(f"store not found: {p}")

    datasets = [canonicalize_latlon(open_store(p)) for p in stores]
    try:
        ref_time = datasets[0]["time"].values
        ref_lat = datasets[0]["lat"].values
        ref_lon = datasets[0]["lon"].values
        for ds in datasets[1:]:
            if not np.array_equal(ds["time"].values, ref_time):
                raise SystemExit("stores have misaligned time axes")
            if not np.array_equal(ds["lat"].values, ref_lat):
                raise SystemExit("stores have misaligned latitude axes")
            if not np.array_equal(ds["lon"].values, ref_lon):
                raise SystemExit("stores have misaligned longitude axes")

        names, coord = [], None
        arrays = []
        for ds in datasets:
            ch_names, ch_coord = channel_names_of(ds)
            if coord is None:
                coord = ch_coord
            elif ch_coord != coord:
                raise SystemExit(f"channel coordinate mismatch: {ch_coord!r} vs {coord!r}")
            names.extend(ch_names)
            arrays.append(ds["data"])

        if args.order:
            order = [n.strip() for n in args.order.split(",") if n.strip()]
            missing = [n for n in order if n not in names]
            if missing:
                raise SystemExit(f"order lists unknown channels: {missing}")
            names = order

        merged = xr.concat(arrays, dim=coord)
        merged = merged.assign_coords({coord: names})
        dims = list(merged.dims)
        if "time" in dims:
            dims.remove("time")
            dims.insert(0, "time")
        if coord in dims:
            dims.remove(coord)
            dims.insert(1, coord)
        merged = merged.transpose(*dims).to_dataset(name="data")

        mean, std = None, None
        if args.allow_write:
            stats_names, mean, std, stats_coord = compute_channel_stats(merged)
            if list(stats_names) != names:
                raise SystemExit("internal channel order mismatch during statistics")
            merged = normalize_ds(merged, mean, std, stats_coord, names)

        print(f"merged channels: {len(names)} ({coord})")
        print("merged dims:", dict(merged.sizes))
        print("first channels:", names[:3], "... last:", names[-3:])
        if not args.allow_write:
            print("\nDRY-RUN: statistics and normalization will run on the merged dataset.")
            print("To execute, pass --allow-write --ack-risk I-understand-this-mutates-zarr")
            return 0

        if args.ack_risk != GUARD_ACK:
            raise SystemExit(f"Refusing write: --ack-risk must equal {GUARD_ACK}")
        run_guard(stores, output, args.overwrite)

        land = {n.strip() for n in args.land_names.split(",") if n.strip()}
        ocean = {n.strip() for n in args.ocean_names.split(",") if n.strip()}
        weight = channel_weights(names, land, ocean)
        write_sidecars(output, names, mean, std, weight, coord)
        merged = merged.load()  # force in-memory arrays to avoid dask multi-thread zarr writes on Windows
        merged["data"].encoding.pop("chunks", None)
        merged["data"].encoding["chunks"] = tuple(merged.sizes[d] for d in merged["data"].dims)
        merged.to_zarr(str(output), mode="w" if args.overwrite else "w-", consolidated=True, safe_chunks=False)
        print(f"sidecars: mean.nc / std.nc / weight.nc -> {output}")
        print(f"WROTE {output}")
        return 0
    finally:
        for ds in datasets:
            ds.close()


if __name__ == "__main__":
    sys.exit(main())
