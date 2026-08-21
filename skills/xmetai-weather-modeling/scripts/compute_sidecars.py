#!/usr/bin/env python3
"""Compute mean/std/weight sidecars for a prepared Zarr dataset.

Sidecars are the per-channel statistics consumed by the core model library:
``mean`` and ``std`` are computed per channel over the non-channel dims
(usually time, lat, lon); ``weight`` follows the core channel-weight
convention (level-scaled, optionally corrected by land/ocean groups,
normalized to a maximum of 1).

Compute sidecars from the unit-converted, log-transformed data *before*
normalization. Output files are 1-D per-channel NetCDF arrays named
``mean.nc`` / ``std.nc`` / ``weight.nc``; the core reader prefers ``.nc``
over ``.npy`` over Zarr variables.

Usage:

    python compute_sidecars.py --input out.zarr --output-dir <dir>
    python compute_sidecars.py --input out.zarr --output-dir <dir> --allow-write [--overwrite]

Read-only by default; writes sidecar files only with ``--allow-write``.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np

try:
    import xarray as xr
except ImportError:  # pragma: no cover - environment without parsing deps
    xr = None


LEVEL_RE = re.compile(r"^([A-Za-z]+)_?(\d+)$")


def open_zarr(path: Path, chunks: dict[str, int] | None = None):
    if xr is None:
        raise SystemExit("xarray is not installed")
    try:
        return xr.open_zarr(path, consolidated=True, chunks=chunks)
    except Exception:
        return xr.open_zarr(path, consolidated=False, chunks=chunks)


def parse_chunks(value: str | None) -> dict[str, int] | None:
    if not value:
        return None
    chunks: dict[str, int] = {}
    for item in value.split(","):
        try:
            name, raw_size = item.split("=", 1)
            size = int(raw_size)
        except ValueError as exc:
            raise SystemExit(f"invalid chunks specification: {value!r}") from exc
        name = name.strip()
        if not name or size == 0 or size < -1:
            raise SystemExit(f"invalid chunk entry: {item!r}")
        chunks[name] = size
    return chunks


def channel_names(ds, coord: str | None) -> tuple[list[str], str]:
    if coord is None:
        for candidate in ("level", "channel"):
            if candidate in ds.coords:
                return [str(v) for v in ds.coords[candidate].values], candidate
        return list(ds.data_vars), ""
    if coord not in ds.coords:
        raise SystemExit(f"channel coordinate {coord!r} not found in {list(ds.coords)}")
    return [str(v) for v in ds.coords[coord].values], coord


def per_channel_stats(ds, names: list[str], coord: str) -> tuple[np.ndarray, np.ndarray]:
    if coord:
        da = ds["data"]
        dims = [d for d in da.dims if d != coord]
        stats = xr.Dataset(
            {"mean": da.mean(dim=dims, skipna=True), "std": da.std(dim=dims, skipna=True)}
        ).compute()
        return (
            np.asarray(stats["mean"].values, dtype=np.float32),
            np.asarray(stats["std"].values, dtype=np.float32),
        )
    stats = xr.Dataset(
        {
            **{f"mean_{i}": ds[name].mean(skipna=True) for i, name in enumerate(names)},
            **{f"std_{i}": ds[name].std(skipna=True) for i, name in enumerate(names)},
        }
    ).compute()
    return (
        np.asarray([stats[f"mean_{i}"].item() for i in range(len(names))], dtype=np.float32),
        np.asarray([stats[f"std_{i}"].item() for i in range(len(names))], dtype=np.float32),
    )


def channel_weights(names: list[str], land: set[str], ocean: set[str]) -> np.ndarray:
    weights = np.ones(len(names), dtype=np.float32)
    for i, name in enumerate(names):
        match = LEVEL_RE.match(name)
        if match is not None:
            level = int(match.group(2))
            weights[i] = max(0.2, level / 1000.0)
    for i, name in enumerate(names):
        if name in land:
            weights[i] *= 0.33
        elif name in ocean:
            weights[i] *= 0.67
    weights /= weights.max()
    return weights


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", required=True, help="prepared Zarr store")
    parser.add_argument("--output-dir", required=True, help="directory for sidecar files")
    parser.add_argument("--channel-coord", default=None, help="level or channel (auto-detected)")
    parser.add_argument("--chunks", default="time=4", help="lazy read chunks, e.g. time=4")
    parser.add_argument("--land-names", default="", help="comma-separated land channel names")
    parser.add_argument("--ocean-names", default="", help="comma-separated ocean channel names")
    parser.add_argument("--allow-write", action="store_true", help="write sidecar files")
    parser.add_argument("--overwrite", action="store_true", help="replace existing sidecar files")
    args = parser.parse_args(argv)

    input_path = Path(args.input).expanduser()
    out_dir = Path(args.output_dir).expanduser()
    if not input_path.exists():
        raise SystemExit(f"input not found: {input_path}")

    ds = open_zarr(input_path, parse_chunks(args.chunks))
    try:
        names, coord = channel_names(ds, args.channel_coord)
        mean, std = per_channel_stats(ds, names, coord)
    finally:
        ds.close()
    land = {n.strip() for n in args.land_names.split(",") if n.strip()}
    ocean = {n.strip() for n in args.ocean_names.split(",") if n.strip()}
    weight = channel_weights(names, land, ocean)

    print(f"input   : {input_path}")
    print(f"channels: {len(names)} ({coord or 'data_vars'})")
    for name, m, s, w in zip(names, mean, std, weight):
        print(f"  {name:8s} mean={m:.4g} std={s:.4g} weight={w:.4g}")

    targets = {key: out_dir / f"{key}.nc" for key in ("mean", "std", "weight")}
    existing = [str(path) for path in targets.values() if path.exists()]
    if existing:
        print(f"existing: {existing}")
    if not args.allow_write:
        print("\nDRY-RUN: nothing was written.")
        print("To execute, pass --allow-write [--overwrite]")
        return 0
    if existing and not args.overwrite:
        raise SystemExit("Refusing to overwrite existing sidecars; pass --overwrite")

    out_dir.mkdir(parents=True, exist_ok=True)
    coord_name = coord or "channel"
    for key, values in (("mean", mean), ("std", std), ("weight", weight)):
        da = xr.DataArray(
            values,
            dims=[coord_name],
            coords={coord_name: names},
            attrs={"long_name": key, "channels": ",".join(names)},
        )
        path = targets[key]
        da.to_netcdf(path)
        print(f"WROTE {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
