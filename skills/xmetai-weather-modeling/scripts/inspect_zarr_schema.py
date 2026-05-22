#!/usr/bin/env python3
"""Read-only schema and light quality inspection for a Zarr dataset."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def import_xarray():
    try:
        import xarray as xr
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise SystemExit("xarray is required for this script.") from exc
    return xr


def open_zarr(path: Path, chunks: str | None):
    xr = import_xarray()
    chunk_arg: Any = None if chunks == "none" else chunks
    try:
        return xr.open_zarr(path, consolidated=True, chunks=chunk_arg)
    except Exception as first_exc:
        try:
            return xr.open_zarr(path, consolidated=False, chunks=chunk_arg)
        except Exception as second_exc:
            raise SystemExit(f"Failed to open Zarr store. consolidated=True: {first_exc}; consolidated=False: {second_exc}") from second_exc


def to_jsonable(value: Any) -> Any:
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)


def summarize_array(da: Any, sample: bool) -> dict[str, Any]:
    info: dict[str, Any] = {
        "dims": list(da.dims),
        "shape": list(da.shape),
        "dtype": str(da.dtype),
    }
    chunks = getattr(getattr(da, "data", None), "chunks", None)
    if chunks is not None:
        info["chunks"] = [list(c) for c in chunks]

    if sample:
        indexer = {dim: 0 for dim in da.dims if da.sizes.get(dim, 0) > 0}
        sampled = da.isel(indexer) if indexer else da
        try:
            finite = sampled.astype("float64")
            info["sample_min"] = to_jsonable(finite.min(skipna=True).values)
            info["sample_max"] = to_jsonable(finite.max(skipna=True).values)
            info["sample_mean"] = to_jsonable(finite.mean(skipna=True).values)
        except Exception as exc:
            info["sample_error"] = str(exc)
    return info


def inspect(path: Path, var: str | None, sample: bool) -> dict[str, Any]:
    if not path.exists():
        raise SystemExit(f"Input path does not exist: {path}")
    ds = open_zarr(path, chunks="auto")
    report: dict[str, Any] = {
        "path": str(path),
        "dims": dict(ds.sizes),
        "data_vars": sorted(ds.data_vars),
        "coords": sorted(ds.coords),
        "attrs": {k: to_jsonable(v) for k, v in ds.attrs.items()},
        "variables": {},
    }

    for name, da in ds.data_vars.items():
        if var and name != var:
            continue
        report["variables"][name] = summarize_array(da, sample=sample)

    for coord_name in ("time", "level", "channel", "lat", "lon"):
        if coord_name in ds.coords:
            coord = ds.coords[coord_name]
            coord_info: dict[str, Any] = {"size": int(coord.size), "dtype": str(coord.dtype)}
            if coord.size:
                coord_info["first"] = to_jsonable(coord.values[0])
                coord_info["last"] = to_jsonable(coord.values[-1])
            report.setdefault("coordinate_summary", {})[coord_name] = coord_info

    if "time" in ds.coords and ds.coords["time"].size > 2:
        try:
            import pandas as pd

            times = pd.DatetimeIndex(ds.coords["time"].values)
            deltas = times.to_series().diff().dropna()
            if not deltas.empty:
                mode_delta = deltas.mode().iloc[0]
                report.setdefault("coordinate_summary", {})["time"]["most_common_delta"] = str(mode_delta)
        except Exception as exc:
            report.setdefault("warnings", []).append(f"Could not infer time frequency: {exc}")

    return report


def print_text(report: dict[str, Any]) -> None:
    print(f"Path: {report['path']}")
    print(f"Dims: {report['dims']}")
    print(f"Data vars: {', '.join(report['data_vars'])}")
    print(f"Coords: {', '.join(report['coords'])}")
    if "coordinate_summary" in report:
        print("\nCoordinates:")
        for name, info in report["coordinate_summary"].items():
            print(f"  {name}: {info}")
    print("\nVariables:")
    for name, info in report["variables"].items():
        print(f"  {name}: dims={info['dims']} shape={info['shape']} dtype={info['dtype']}")
        chunks = info.get("chunks")
        if chunks:
            print(f"    chunks={chunks}")
        sample_keys = [k for k in ("sample_min", "sample_max", "sample_mean", "sample_error") if k in info]
        for key in sample_keys:
            print(f"    {key}={info[key]}")
    for warning in report.get("warnings", []):
        print(f"Warning: {warning}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", "-i", type=Path, required=True, help="Path to Zarr store.")
    parser.add_argument("--var", "-v", help="Optional data variable to inspect.")
    parser.add_argument("--sample", action="store_true", help="Compute a tiny first-index sample summary.")
    parser.add_argument("--json", action="store_true", help="Print JSON.")
    args = parser.parse_args()

    report = inspect(args.input, args.var, args.sample)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print_text(report)


if __name__ == "__main__":
    main()
