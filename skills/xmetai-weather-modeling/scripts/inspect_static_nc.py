#!/usr/bin/env python3
"""Read-only inspection for static meteorological NetCDF files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def import_xarray():
    try:
        import xarray as xr
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise SystemExit("xarray is required for this script.") from exc
    return xr


def jsonable(value: Any) -> Any:
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return str(value)


def inspect(path: Path, sample: bool) -> dict[str, Any]:
    if not path.exists():
        raise SystemExit(f"Input file does not exist: {path}")
    xr = import_xarray()
    ds = xr.open_dataset(path)
    report: dict[str, Any] = {
        "path": str(path),
        "dims": dict(ds.sizes),
        "data_vars": sorted(ds.data_vars),
        "coords": sorted(ds.coords),
        "variables": {},
    }
    for name, da in ds.data_vars.items():
        info: dict[str, Any] = {
            "dims": list(da.dims),
            "shape": list(da.shape),
            "dtype": str(da.dtype),
            "attrs": {k: jsonable(v) for k, v in da.attrs.items()},
        }
        if sample:
            try:
                arr = da.astype("float64")
                info["min"] = jsonable(arr.min(skipna=True).values)
                info["max"] = jsonable(arr.max(skipna=True).values)
                info["mean"] = jsonable(arr.mean(skipna=True).values)
            except Exception as exc:
                info["sample_error"] = str(exc)
        report["variables"][name] = info
    return report


def print_text(report: dict[str, Any]) -> None:
    print(f"Path: {report['path']}")
    print(f"Dims: {report['dims']}")
    print(f"Coords: {', '.join(report['coords'])}")
    print("\nVariables:")
    for name, info in report["variables"].items():
        print(f"  {name}: dims={info['dims']} shape={info['shape']} dtype={info['dtype']}")
        for key in ("min", "max", "mean", "sample_error"):
            if key in info:
                print(f"    {key}={info[key]}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", "-i", type=Path, required=True, help="Path to NetCDF file.")
    parser.add_argument("--sample", action="store_true", help="Compute simple statistics for each variable.")
    parser.add_argument("--json", action="store_true", help="Print JSON.")
    args = parser.parse_args()

    report = inspect(args.input, args.sample)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print_text(report)


if __name__ == "__main__":
    main()
