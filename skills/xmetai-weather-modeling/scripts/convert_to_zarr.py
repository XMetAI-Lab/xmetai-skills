#!/usr/bin/env python3
"""Convert meteorological NetCDF/Zarr input into a normalized Zarr store.

The script is dry-run by default: it prints the conversion plan (input
metadata, applied steps, target layout) without writing anything. To perform
the write, pass ``--allow-write`` and ``--ack-risk``; the Zarr write guard is
then executed before any mutation.

Usage:

    python convert_to_zarr.py --input era5_sample.nc --output out.zarr
    python convert_to_zarr.py --input era5_sample.nc --output out.zarr --steps-config steps.json
    python convert_to_zarr.py --input era5_sample.nc --output out.zarr --allow-write --ack-risk I-understand-this-mutates-zarr

Steps config (JSON or YAML):

    {
      "steps": [
        {"rename": {"total_cloud_cover": "clt"}},
        {"keep_vars": ["z", "t", "q"]},
        {"time": {"start": "2023-06-01", "end": "2023-06-02"}},
        {"resample": {"freq": "6h", "operator": "mean"}}
      ]
    }

Unknown steps abort the plan. The script never writes without the guard.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

try:
    import xarray as xr
except ImportError:  # pragma: no cover - environment without parsing deps
    xr = None

GUARD_ACK = "I-understand-this-mutates-zarr"


def is_zarr_dir(path: Path) -> bool:
    return (path / ".zgroup").is_file() or (path / "zarr.json").is_file()


def format_label(path: Path) -> str:
    if is_zarr_dir(path):
        return "zarr"
    if path.suffix.lower() in (".grib", ".grb", ".grib1", ".grib2"):
        return "grib"
    return "netcdf"


def open_input(path: Path):
    if xr is None:
        raise SystemExit("xarray is not installed")
    if is_zarr_dir(path):
        return xr.open_zarr(path)
    return xr.open_dataset(path)


def load_steps_config(config: str | None) -> list[dict[str, Any]]:
    if config is None:
        return []
    path = Path(config).expanduser()
    if not path.is_file():
        raise SystemExit(f"steps config not found: {config}")
    text = path.read_text(encoding="utf-8-sig")
    if path.suffix.lower() in (".yaml", ".yml"):
        try:
            import yaml
        except ImportError as exc:
            raise SystemExit("PyYAML is required for .yaml config; use JSON instead") from exc
        payload = yaml.safe_load(text)
    else:
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"steps config must be JSON or YAML: {exc}") from exc
    steps = payload.get("steps")
    if steps is None:
        return []
    if not isinstance(steps, list):
        raise SystemExit("'steps' in config must be a list")
    return steps


def apply_steps(ds, steps: list[dict[str, Any]]):
    """Apply declarative steps to an in-memory dataset (no writes)."""
    for step in steps:
        if "rename" in step:
            ds = ds.rename(step["rename"])
        elif "keep_vars" in step:
            ds = ds[step["keep_vars"]]
        elif "time" in step:
            t = step["time"]
            ds = ds.sel(time=slice(t.get("start"), t.get("end")))
        elif "resample" in step:
            r = step["resample"]
            operator = r.get("operator", "mean")
            if "time" not in ds.dims:
                raise SystemExit("resample step requires a time dimension")
            func = getattr(ds.resample(time=r["freq"]), operator, None)
            if func is None:
                raise SystemExit(f"unsupported resample operator: {operator}")
            ds = func()
        else:
            raise SystemExit(f"unknown step: {list(step)}")
    return ds


def describe(ds) -> dict[str, Any]:
    return {
        "dims": dict(ds.sizes),
        "variables": list(ds.data_vars),
        "coords": list(ds.coords),
        "units": {str(name): ds[name].attrs.get("units") for name in ds.data_vars},
    }


def run_guard(input_path: Path, output_path: Path, overwrite: bool) -> None:
    guard = Path(__file__).resolve().parent / "zarr_write_guard.py"
    cmd = [
        sys.executable,
        str(guard),
        "--operation",
        "convert",
        "--input",
        str(input_path),
        "--output",
        str(output_path),
        "--allow-write",
        "--ack-risk",
        GUARD_ACK,
    ]
    if overwrite:
        cmd.append("--overwrite")
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        detail = (exc.stdout or "") + (exc.stderr or "")
        raise SystemExit(f"Zarr write guard refused conversion:\n{detail}") from exc


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", required=True, help="NetCDF file or Zarr store to convert")
    parser.add_argument("--output", required=True, help="target Zarr store path")
    parser.add_argument("--steps-config", default=None, help="JSON/YAML steps config")
    parser.add_argument("--allow-write", action="store_true", help="confirm the write after guard approval")
    parser.add_argument("--overwrite", action="store_true", help="confirm replacing an existing output store")
    parser.add_argument("--ack-risk", default=None, help=f"must equal: {GUARD_ACK}")
    args = parser.parse_args(argv)

    input_path = Path(args.input).expanduser()
    output_path = Path(args.output).expanduser()
    if not input_path.exists():
        raise SystemExit(f"input not found: {input_path}")

    steps = load_steps_config(args.steps_config)
    try:
        ds = open_input(input_path)
    except Exception as exc:
        if format_label(input_path) == "grib":
            raise SystemExit(
                f"GRIB decode failed: {type(exc).__name__}: {exc}\n"
                "Hint: multi-variable ERA5-Land GRIB mixes GRIB editions and may not "
                "decode as a whole; prefer NetCDF for multi-variable downloads or split "
                "the GRIB by variable (see references/data-preprocessing.md)."
            ) from exc
        raise
    try:
        before = describe(ds)
        ds = apply_steps(ds, steps)
        after = describe(ds)
    finally:
        ds.close()

    print(f"input : {input_path}")
    print(f"format: {format_label(input_path)}")
    print(f"before: {before['dims']} vars={before['variables']}")
    if steps:
        print(f"steps : {len(steps)} (see config: {args.steps_config})")
    print(f"after : {after['dims']} vars={after['variables']}")
    print(f"output: {output_path}")

    if not args.allow_write:
        print("\nDRY-RUN: nothing was written.")
        print("To execute, pass --allow-write --ack-risk I-understand-this-mutates-zarr")
        return 0

    if args.ack_risk != GUARD_ACK:
        raise SystemExit(f"Refusing write: --ack-risk must equal {GUARD_ACK}")

    run_guard(input_path, output_path, args.overwrite)

    # Guard passed; reopen the dataset for the actual write.
    ds = open_input(input_path)
    try:
        ds = apply_steps(ds, steps)
        ds.to_zarr(str(output_path), mode="w" if args.overwrite else "w-")
    finally:
        ds.close()
    print(f"\nWROTE {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
