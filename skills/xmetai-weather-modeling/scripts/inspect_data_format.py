#!/usr/bin/env python3
"""Read-only data format detection and metadata summary.

Detects the format of meteorological data files by extension and magic
bytes, then reads metadata (dims, coordinates, variables, units) without
loading data values. Directories are scanned recursively; Zarr stores are
recognized by their store marker and treated as a single target.

Usage:

    python inspect_data_format.py --path <file-or-directory>
    python inspect_data_format.py --path <file-or-directory> --json
    python inspect_data_format.py --path <file-or-directory> --config expected.json

``--config`` accepts a JSON file with an ``expected_variables`` list used
for a lightweight name-level reconciliation against detected variables.

This script never writes files and never loads array data.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterator

try:
    import xarray as xr
except ImportError:  # pragma: no cover - environment without parsing deps
    xr = None

EXTENSION_FORMATS = {
    ".nc": "netcdf",
    ".nc4": "netcdf",
    ".cdf": "netcdf",
    ".h5": "hdf5",
    ".hdf": "hdf5",
    ".grib": "grib",
    ".grb": "grib",
    ".grib1": "grib",
    ".grib2": "grib",
    ".npz": "npz",
    ".npy": "npy",
    ".bin": "binary",
}

MAGIC_RULES = (
    ("netcdf", b"CDF"),
    ("hdf5", b"\x89HDF\r\n\x1a\n"),
    ("grib", b"GRIB"),
    ("zip", b"PK\x03\x04"),
)


def _read_head(path: Path, n: int = 8) -> bytes:
    with path.open("rb") as handle:
        return handle.read(n)


def is_zarr_dir(path: Path) -> bool:
    return (path / ".zgroup").is_file() or (path / "zarr.json").is_file()


def detect_magic(path: Path) -> str | None:
    head = _read_head(path)
    for fmt, magic in MAGIC_RULES:
        if head.startswith(magic):
            return fmt
    return None


def iter_targets(path: Path) -> Iterator[Path]:
    """Yield files and Zarr directories under ``path`` (recursively)."""
    if path.is_file():
        yield path
        return
    if is_zarr_dir(path):
        yield path
        return
    for child in sorted(path.iterdir()):
        if child.is_dir():
            if is_zarr_dir(child):
                yield child
            else:
                yield from iter_targets(child)
        else:
            yield child


def read_metadata(target: Path, fmt: str) -> dict[str, Any]:
    """Layer 2: read metadata with xarray; never loads data values."""
    if xr is None:
        return {"metadata_error": "xarray is not installed"}
    try:
        if fmt == "zarr":
            ds = xr.open_zarr(target)
        else:
            ds = xr.open_dataset(target)
        try:
            return {
                "dims": dict(ds.sizes),
                "variables": list(ds.data_vars),
                "coords": list(ds.coords),
                "units": {
                    str(name): ds[name].attrs.get("units")
                    for name in ds.data_vars
                },
            }
        finally:
            ds.close()
    except Exception as exc:  # pragma: no cover - read errors vary by input
        return {"metadata_error": f"{type(exc).__name__}: {exc}"}


def classify(target: Path, ext_fmt: str | None, magic_fmt: str | None) -> str:
    """Layer 1 status: recognized / mismatch / decode-pending / unsupported."""
    if is_zarr_dir(target):
        return "recognized"
    if ext_fmt == "grib" and magic_fmt == "grib":
        return "decode-pending"
    if ext_fmt in ("npy", "npz"):
        return "recognized"
    # NetCDF classic (CDF) and NetCDF-4 (HDF5) are the same family.
    if ext_fmt in ("netcdf", "hdf5") and magic_fmt in ("netcdf", "hdf5"):
        return "recognized"
    if ext_fmt == magic_fmt and ext_fmt is not None:
        return "recognized"
    if ext_fmt is None and magic_fmt is not None:
        return "recognized-by-magic"
    if ext_fmt is not None and magic_fmt is None:
        return "mismatch"
    if ext_fmt is not None and magic_fmt is not None and ext_fmt != magic_fmt:
        return "mismatch"
    return "unsupported"


def inspect_target(target: Path) -> dict[str, Any]:
    if is_zarr_dir(target):
        result: dict[str, Any] = {
            "path": str(target),
            "format": "zarr",
            "magic": "zarr",
            "size_bytes": None,
            "status": "recognized",
        }
        result.update(read_metadata(target, "zarr"))
        return result

    ext_fmt = EXTENSION_FORMATS.get(target.suffix.lower())
    magic_fmt = detect_magic(target)
    result = {
        "path": str(target),
        "format": ext_fmt,
        "magic": magic_fmt,
        "size_bytes": target.stat().st_size,
        "status": classify(target, ext_fmt, magic_fmt),
    }

    effective_format = ext_fmt or magic_fmt
    if result["status"] in ("recognized", "recognized-by-magic") and effective_format in (
        "netcdf",
        "hdf5",
    ):
        result.update(read_metadata(target, "netcdf"))
    return result


def check_variables(metadata: dict[str, Any], expected: list[str]) -> dict[str, Any]:
    detected = metadata.get("variables") or []
    return {
        "matched": sorted(set(expected) & set(detected)),
        "missing": sorted(set(expected) - set(detected)),
    }


def load_expected(config: str | None) -> list[str] | None:
    if config is None:
        return None
    path = Path(config)
    if not path.is_file():
        raise SystemExit(f"config file not found: {config}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"config must be JSON: {config} ({exc})") from exc
    expected = payload.get("expected_variables")
    if not isinstance(expected, list) or not all(isinstance(v, str) for v in expected):
        raise SystemExit("config JSON must contain an 'expected_variables' string list")
    return expected


def print_text(results: list[dict[str, Any]], summary: dict[str, int]) -> None:
    for result in results:
        print(f"target: {result['path']}")
        print(f"  format: {result['format'] or 'unknown'} (magic: {result['magic'] or 'none'})")
        if result.get("size_bytes") is not None:
            print(f"  size: {result['size_bytes'] / 1024:.0f} KiB")
        if result.get("dims"):
            dims = ", ".join(f"{k}({v})" for k, v in result["dims"].items())
            print(f"  dims: {dims}")
        if result.get("variables"):
            print(f"  variables: {', '.join(result['variables'])}")
        if result.get("units"):
            units = ", ".join(f"{k}[{v}]" if v else f"{k}[?]" for k, v in result["units"].items())
            print(f"  units: {units}")
        if result.get("config_check"):
            check = result["config_check"]
            print(f"  config: matched={check['matched']} missing={check['missing']}")
        if result.get("metadata_error"):
            print(f"  metadata error: {result['metadata_error']}")
        print(f"  status: {result['status']}")
    print()
    print("summary:", ", ".join(f"{k}={v}" for k, v in sorted(summary.items())))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--path", required=True, help="file or directory to inspect")
    parser.add_argument("--json", action="store_true", help="emit JSON output")
    parser.add_argument("--config", default=None, help="JSON file with expected_variables")
    args = parser.parse_args(argv)

    root = Path(args.path).expanduser()
    if not root.exists():
        raise SystemExit(f"path not found: {root}")

    expected = load_expected(args.config)
    results = []
    for target in iter_targets(root):
        result = inspect_target(target)
        if expected is not None and result.get("variables") is not None:
            result["config_check"] = check_variables(result, expected)
        results.append(result)

    summary: dict[str, int] = {}
    for result in results:
        status = result["status"]
        summary[status] = summary.get(status, 0) + 1

    if args.json:
        print(json.dumps({"files": results, "summary": summary}, indent=2, ensure_ascii=False))
    else:
        print_text(results, summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
