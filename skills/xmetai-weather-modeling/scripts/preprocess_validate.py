#!/usr/bin/env python3
"""Validate a prepared NetCDF/Zarr dataset after preprocessing or conversion.

Checks:

1. Variables: every expected variable is present.
2. Units: detected units match expected units for listed variables.
3. Time: coordinate exists, is monotonic, continuous at the expected
   frequency, and covers the requested range when provided.

Usage:

    python preprocess_validate.py --path era5_sample.nc --config expected.json
    python preprocess_validate.py --path out.zarr --variables z,t,q --freq 6h
    python preprocess_validate.py --path out.zarr --config expected.json --json

Config file (JSON or YAML):

    {
      "variables": ["z", "t", "q"],
      "units": {"z": "m2 s-2", "t": "K"},
      "freq": "6h",
      "time": {"start": "2023-06-01", "end": "2023-06-03"}
    }

Exit code is 0 when every check passes and 1 otherwise. Read-only.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

try:
    import pandas as pd
    import xarray as xr
except ImportError:  # pragma: no cover - environment without parsing deps
    pd = None
    xr = None


def is_zarr_dir(path: Path) -> bool:
    return (path / ".zgroup").is_file() or (path / "zarr.json").is_file()


def open_input(path: Path):
    if xr is None:
        raise SystemExit("xarray is not installed")
    if is_zarr_dir(path):
        return xr.open_zarr(path)
    return xr.open_dataset(path)


def load_expected(config: str | None) -> dict[str, Any]:
    if config is None:
        return {}
    path = Path(config).expanduser()
    if not path.is_file():
        raise SystemExit(f"config not found: {config}")
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
            raise SystemExit(f"config must be JSON or YAML: {exc}") from exc
    return payload


def parse_freq_hours(freq: str) -> float:
    f = freq.strip().lower()
    if f.endswith("h"):
        return float(f[:-1])
    if f.endswith("d"):
        return float(f[:-1]) * 24.0
    raise SystemExit(f"unsupported freq format: {freq} (use e.g. 6h, 12h, 1d)")


def check_variables(ds, expected: list[str]) -> dict[str, Any]:
    found = list(ds.data_vars)
    missing = [v for v in expected if v not in found]
    return {
        "check": "variables",
        "status": "PASS" if not missing else "FAIL",
        "detail": f"expected={len(expected)} found={len(found)} missing={missing or 'none'}",
    }


def check_units(ds, units: dict[str, str]) -> dict[str, Any]:
    if not units:
        return {"check": "units", "status": "INFO", "detail": "no expected units provided"}
    failures = []
    notes = []
    for name, expected in units.items():
        if name not in ds.data_vars:
            continue
        actual = str(ds[name].attrs.get("units", "")).strip()
        if actual != expected.strip():
            failures.append(f"{name}[{actual or '?'}] expected {expected}")
        else:
            notes.append(f"{name}[ok]")
    detail = "; ".join(notes) if notes else "no units checked"
    if failures:
        detail = "; ".join(failures)
    return {"check": "units", "status": "FAIL" if failures else "PASS", "detail": detail}


def check_time(
    ds,
    freq: str | None,
    start: str | None,
    end: str | None,
    time_coord: str | None = None,
) -> dict[str, Any]:
    candidates = [time_coord] if time_coord else ["time", "valid_time"]
    found_coord = next((name for name in candidates if name in ds.coords or name in ds.dims), None)
    if found_coord is None:
        return {"check": "time", "status": "FAIL", "detail": f"no time coordinate (looked for {candidates})"}
    times = ds[found_coord].values
    if not pd.Series(times).is_monotonic_increasing:
        return {"check": "time", "status": "FAIL", "detail": "time is not monotonic increasing"}

    detail = f"coord={found_coord} steps={len(times)} range={times[0]} ~ {times[-1]}"
    if freq is None:
        return {"check": "time", "status": "INFO", "detail": f"{detail}; no expected freq provided"}

    expected_hours = parse_freq_hours(freq)
    diffs = np.diff(times.astype("datetime64[ns]").astype("int64")) / 1e9 / 3600.0
    tolerance = max(expected_hours * 0.05, 1e-6)
    gaps = diffs[diffs > expected_hours + tolerance]
    dense = diffs[diffs < expected_hours - tolerance]

    failures = []
    if len(gaps) > 0:
        failures.append(f"{len(gaps)} gap(s) at freq {freq}")
    if start is not None and pd.Timestamp(times[0]) > pd.Timestamp(start):
        failures.append(f"missing data before {start} (first={times[0]})")
    if end is not None and pd.Timestamp(times[-1]) < pd.Timestamp(end):
        failures.append(f"missing data after {end} (last={times[-1]})")
    if len(dense) > 0:
        failures.append(f"{len(dense)} step(s) denser than {freq} (info)")

    detail += f" freq={freq}"
    if len(gaps) > 0:
        gap_index = np.where(diffs > expected_hours + tolerance)[0]
        detail += f" gaps_after={list(times[gap_index])}"
    return {"check": "time", "status": "FAIL" if failures else "PASS", "detail": detail}


def print_text(results: list[dict[str, Any]], overall: str) -> None:
    for r in results:
        print(f"{r['check']:10s} {r['status']:5s} {r['detail']}")
    print(f"\noverall: {overall}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--path", required=True, help="NetCDF file or Zarr store to validate")
    parser.add_argument("--config", default=None, help="JSON/YAML expected checks")
    parser.add_argument("--variables", default=None, help="comma-separated expected variables")
    parser.add_argument("--freq", default=None, help="expected time frequency, e.g. 6h, 1d")
    parser.add_argument("--start", default=None, help="expected coverage start")
    parser.add_argument("--end", default=None, help="expected coverage end")
    parser.add_argument("--json", action="store_true", help="emit JSON output")
    args = parser.parse_args(argv)

    path = Path(args.path).expanduser()
    if not path.exists():
        raise SystemExit(f"path not found: {path}")

    expected = load_expected(args.config)
    variables = args.variables.split(",") if args.variables else expected.get("variables", [])
    units = expected.get("units", {})
    freq = args.freq or expected.get("freq")
    time_cfg = expected.get("time", {})
    start = args.start or time_cfg.get("start")
    end = args.end or time_cfg.get("end")
    time_coord = expected.get("time_coord")

    ds = open_input(path)
    try:
        results = [
            check_variables(ds, variables),
            check_units(ds, units),
            check_time(ds, freq, start, end, time_coord),
        ]
    finally:
        ds.close()

    overall = "VALID" if not any(r["status"] == "FAIL" for r in results) else "INVALID"
    if args.json:
        print(json.dumps({"target": str(path), "checks": results, "overall": overall}, indent=2, ensure_ascii=False))
    else:
        print(f"target: {path}")
        print_text(results, overall)
    return 0 if overall == "VALID" else 1


if __name__ == "__main__":
    sys.exit(main())
