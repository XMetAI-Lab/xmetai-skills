"""Write guards, conversion fingerprints, and resume validation."""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

GUARD_ACK = "I-understand-this-mutates-zarr"


def describe(ds) -> dict[str, Any]:
    return {
        "dims": dict(ds.sizes),
        "variables": list(ds.data_vars),
        "coords": list(ds.coords),
        "units": {str(name): ds[name].attrs.get("units") for name in ds.data_vars},
    }


def run_guard(input_paths: list[Path], output_path: Path, overwrite: bool, resume: bool = False) -> None:
    guard = Path(__file__).resolve().parent.parent / "zarr_write_guard.py"
    cmd = [
        sys.executable,
        str(guard),
        "--operation",
        "append" if resume else "convert",
        "--output",
        str(output_path),
        "--allow-write",
        "--ack-risk",
        GUARD_ACK,
    ]
    for input_path in input_paths:
        cmd += ["--input", str(input_path)]
    if overwrite:
        cmd.append("--overwrite")
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        detail = (exc.stdout or "") + (exc.stderr or "")
        raise SystemExit(f"Zarr write guard refused conversion:\n{detail}") from exc


def conversion_state_path(output_path: Path) -> Path:
    """Return the audit-state path without placing non-Zarr files in the store."""
    return output_path.with_name(f"{output_path.name}.conversion.json")


def conversion_fingerprint(
    input_paths: list[Path], steps: list[dict[str, Any]], input_chunks, output_chunks, execution=None
) -> dict[str, Any]:
    """Describe inputs and transformation settings used to validate a resume."""
    inputs = [
        {
            "path": str(path.resolve()),
            "size": path.stat().st_size if path.is_file() else None,
            "mtime_ns": path.stat().st_mtime_ns,
        }
        for path in input_paths
    ]
    contract = {
        "steps": steps,
        "input_chunks": input_chunks,
        "output_chunks": output_chunks,
        "execution": execution or {},
    }
    encoded = json.dumps(contract, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {"contract_sha256": hashlib.sha256(encoded).hexdigest(), "inputs": inputs}


def write_conversion_state(path: Path, state: dict[str, Any]) -> None:
    """Atomically replace the small conversion audit file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def _time_values(ds) -> np.ndarray:
    if "time" not in ds.coords or "time" not in ds.dims:
        raise SystemExit("incremental Zarr conversion requires a time dimension and coordinate")
    values = np.asarray(ds.time.values)
    if values.ndim != 1 or values.size == 0:
        raise SystemExit("incremental Zarr conversion requires a non-empty one-dimensional time coordinate")
    if np.any(values[1:] <= values[:-1]):
        raise SystemExit("incremental Zarr conversion requires strictly increasing unique time values")
    return values


def validate_resume_prefix(planned, existing) -> int:
    """Return the first missing index when the existing times are an exact prefix."""
    planned_values = _time_values(planned)
    existing_values = _time_values(existing)
    if existing_values.size > planned_values.size or not np.array_equal(
        existing_values, planned_values[: existing_values.size]
    ):
        raise SystemExit(
            "resume refused: existing Zarr times are not an exact prefix of the planned output; "
            "overlapping, missing, reordered, or unrelated times require a new output or approved overwrite"
        )
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
    return int(existing_values.size)
