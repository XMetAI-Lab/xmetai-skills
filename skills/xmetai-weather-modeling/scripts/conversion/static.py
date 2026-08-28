"""Static-field output validation and extraction."""
from __future__ import annotations

from pathlib import Path


def guard_static_output(input_paths: list[Path], output_path: Path, overwrite: bool) -> None:
    """Apply the non-destructive path checks used for a static NetCDF write."""
    output_resolved = output_path.resolve()
    if output_path.suffix.lower() != ".nc":
        raise SystemExit("static-netcdf output must use a .nc suffix")
    if any(path.resolve() == output_resolved for path in input_paths):
        raise SystemExit("Refusing static conversion with output equal to an input")
    if output_path.exists() and not overwrite:
        raise SystemExit(f"Refusing to replace existing static output without --overwrite: {output_path}")
    if output_path.exists() and not output_path.is_file():
        raise SystemExit(f"static-netcdf output exists and is not a file: {output_path}")


def static_dataarray(ds):
    """Validate and return a core-compatible static DataArray."""
    if list(ds.data_vars) != ["const"]:
        raise SystemExit("static-netcdf requires exactly one data variable named 'const'")
    da = ds["const"]
    if da.dims != ("channel", "lat", "lon"):
        raise SystemExit(
            "static-netcdf requires const dimensions ('channel', 'lat', 'lon'), "
            f"got {da.dims}"
        )
    if da.sizes["channel"] == 0 or da.sizes["lat"] == 0 or da.sizes["lon"] == 0:
        raise SystemExit("static-netcdf dimensions must be non-empty")
    return da
