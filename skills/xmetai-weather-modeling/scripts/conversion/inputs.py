"""Input discovery and lazy dataset opening."""
from __future__ import annotations

import glob
import importlib.util
from pathlib import Path

try:
    import xarray as xr
except ImportError:  # pragma: no cover
    xr = None


def is_zarr_dir(path: Path) -> bool:
    return (path / ".zgroup").is_file() or (path / "zarr.json").is_file()


def format_label(path: Path) -> str:
    if is_zarr_dir(path):
        return "zarr"
    if path.suffix.lower() in (".grib", ".grb", ".grib1", ".grib2"):
        return "grib"
    return "netcdf"


def netcdf_engine_for(paths: list[Path] | tuple[Path, ...]) -> str | None:
    """Prefer h5netcdf for HDF5 files, avoiding Windows netCDF4 reopen failures."""
    signatures = []
    for path in paths:
        try:
            with path.open("rb") as stream:
                signatures.append(stream.read(8))
        except OSError:
            return None
    hdf5 = b"\x89HDF\r\n\x1a\n"
    if signatures and all(signature == hdf5 for signature in signatures):
        if importlib.util.find_spec("h5netcdf") is not None:
            return "h5netcdf"
    # netCDF4 can read both classic CDF and HDF5. Keep xarray's backend
    # discovery as a fallback when it is not installed.
    if importlib.util.find_spec("netCDF4") is not None:
        return "netcdf4"
    return None


def _open_grib_by_shortnames(path: Path):
    """Fallback for GRIB files cfgrib cannot open as a whole (for example
    multi-variable ERA5-Land files mixing GRIB editions): read each variable
    with ``filter_by_keys`` and merge the results."""
    try:
        from eccodes import codes_grib_new_from_file, codes_get, codes_release
    except ImportError as exc:  # pragma: no cover - cfgrib depends on eccodes
        raise RuntimeError("eccodes is unavailable for the per-variable GRIB fallback") from exc

    shortnames: list[str] = []
    with open(path, "rb") as handle:
        while True:
            gid = codes_grib_new_from_file(handle)
            if gid is None:
                break
            try:
                name = codes_get(gid, "shortName")
            finally:
                codes_release(gid)
            if name not in shortnames:
                shortnames.append(name)
    if not shortnames:
        raise RuntimeError(f"no GRIB messages found in {path}")

    datasets = []
    try:
        for name in shortnames:
            datasets.append(
                xr.open_dataset(
                    path,
                    engine="cfgrib",
                    backend_kwargs={"filter_by_keys": {"shortName": name}, "indexpath": ""},
                )
            )
        return xr.merge(datasets, compat="override")
    finally:
        for ds in datasets:
            ds.close()


def open_input(path: Path):
    if xr is None:
        raise SystemExit("xarray is not installed")
    if is_zarr_dir(path):
        return xr.open_zarr(path)
    if format_label(path) == "grib":
        try:
            return xr.open_dataset(path, engine="cfgrib", backend_kwargs={"indexpath": ""})
        except Exception:
            return _open_grib_by_shortnames(path)
    engine = netcdf_engine_for([path])
    return xr.open_dataset(path, engine=engine) if engine else xr.open_dataset(path)


def open_inputs(paths: list[Path], chunks: dict[str, int] | None = None):
    """Open one input or lazily combine homogeneous NetCDF files by coordinates."""
    if len(paths) == 1:
        ds = open_input(paths[0])
        if not chunks:
            return ds
        applicable = {name: size for name, size in chunks.items() if name in ds.dims}
        return ds.chunk(applicable) if applicable else ds
    unsupported = [str(p) for p in paths if format_label(p) != "netcdf"]
    if unsupported:
        raise SystemExit(
            "multiple inputs currently require NetCDF files; convert Zarr/GRIB inputs "
            f"separately: {unsupported[:3]}"
        )
    engine = netcdf_engine_for(paths)
    return xr.open_mfdataset(
        [str(p) for p in paths],
        combine="by_coords",
        # netCDF4/HDF5 file opens are not reliably thread-safe on Windows.
        # Data variables remain lazy and downstream Dask reductions/writes can
        # still execute chunk tasks concurrently.
        parallel=False,
        chunks=chunks,
        data_vars="minimal",
        coords="minimal",
        compat="override",
        engine=engine,
    )


def parse_chunks(value: str | None) -> dict[str, int] | None:
    """Parse ``time=4,level=76`` style chunk specifications."""
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


def resolve_inputs(values: list[str], pattern: str | None) -> list[Path]:
    """Resolve explicit inputs plus an optional glob into sorted unique paths."""
    raw = list(values)
    if pattern:
        raw.extend(sorted(glob.glob(str(Path(pattern).expanduser()))))
    paths: list[Path] = []
    seen: set[Path] = set()
    for value in raw:
        path = Path(value).expanduser()
        resolved = path.resolve()
        if resolved not in seen:
            paths.append(path)
            seen.add(resolved)
    if not paths:
        raise SystemExit("no inputs matched --input/--input-glob")
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise SystemExit(f"input not found: {missing[:3]}")
    return paths
