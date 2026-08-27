"""Coordinate canonicalization and configurable regridding."""
from __future__ import annotations

import numpy as np

try:
    import xarray as xr
except ImportError:  # pragma: no cover
    xr = None


def canonicalize_latlon(ds):
    """Rename ``latitude``/``longitude`` coordinates to ``lat``/``lon``.

    Core dataset classes (``MultiZarrDataset`` bbox path, ``GraphCastDataset``)
    access ``ds.lat`` / ``ds.lon`` directly, while CDS NetCDF/GRIB inputs carry
    ``latitude``/``longitude``. Keeping the output Zarr coordinates as
    ``lat``/``lon`` makes conversion products consumable without per-dataset
    adapters.
    """
    rename = {}
    if "latitude" in ds.coords:
        rename["latitude"] = "lat"
    if "longitude" in ds.coords:
        rename["longitude"] = "lon"
    return ds.rename(rename) if rename else ds


def _s2s_target_coords() -> dict[str, np.ndarray]:
    """Return the exact 121 x 240 grid used by the S2S cla.zarr contract."""
    return {
        "lat": np.linspace(90.0, -90.0, 121, dtype=np.float64),
        "lon": np.arange(240, dtype=np.float64) * 1.5,
    }

TARGET_GRIDS = {"s2s_1.5deg": _s2s_target_coords}
REGRID_METHODS = frozenset({"linear", "nearest"})

def _prepare_source_grid(ds):
    """Canonicalize a rectilinear source grid for xarray interpolation."""
    ds = canonicalize_latlon(ds)
    missing = [name for name in ("lat", "lon") if name not in ds.coords]
    if missing:
        raise SystemExit(f"regrid requires coordinate(s): {missing}")
    for name in ("lat", "lon"):
        coord = ds[name]
        if coord.ndim != 1 or coord.dims != (name,):
            raise SystemExit(f"regrid requires a 1-D {name!r} coordinate")
        values = np.asarray(coord.values, dtype=np.float64)
        if not np.isfinite(values).all():
            raise SystemExit(f"regrid coordinate {name!r} contains non-finite values")

    lon = np.mod(np.asarray(ds.lon.values, dtype=np.float64), 360.0)
    ds = ds.assign_coords(lon=lon).sortby("lon").sortby("lat")
    for name in ("lat", "lon"):
        values = np.asarray(ds[name].values)
        if np.unique(values).size != values.size:
            raise SystemExit(f"regrid coordinate {name!r} contains duplicates")
    return ds


def regrid_dataset(ds, config):
    """Lazily interpolate a rectilinear dataset onto a named target grid."""
    if isinstance(config, str):
        config = {"target": config}
    if not isinstance(config, dict):
        raise SystemExit("regrid must be a target name or a mapping")
    target_name = str(config.get("target", "s2s_1.5deg"))
    if target_name not in TARGET_GRIDS:
        raise SystemExit(f"unsupported regrid target: {target_name!r}")
    default_method = str(config.get("method", "linear"))
    variable_methods = config.get("variable_methods", {})
    if not isinstance(variable_methods, dict):
        raise SystemExit("regrid variable_methods must be a mapping")
    methods = {str(name): str(method) for name, method in variable_methods.items()}
    unknown_vars = sorted(set(methods) - set(ds.data_vars))
    if unknown_vars:
        raise SystemExit(f"regrid variable_methods variables not found: {unknown_vars}")
    invalid = sorted({default_method, *methods.values()} - REGRID_METHODS)
    if invalid:
        raise SystemExit(f"unsupported regrid method(s): {invalid}; use linear or nearest")

    ds = _prepare_source_grid(ds)
    target = TARGET_GRIDS[target_name]()
    source_lat = np.asarray(ds.lat.values, dtype=np.float64)
    source_lon = np.asarray(ds.lon.values, dtype=np.float64)
    lat_index = np.searchsorted(source_lat, target["lat"])
    lon_index = np.searchsorted(source_lon, target["lon"])
    aligned = (
        np.all(lat_index < source_lat.size)
        and np.all(lon_index < source_lon.size)
        and np.allclose(source_lat[lat_index], target["lat"], rtol=0.0, atol=1e-10)
        and np.allclose(source_lon[lon_index], target["lon"], rtol=0.0, atol=1e-10)
    )
    if aligned:
        # Standard 0.25-degree ERA5 contains every 1.5-degree target point.
        # Indexing is exact, lazy, and avoids loading SciPy for interpolation.
        result = ds.isel(lat=lat_index, lon=lon_index).assign_coords(target)
        return result

    grouped: dict[str, list[str]] = {}
    for name in ds.data_vars:
        grouped.setdefault(methods.get(str(name), default_method), []).append(str(name))
    try:
        parts = [ds[names].interp(target, method=method) for method, names in grouped.items()]
    except ModuleNotFoundError as exc:
        if exc.name == "scipy":
            raise SystemExit(
                "regrid requires scipy when the source grid is not exactly aligned "
                "with s2s_1.5deg"
            ) from exc
        raise
    result = xr.merge(parts, compat="override")
    result.attrs.update(ds.attrs)
    if result.sizes.get("lat") != 121 or result.sizes.get("lon") != 240:
        raise SystemExit("s2s_1.5deg regrid did not produce the required 121 x 240 grid")
    if not np.array_equal(result.lat.values, target["lat"]) or not np.array_equal(
        result.lon.values, target["lon"]
    ):
        raise SystemExit("s2s_1.5deg output coordinates do not match the target contract")
    return result
