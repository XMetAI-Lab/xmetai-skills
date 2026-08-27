#!/usr/bin/env python3
"""Convert meteorological NetCDF/Zarr input into a normalized Zarr store.

The script is dry-run by default: it prints the conversion plan (input
metadata, applied steps, target layout) without writing anything. To perform
the write, pass ``--allow-write`` and ``--ack-risk``; the Zarr write guard is
then executed before any mutation.

Usage:

    python convert_to_zarr.py --input era5_sample.nc --output out.zarr
    python convert_to_zarr.py --input-glob "era5_*.nc" --output out.zarr
    python convert_to_zarr.py --input era5_sample.nc --output out.zarr --steps-config steps.json
    python convert_to_zarr.py --input era5_sample.nc --output out.zarr --allow-write --ack-risk I-understand-this-mutates-zarr

Steps config (JSON or YAML):

    {
      "steps": [
        {"rename": {"total_cloud_cover": "clt"}},
        {"keep_vars": ["z", "t", "q"]},
        {"time": {"start": "2023-06-01", "end": "2023-06-02"}},
        {"resample": {"freq": "6h", "operator": "mean"}},
        {"daily_aggregation": {
            "window_hours": 24, "label_hour": 0, "incomplete": "error",
            "variables": {
                "tp": {"operator": "sum", "factor": 1000, "units": "mm"},
                "ttr": {"operator": "sum", "factor": 1.1574074074074073e-5,
                        "units": "W m-2"}
            }}},
        {"regrid": {"target": "s2s_1.5deg", "method": "linear",
                    "variable_methods": {"lsm": "nearest"}}},
        {"merge_static": {"order": ["z", "lsm"], "coord": "channel",
                          "name": "const"}},
        {"units": {"q": 1000}},
        {"log1p": ["tp"]},
        {"split_levels": {"vars": ["z", "t", "u", "v", "q"],
                          "level_coord": "pressure_level",
                          "levels": [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]}},
        {"merge_to_data": {"coord": "level", "order": ["z1000", "z925", "z850", "z700", "z600", "z500", "z400", "z300", "z250", "z200", "z150", "z100", "z50", "t1000", "t925", "t850", "t700", "t600", "t500", "t400", "t300", "t250", "t200", "t150", "t100", "t50", "u1000", "u925", "u850", "u700", "u600", "u500", "u400", "u300", "u250", "u200", "u150", "u100", "u50", "v1000", "v925", "v850", "v700", "v600", "v500", "v400", "v300", "v250", "v200", "v150", "v100", "v50", "q1000", "q925", "q850", "q700", "q600", "q500", "q400", "q300", "q250", "q200", "q150", "q100", "q50"]}},
        {"normalize": true}
      ]
    }

``merge_to_data`` combines data variables into a single ``data`` variable
with the given channel coordinate (``level`` or ``channel``), matching the
model library layout. ``split_levels`` expands a variable with a level
dimension (for example CDS pressure levels) into one variable per level
using ``name_template`` (default ``{var}{level}``), so it can feed
``merge_to_data``. ``units`` multiplies variables by the given factors and
``log1p`` applies ``log1p(clip(min=0))`` to the listed variables. GRIB inputs
that mix GRIB editions are read per variable and merged automatically;
cfgrib index files are not written (``indexpath=""``). ``normalize`` computes
per-channel mean/std and cosine-latitude spatial weights from the prepared data,
scales the stored values with ``(x - mean) / std``, and writes
``mean.nc`` / ``std.nc`` / ``weight.nc`` next to the output Zarr. Unknown
steps abort the plan. The script never writes without the guard.

Static fields use the same transforms, followed by ``merge_static`` and
``--output-format static-netcdf``.  Every time-like dimension on a static
variable must have length one; it is removed before writing a core-compatible
``const(channel, lat, lon)`` DataArray.

Precipitation unit convention: precipitation channels are normalized to
**mm accumulated values** before ``log1p``/``normalize``. ERA5 and ERA5-Land
deliver ``tp`` in metres (step-accumulated), so the steps config multiplies
it by 1000 (``"tp": 1000``). For hourly ERA5 accumulations feeding S2S, use
``daily_aggregation`` instead: it applies a configurable interval-ending
window, aggregation operator, factor, offset, and output unit per variable.
The example sums the 24 samples through 00 UTC, converts ``tp`` from metres
to mm/day, and converts summed ``ttr`` energy from J/m2 to daily-mean W/m2.
The same step can aggregate custom variables such as ``ssr``. Do not apply
another ``units`` conversion to those variables afterwards. Rate-form precipitation
(``kg m-2 s-1`` or ``mm/h`` averages) must first be multiplied by the
accumulation length in seconds. The accumulation window must follow the
selected model contract (for example daily totals for S2S, 6-hourly for IWC).
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import importlib.util
import json
import re
import subprocess
import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np

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


def daily_aggregation(ds, config):
    """Aggregate configurable hourly variables into interval-ending daily fields."""
    if not isinstance(config, dict):
        raise SystemExit("daily_aggregation must be a mapping")
    if "time" not in ds.dims or "time" not in ds.coords:
        raise SystemExit("daily_aggregation requires a time coordinate")

    variables = config.get("variables")
    if not isinstance(variables, dict) or not variables:
        raise SystemExit("daily_aggregation variables must be a non-empty mapping")
    window_hours = int(config.get("window_hours", 24))
    label_hour = int(config.get("label_hour", 0))
    incomplete = str(config.get("incomplete", "error"))
    if window_hours <= 0 or window_hours > 24:
        raise SystemExit("daily_aggregation window_hours must be between 1 and 24")
    if label_hour < 0 or label_hour > 23:
        raise SystemExit("daily_aggregation label_hour must be between 0 and 23")
    if incomplete not in {"error", "drop"}:
        raise SystemExit("daily_aggregation incomplete must be 'error' or 'drop'")

    variable_specs = {}
    for raw_name, raw_spec in variables.items():
        name = str(raw_name)
        if not isinstance(raw_spec, dict):
            raise SystemExit(f"daily_aggregation: specification for {name!r} must be a mapping")
        operator = str(raw_spec.get("operator", "sum"))
        if operator not in {"sum", "mean", "min", "max"}:
            raise SystemExit(
                f"daily_aggregation: unsupported operator {operator!r} for {name!r}; "
                "use sum, mean, min, or max"
            )
        variable_specs[name] = {
            "operator": operator,
            "factor": float(raw_spec.get("factor", 1.0)),
            "offset": float(raw_spec.get("offset", 0.0)),
            "units": raw_spec.get("units"),
        }

    missing_vars = [name for name in variable_specs if name not in ds.data_vars]
    if missing_vars:
        raise SystemExit(f"daily_aggregation variables not found: {missing_vars}")
    for name in variable_specs:
        if "time" not in ds[name].dims:
            raise SystemExit(f"daily_aggregation: {name!r} has no time dimension")

    times = np.asarray(ds.time.values)
    if times.ndim != 1 or times.size == 0 or not np.issubdtype(times.dtype, np.datetime64):
        raise SystemExit("daily_aggregation requires a non-empty datetime64 time coordinate")
    times_ns = times.astype("datetime64[ns]")
    if np.any(np.isnat(times_ns)):
        raise SystemExit("daily_aggregation time contains NaT")
    if np.any(np.diff(times_ns) <= np.timedelta64(0, "ns")):
        raise SystemExit("daily_aggregation time must be strictly increasing and unique")
    if np.any(times_ns != times_ns.astype("datetime64[h]")):
        raise SystemExit("daily_aggregation timestamps must be aligned to whole hours")

    hour = np.timedelta64(1, "h")
    time_hours = times_ns.astype("datetime64[h]").astype(np.int64)
    available = set(time_hours.tolist())
    candidates = times_ns[np.mod(time_hours, 24) == label_hour]
    complete = []
    incomplete_ends = []
    for end in candidates:
        end_hour = int(end.astype("datetime64[h]").astype(np.int64))
        if all(end_hour - offset in available for offset in range(window_hours)):
            complete.append(end)
        else:
            incomplete_ends.append(end)

    trailing_partial = bool(candidates.size and times_ns[-1] > candidates[-1])
    leading_partial = bool(candidates.size and times_ns[0] > candidates[0] - (window_hours - 1) * hour)
    if not candidates.size:
        trailing_partial = True
    if incomplete == "error" and (incomplete_ends or leading_partial or trailing_partial):
        details = []
        if incomplete_ends:
            labels = [str(value.astype("datetime64[h]")) for value in incomplete_ends[:3]]
            details.append(f"incomplete daily window(s) ending {labels}")
        if trailing_partial:
            details.append("trailing hours do not end at 00 UTC")
        raise SystemExit(
            f"daily_aggregation requires complete {window_hours}-hour windows "
            f"ending at {label_hour:02d} UTC: " + "; ".join(details)
        )
    if not complete:
        raise SystemExit("daily_aggregation found no complete window")

    target_times = np.asarray(complete, dtype="datetime64[ns]")
    output_vars = {}
    for name in ds.data_vars:
        da = ds[name]
        if name not in variable_specs:
            output_vars[name] = da.sel(time=target_times) if "time" in da.dims else da
            continue
        spec = variable_specs[name]
        rolling = da.rolling(time=window_hours, min_periods=window_hours)
        aggregated = getattr(rolling, spec["operator"])(skipna=False).sel(time=target_times)
        aggregated = aggregated * spec["factor"] + spec["offset"]
        aggregated.attrs.update(da.attrs)
        aggregated.attrs.update(
            aggregation=spec["operator"],
            aggregation_window=f"{window_hours} hours",
            time_label="window_end",
        )
        if spec["units"] is not None:
            aggregated.attrs["units"] = str(spec["units"])
        output_vars[name] = aggregated
    return xr.Dataset(output_vars, attrs=ds.attrs)


def s2s_daily_accumulation(ds, config):
    """Backward-compatible tp/ttr shorthand for ``daily_aggregation``."""
    if not isinstance(config, dict):
        raise SystemExit("s2s_daily_accumulation must be a mapping")
    tp_name = str(config.get("tp", "tp"))
    ttr_name = str(config.get("ttr", "ttr"))
    return daily_aggregation(
        ds,
        {
            "window_hours": 24,
            "label_hour": 0,
            "incomplete": config.get("incomplete", "error"),
            "variables": {
                tp_name: {"operator": "sum", "factor": 1000.0, "units": "mm"},
                ttr_name: {"operator": "sum", "factor": 1.0 / 86400.0, "units": "W m-2"},
            },
        },
    )


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
        elif "s2s_daily_accumulation" in step:
            ds = s2s_daily_accumulation(ds, step["s2s_daily_accumulation"])
        elif "daily_aggregation" in step:
            ds = daily_aggregation(ds, step["daily_aggregation"])
        elif "regrid" in step:
            ds = regrid_dataset(ds, step["regrid"])
        elif "units" in step:
            for name, factor in step["units"].items():
                if name not in ds.data_vars:
                    raise SystemExit(f"units: variable not found: {name}")
                ds[name] = ds[name] * float(factor)
        elif "log1p" in step:
            for name in step["log1p"]:
                if name not in ds.data_vars:
                    raise SystemExit(f"log1p: variable not found: {name}")
                ds[name] = np.log1p(ds[name].clip(min=0))
        elif "split_levels" in step:
            cfg = step["split_levels"]
            vars_to_split = [str(v) for v in cfg.get("vars", [])]
            level_coord = str(cfg.get("level_coord", "pressure_level"))
            template = str(cfg.get("name_template", "{var}{level}"))
            levels = cfg.get("levels")
            if levels is None:
                if level_coord not in ds.coords:
                    raise SystemExit(f"split_levels: coordinate {level_coord!r} not found")
                levels = [int(v) for v in ds.coords[level_coord].values]
            new_vars = {}
            for var in vars_to_split:
                if var not in ds.data_vars:
                    raise SystemExit(f"split_levels: variable not found: {var}")
                da = ds[var]
                if level_coord not in da.dims:
                    raise SystemExit(f"split_levels: {var} has no {level_coord} dim")
                for level in levels:
                    sub = da.sel({level_coord: level}, drop=True)
                    name = template.format(var=var, level=level)
                    new_vars[name] = sub
            kept = {name: ds[name] for name in ds.data_vars if name not in vars_to_split}
            kept.update(new_vars)
            ds = xr.Dataset(kept)
        elif "merge_to_data" in step:
            cfg = step["merge_to_data"]
            coord = str(cfg.get("coord", "level"))
            requested_order = [str(n) for n in cfg.get("order", [])]
            if len(requested_order) != len(set(requested_order)):
                raise SystemExit("merge_to_data order contains duplicate channel names")
            names = requested_order or default_channel_order(list(ds.data_vars))
            if len(names) == 0:
                raise SystemExit("merge_to_data: no variables to merge")
            missing = [n for n in names if n not in ds.data_vars]
            if missing:
                raise SystemExit(f"merge_to_data: variables not found: {missing}")
            merged = xr.concat([ds[name] for name in names], dim=coord)
            merged = merged.assign_coords({coord: names})
            dims = list(merged.dims)
            if "time" in dims:
                dims.remove("time")
                dims.insert(0, "time")
            if coord in dims:
                dims.remove(coord)
                dims.insert(1, coord)
            ds = merged.transpose(*dims).to_dataset(name="data")
        elif "merge_static" in step:
            cfg = step["merge_static"]
            if not isinstance(cfg, dict):
                raise SystemExit("merge_static must be a mapping")
            coord = str(cfg.get("coord", "channel"))
            output_name = str(cfg.get("name", "const"))
            names = [str(n) for n in cfg.get("order", list(ds.data_vars))]
            if not names:
                raise SystemExit("merge_static: no variables to merge")
            if len(names) != len(set(names)):
                raise SystemExit("merge_static order contains duplicate variable names")
            missing = [name for name in names if name not in ds.data_vars]
            if missing:
                raise SystemExit(f"merge_static: variables not found: {missing}")
            fields = []
            for name in names:
                field = ds[name]
                for time_dim in ("time", "valid_time"):
                    if time_dim in field.dims:
                        if field.sizes[time_dim] != 1:
                            raise SystemExit(
                                f"merge_static: {name!r} has {field.sizes[time_dim]} "
                                f"values on {time_dim!r}; expected exactly one"
                            )
                        field = field.isel({time_dim: 0}, drop=True)
                extra_dims = [dim for dim in field.dims if dim not in ("lat", "lon")]
                if extra_dims:
                    raise SystemExit(f"merge_static: {name!r} has unsupported dims: {extra_dims}")
                if "lat" not in field.dims or "lon" not in field.dims:
                    raise SystemExit(f"merge_static: {name!r} must have lat/lon dimensions")
                fields.append(field.transpose("lat", "lon"))
            merged = xr.concat(fields, dim=coord).assign_coords({coord: names})
            merged = merged.transpose(coord, "lat", "lon").reset_coords(drop=True)
            ds = merged.to_dataset(name=output_name)
        elif "normalize" in step:
            pass  # handled after apply_steps in main (compute stats + write sidecars)
        elif "flatten_step" in step:
            if "data" not in ds.data_vars or "step" not in ds["data"].dims:
                raise SystemExit("flatten_step requires a 'data' variable with a step dim")
            da = ds["data"]
            vt = ds["valid_time"] if "valid_time" in ds.coords else None
            da = da.stack(sample=("time", "step"))
            da = da.reset_index("sample", drop=True)
            if vt is not None:
                da = da.assign_coords(sample=("sample", vt.stack(sample=("time", "step")).values))
            keep = ~np.isnan(da).all(dim=[d for d in da.dims if d != "sample"])
            da = da.isel(sample=keep).rename(sample="time").sortby("time")
            dims = [d for d in ("time", "level", "channel", "latitude", "longitude") if d in da.dims]
            ds = da.transpose(*dims).to_dataset(name="data")
        else:
            raise SystemExit(f"unknown step: {list(step)}")
    return ds


LEVEL_RE = re.compile(r"^([A-Za-z]+)_?(\d+)$")

# Canonical S2S order read from D:\cla.zarr/level. This is an ordering
# default, not a required schema: partial and single-channel datasets remain
# valid, while non-S2S channels are retained after the known channels.
CLA_76_CHANNELS = [
    *[f"z{level}" for level in (1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50)],
    *[f"t{level}" for level in (1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50)],
    *[f"u{level}" for level in (1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50)],
    *[f"v{level}" for level in (1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50)],
    *[f"q{level}" for level in (1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50)],
    "t2m", "d2m", "sst", "ttr", "10u", "10v", "100u", "100v", "msl", "tcwv", "tp",
]
CLA_76_CHANNEL_SET = frozenset(CLA_76_CHANNELS)


def default_channel_order(names: list[str]) -> list[str]:
    """Order known channels like cla.zarr, then retain unknown channels."""
    if len(names) != len(set(names)):
        duplicates = sorted({name for name in names if names.count(name) > 1})
        raise SystemExit(f"duplicate channel names: {duplicates}")
    available = set(names)
    canonical = [name for name in CLA_76_CHANNELS if name in available]
    extras = [name for name in names if name not in CLA_76_CHANNEL_SET]
    return canonical + extras


def compute_channel_stats(ds):
    """Return (channel names, mean, std, channel coord) from the prepared dataset."""
    coord = next((c for c in ("level", "channel") if c in ds.coords), "")
    if coord and "data" in ds.data_vars:
        names = [str(v) for v in ds.coords[coord].values]
        da = ds["data"]
        dims = [d for d in da.dims if d != coord]
        stats = xr.Dataset(
            {"mean": da.mean(dim=dims, skipna=True), "std": da.std(dim=dims, skipna=True)}
        ).compute()
        mean = np.asarray(stats["mean"].values, dtype=np.float32)
        std = np.asarray(stats["std"].values, dtype=np.float32)
        return names, mean, std, coord
    names = list(ds.data_vars)
    stats = xr.Dataset(
        {
            **{f"mean_{i}": ds[name].mean(skipna=True) for i, name in enumerate(names)},
            **{f"std_{i}": ds[name].std(skipna=True) for i, name in enumerate(names)},
        }
    ).compute()
    mean = np.asarray([stats[f"mean_{i}"].item() for i in range(len(names))], dtype=np.float32)
    std = np.asarray([stats[f"std_{i}"].item() for i in range(len(names))], dtype=np.float32)
    return names, mean, std, ""


def channel_moments(ds):
    """Compute mergeable per-channel count/mean/M2 for one prepared batch."""
    coord = next((c for c in ("level", "channel") if c in ds.coords), "")
    if coord and "data" in ds.data_vars:
        names = [str(v) for v in ds.coords[coord].values]
        da = ds["data"]
        dims = [d for d in da.dims if d != coord]
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*invalid value encountered in divide.*")
            warnings.filterwarnings("ignore", message=".*Degrees of freedom <= 0.*")
            reduced = xr.Dataset(
                {
                    "count": da.count(dim=dims),
                    "mean": da.mean(dim=dims, skipna=True),
                    "var": da.var(dim=dims, skipna=True),
                }
            ).compute()
        count = np.asarray(reduced["count"].values, dtype=np.int64)
        mean = np.asarray(reduced["mean"].values, dtype=np.float64)
        variance = np.asarray(reduced["var"].values, dtype=np.float64)
        m2 = np.multiply(variance, count, out=np.zeros_like(variance), where=count > 0)
        mean = np.where(count > 0, mean, 0.0)
        return names, count, mean, m2, coord
    names = list(ds.data_vars)
    count = []
    mean = []
    m2 = []
    for name in names:
        da = ds[name]
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*invalid value encountered in divide.*")
            warnings.filterwarnings("ignore", message=".*Degrees of freedom <= 0.*")
            reduced = xr.Dataset(
                {"count": da.count(), "mean": da.mean(skipna=True), "var": da.var(skipna=True)}
            ).compute()
        n = int(reduced["count"].item())
        avg = float(reduced["mean"].item()) if n else 0.0
        count.append(n)
        mean.append(avg)
        m2.append(float(reduced["var"].item()) * n if n else 0.0)
    return (
        names,
        np.asarray(count, dtype=np.int64),
        np.asarray(mean, dtype=np.float64),
        np.asarray(m2, dtype=np.float64),
        "",
    )


def merge_channel_moments(left, right):
    """Merge two Chan/Welford sufficient-statistic tuples."""
    l_names, l_count, l_mean, l_m2, l_coord = left
    r_names, r_count, r_mean, r_m2, r_coord = right
    if l_names != r_names or l_coord != r_coord:
        raise SystemExit("streaming statistics batches have different channel contracts")
    total = l_count + r_count
    delta = r_mean - l_mean
    ratio = np.divide(r_count, total, out=np.zeros_like(r_mean), where=total > 0)
    mean = l_mean + delta * ratio
    cross = np.divide(
        l_count * r_count,
        total,
        out=np.zeros_like(r_mean),
        where=total > 0,
    )
    m2 = l_m2 + r_m2 + delta * delta * cross
    return l_names, total, mean, m2, l_coord


def finalize_channel_moments(moments):
    """Convert count/mean/M2 into float32 population mean/std sidecars."""
    names, count, mean, m2, coord = moments
    empty = [name for name, value in zip(names, count) if value <= 0]
    invalid = [
        name
        for name, n, avg, total in zip(names, count, mean, m2)
        if n > 0 and (not np.isfinite(avg) or not np.isfinite(total))
    ]
    if empty:
        raise SystemExit(f"normalization statistics have no valid samples for channels: {empty}")
    if invalid:
        raise SystemExit(f"normalization statistics are non-finite for channels: {invalid}")
    variance = np.divide(m2, count, out=np.zeros_like(m2), where=count > 0)
    std = np.sqrt(np.maximum(variance, 0.0))
    if not np.all(np.isfinite(std)):
        bad = [name for name, value in zip(names, std) if not np.isfinite(value)]
        raise SystemExit(f"normalization standard deviations are non-finite for channels: {bad}")
    return names, mean.astype(np.float32), std.astype(np.float32), coord


def steps_for_file_batch(steps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Drop only batch-edge partial windows; overlaps recover owned outputs."""
    adjusted = json.loads(json.dumps(steps))
    for step in adjusted:
        if "daily_aggregation" in step and isinstance(step["daily_aggregation"], dict):
            step["daily_aggregation"]["incomplete"] = "drop"
        if "s2s_daily_accumulation" in step and isinstance(step["s2s_daily_accumulation"], dict):
            step["s2s_daily_accumulation"]["incomplete"] = "drop"
    return adjusted


def input_file_batches(paths: list[Path], size: int, overlap: int = 1):
    """Yield bounded path groups with neighboring files for temporal windows."""
    if size <= 0 or overlap < 0:
        raise SystemExit("input batch size must be positive and overlap non-negative")
    for start in range(0, len(paths), size):
        stop = min(start + size, len(paths))
        group_start = max(0, start - overlap)
        group_stop = min(len(paths), stop + overlap)
        yield paths[group_start:group_stop], paths[start:stop]


def catalog_time_periods(paths: list[Path]):
    """Inspect files one at a time and group files with identical time coverage."""
    periods: dict[tuple[Any, Any], list[Path]] = {}
    for path in paths:
        ds = open_input(path)
        try:
            time_name = next((name for name in ("time", "valid_time") if name in ds.coords), None)
            if time_name is None:
                raise SystemExit(f"streaming multi-file input has no time coordinate: {path}")
            values = np.asarray(ds[time_name].values).reshape(-1)
            if values.size == 0:
                raise SystemExit(f"streaming multi-file input has an empty time coordinate: {path}")
            key = (values.min(), values.max())
            periods.setdefault(key, []).append(path)
        finally:
            ds.close()
    return [(start, end, periods[(start, end)]) for start, end in sorted(periods)]


def period_batches(periods, size: int, overlap: int = 1):
    """Yield bounded chronological period groups and their owned time interval."""
    if size <= 0 or overlap < 0:
        raise SystemExit("input period batch size must be positive and overlap non-negative")
    for start in range(0, len(periods), size):
        stop = min(start + size, len(periods))
        selected = periods[max(0, start - overlap) : min(len(periods), stop + overlap)]
        paths = [path for _, _, group in selected for path in group]
        yield paths, periods[start][0], periods[stop - 1][1]


def prepared_file_batches(
    paths: list[Path],
    *,
    period_batch_size: int,
    overlap_periods: int,
    chunks,
    steps,
):
    """Open, transform, yield, and close only a bounded set of time periods."""
    periods = catalog_time_periods(paths)
    adjusted_steps = steps_for_file_batch(steps)
    last_emitted = None
    for group, _, _ in period_batches(periods, period_batch_size, overlap_periods):
        ds = open_inputs(group, chunks)
        try:
            prepared = canonicalize_latlon(apply_steps(ds, adjusted_steps))
            values = _time_values(prepared)
            mask = np.ones(values.size, dtype=bool) if last_emitted is None else values > last_emitted
            if np.any(mask):
                selected = prepared.isel(time=np.flatnonzero(mask))
                last_emitted = _time_values(selected)[-1]
                yield selected, len(group)
        finally:
            ds.close()


def normalize_ds(ds, mean, std, coord, names):
    """Scale the dataset with (x - mean) / std per channel (std 0 -> 1)."""
    safe_std = np.where(std == 0, 1.0, std)
    if coord and "data" in ds.data_vars:
        m = xr.DataArray(mean, dims=[coord], coords={coord: names})
        s = xr.DataArray(safe_std, dims=[coord], coords={coord: names})
        return ds.assign(data=(ds["data"] - m) / s)
    for i, name in enumerate(names):
        ds[name] = (ds[name] - mean[i]) / safe_std[i]
    return ds


def latitude_weights(ds):
    """Return core-compatible spatial weights ``cos(abs(latitude))``."""
    if "lat" not in ds.coords or ds["lat"].ndim != 1:
        raise SystemExit("weight.nc generation requires a one-dimensional lat coordinate")
    lat = np.asarray(ds["lat"].values, dtype=np.float64)
    if lat.size == 0 or not np.isfinite(lat).all():
        raise SystemExit("weight.nc generation requires finite non-empty latitude values")
    weights = np.maximum(np.cos(np.deg2rad(np.abs(lat))), 0.0).astype(np.float32)
    return lat.astype(np.float32), weights


def write_sidecars(out_dir, names, mean, std, lat, weight, coord_name):
    """Write channel mean/std and a separate latitude-weight sidecar."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dim = coord_name or "channel"
    for key, values in (("mean", mean), ("std", std)):
        da = xr.DataArray(values, dims=[dim], coords={dim: names}, attrs={"long_name": key})
        da.to_netcdf(out_dir / f"{key}.nc")
    xr.DataArray(
        weight,
        dims=["lat"],
        coords={"lat": lat},
        attrs={"long_name": "cosine latitude area weight", "formula": "cos(abs(lat))"},
    ).to_netcdf(out_dir / "weight.nc")


def describe(ds) -> dict[str, Any]:
    return {
        "dims": dict(ds.sizes),
        "variables": list(ds.data_vars),
        "coords": list(ds.coords),
        "units": {str(name): ds[name].attrs.get("units") for name in ds.data_vars},
    }


def run_guard(input_paths: list[Path], output_path: Path, overwrite: bool, resume: bool = False) -> None:
    guard = Path(__file__).resolve().parent / "zarr_write_guard.py"
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


def write_incremental_zarr(
    ds,
    output_path: Path,
    *,
    batch_time: int,
    resume: bool,
    overwrite: bool = False,
    state_path: Path,
    state: dict[str, Any],
) -> None:
    """Write time batches, deriving recovery progress from the store itself."""
    if batch_time <= 0:
        raise SystemExit("--batch-time must be a positive integer")
    total = _time_values(ds).size
    start = 0
    if resume:
        if output_path.exists():
            try:
                existing = xr.open_zarr(output_path, consolidated=False)
                try:
                    start = validate_resume_prefix(ds, existing)
                finally:
                    existing.close()
            except SystemExit:
                raise
            except Exception as exc:
                raise SystemExit(f"resume refused: existing Zarr cannot be opened safely: {exc}") from exc
        else:
            # A guarded run may fail during statistics before the store exists.
            # In that state resume means recompute statistics and create the store.
            start = 0
    elif output_path.exists() and not overwrite:
        raise SystemExit(f"output already exists: {output_path}; use --resume or approved --overwrite")

    state.update({"status": "running", "total_time_steps": int(total), "completed_time_steps": start})
    write_conversion_state(state_path, state)
    try:
        for offset in range(start, total, batch_time):
            stop = min(offset + batch_time, total)
            batch = ds.isel(time=slice(offset, stop))
            if offset == 0:
                batch.to_zarr(str(output_path), mode="w" if overwrite else "w-", consolidated=False)
            else:
                batch.to_zarr(str(output_path), mode="a", append_dim="time", consolidated=False)
            state["completed_time_steps"] = stop
            state["last_time"] = str(np.asarray(ds.time.values)[stop - 1])
            write_conversion_state(state_path, state)
            print(f"batch: {offset}:{stop} / {total}")
        try:
            import zarr

            zarr.consolidate_metadata(str(output_path))
        except Exception as exc:
            raise SystemExit(f"data batches were written but metadata consolidation failed: {exc}") from exc
    except BaseException as exc:
        state["status"] = "failed"
        state["error"] = f"{type(exc).__name__}: {exc}"
        write_conversion_state(state_path, state)
        raise
    state.update({"status": "completed", "completed_time_steps": int(total)})
    state.pop("error", None)
    write_conversion_state(state_path, state)


def streaming_channel_stats(batch_factory, state_path: Path | None = None, state=None):
    """Compute global channel statistics by merging bounded prepared batches."""
    merged = None
    batches = 0
    try:
        if state_path is not None and state is not None:
            state.update({"status": "running", "phase": "statistics", "statistics_batches": 0})
            write_conversion_state(state_path, state)
        for prepared, _ in batch_factory():
            current = channel_moments(prepared)
            merged = current if merged is None else merge_channel_moments(merged, current)
            batches += 1
            if state_path is not None and state is not None:
                names, count, mean, m2, coord = merged
                state.update(
                    {
                        "statistics_batches": batches,
                        "statistics": {
                            "names": names,
                            "coord": coord,
                            "count": count.tolist(),
                            "mean": mean.tolist(),
                            "m2": m2.tolist(),
                        },
                    }
                )
                write_conversion_state(state_path, state)
    except BaseException as exc:
        if state_path is not None and state is not None:
            state.update({"status": "failed", "error": f"{type(exc).__name__}: {exc}"})
            write_conversion_state(state_path, state)
        raise
    if merged is None:
        raise SystemExit("streaming statistics found no prepared time batches")
    return finalize_channel_moments(merged)


def validate_append_contract(planned, existing) -> None:
    """Validate non-time schema before appending a prepared file batch."""
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


def write_streaming_file_batches(
    batch_factory,
    output_path: Path,
    *,
    batch_time: int,
    resume: bool,
    overwrite: bool,
    output_chunks,
    state_path: Path,
    state: dict[str, Any],
    normalization=None,
) -> None:
    """Append transformed file batches while verifying the complete resume prefix."""
    existing = None
    existing_times = np.asarray([], dtype="datetime64[ns]")
    cursor = 0
    if resume and output_path.exists():
        try:
            existing = xr.open_zarr(output_path, consolidated=False)
            existing_times = _time_values(existing)
        except Exception as exc:
            state.update({"status": "failed", "phase": "write", "error": f"{type(exc).__name__}: {exc}"})
            write_conversion_state(state_path, state)
            raise SystemExit(f"resume refused: existing Zarr cannot be opened safely: {exc}") from exc
    created = existing is not None
    completed = int(existing_times.size)
    state.update({"status": "running", "phase": "write", "completed_time_steps": completed})
    write_conversion_state(state_path, state)
    try:
        for prepared, open_count in batch_factory():
            if normalization is not None:
                names, mean, std, coord = normalization
                prepared = normalize_ds(prepared, mean, std, coord, names)
            if output_chunks:
                applicable = {name: size for name, size in output_chunks.items() if name in prepared.dims}
                prepared = prepared.chunk(applicable)
            times = _time_values(prepared)
            if existing is not None:
                validate_append_contract(prepared, existing)
            prefix_count = min(max(existing_times.size - cursor, 0), times.size)
            if prefix_count:
                if not np.array_equal(times[:prefix_count], existing_times[cursor : cursor + prefix_count]):
                    raise SystemExit(
                        "resume refused: existing Zarr times are not the exact prefix produced by file batches"
                    )
                cursor += prefix_count
                prepared = prepared.isel(time=slice(prefix_count, None))
            if prepared.sizes.get("time", 0) == 0:
                continue
            if cursor < existing_times.size:
                raise SystemExit("resume refused: planned file batches skip existing output times")
            for offset in range(0, prepared.sizes["time"], batch_time):
                stop = min(offset + batch_time, prepared.sizes["time"])
                piece = prepared.isel(time=slice(offset, stop))
                if not created:
                    piece.to_zarr(
                        str(output_path), mode="w" if overwrite else "w-", consolidated=False
                    )
                    created = True
                else:
                    piece.to_zarr(str(output_path), mode="a", append_dim="time", consolidated=False)
                completed += piece.sizes["time"]
                state.update(
                    {
                        "completed_time_steps": completed,
                        "last_time": str(np.asarray(piece.time.values)[-1]),
                        "max_open_input_files": max(state.get("max_open_input_files", 0), open_count),
                    }
                )
                write_conversion_state(state_path, state)
                print(f"stream batch: +{piece.sizes['time']} -> {completed} (input files open: {open_count})")
        if resume and cursor != existing_times.size:
            raise SystemExit("resume refused: planned input ended before the existing Zarr prefix")
        if not created:
            raise SystemExit("streaming conversion produced no output time steps")
        import zarr

        zarr.consolidate_metadata(str(output_path))
    except BaseException as exc:
        state.update({"status": "failed", "error": f"{type(exc).__name__}: {exc}"})
        write_conversion_state(state_path, state)
        raise
    finally:
        if existing is not None:
            existing.close()
    state.update({"status": "completed", "phase": "complete", "completed_time_steps": completed})
    state.pop("error", None)
    write_conversion_state(state_path, state)


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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", action="append", default=[], help="input path; repeat for multiple NetCDF files")
    parser.add_argument("--input-glob", default=None, help="glob for a homogeneous NetCDF collection")
    parser.add_argument("--output", required=True, help="target Zarr store or static NetCDF path")
    parser.add_argument(
        "--output-format",
        choices=("zarr", "static-netcdf"),
        default="zarr",
        help="write the main Zarr store (default) or a core-compatible const.nc",
    )
    parser.add_argument("--steps-config", default=None, help="JSON/YAML steps config")
    parser.add_argument("--input-chunks", default="time=4", help="lazy input chunks, e.g. time=4")
    parser.add_argument(
        "--output-chunks",
        default="time=1,level=-1,channel=-1,lat=-1,lon=-1",
        help="output chunks; -1 means the complete dimension",
    )
    parser.add_argument(
        "--batch-time",
        type=int,
        default=31,
        help="number of output time steps per Zarr append batch (default: 31)",
    )
    parser.add_argument(
        "--input-period-batch",
        type=int,
        default=1,
        help="number of distinct source time periods opened per batch (default: 1)",
    )
    parser.add_argument(
        "--input-overlap-periods",
        type=int,
        default=1,
        help="neighboring source periods included for cross-boundary transforms (default: 1)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="continue an interrupted incremental write when existing times are an exact prefix",
    )
    parser.add_argument("--allow-write", action="store_true", help="confirm the write after guard approval")
    parser.add_argument("--overwrite", action="store_true", help="confirm replacing an existing output store")
    parser.add_argument("--ack-risk", default=None, help=f"must equal: {GUARD_ACK}")
    args = parser.parse_args(argv)

    input_paths = resolve_inputs(args.input, args.input_glob)
    input_path = input_paths[0]
    output_path = Path(args.output).expanduser()
    input_chunks = parse_chunks(args.input_chunks)
    output_chunks = parse_chunks(args.output_chunks)

    if args.resume and args.overwrite:
        raise SystemExit("--resume and --overwrite are mutually exclusive")
    if args.resume and args.output_format != "zarr":
        raise SystemExit("--resume is supported only for Zarr output")
    if args.batch_time <= 0:
        raise SystemExit("--batch-time must be a positive integer")
    if args.input_period_batch <= 0 or args.input_overlap_periods < 0:
        raise SystemExit("input period batch must be positive and overlap must be non-negative")

    steps = load_steps_config(args.steps_config)
    do_normalize = any("normalize" in step for step in steps)
    do_merge_static = any("merge_static" in step for step in steps)
    if args.output_format == "static-netcdf":
        if not do_merge_static:
            raise SystemExit("static-netcdf output requires a merge_static step")
        if do_normalize:
            raise SystemExit("static-netcdf output does not support normalize")
    elif do_merge_static:
        raise SystemExit("merge_static requires --output-format static-netcdf")
    stream_mode = (
        args.output_format == "zarr"
        and len(input_paths) > 1
        and all(format_label(path) == "netcdf" for path in input_paths)
    )
    try:
        if stream_mode:
            periods = catalog_time_periods(input_paths)
            preview_paths, _, _ = next(period_batches(periods, args.input_period_batch, args.input_overlap_periods))
            ds = open_inputs(preview_paths, input_chunks)
        else:
            ds = open_inputs(input_paths, input_chunks)
    except Exception as exc:
        if format_label(input_path) == "grib":
            raise SystemExit(
                f"GRIB decode failed: {type(exc).__name__}: {exc}\n"
                "Hint: multi-variable GRIB may mix GRIB editions; the automatic "
                "per-variable fallback also failed. Prefer NetCDF for multi-variable "
                "downloads or split the GRIB by variable "
                "(see references/data-preprocessing.md)."
            ) from exc
        raise
    try:
        before = describe(ds)
        ds = apply_steps(ds, steps_for_file_batch(steps) if stream_mode else steps)
        ds = canonicalize_latlon(ds)
        if args.output_format == "static-netcdf":
            static_dataarray(ds)
        after = describe(ds)
    finally:
        ds.close()

    print(f"inputs: {len(input_paths)} ({input_paths[0]}{f' ... {input_paths[-1]}' if len(input_paths) > 1 else ''})")
    print(f"format: {format_label(input_path)}")
    print(f"before: {before['dims']} vars={before['variables']}")
    if steps:
        print(f"steps : {len(steps)} (see config: {args.steps_config})")
    print(f"after : {after['dims']} vars={after['variables']}")
    print(f"output: {output_path}")
    print(f"output format: {args.output_format}")
    print(f"chunks: input={input_chunks or 'backend'} output={output_chunks or 'backend'}")
    if args.output_format == "zarr":
        print(f"incremental: batch_time={args.batch_time} resume={args.resume}")
    if stream_mode:
        print(
            f"file streaming: periods_per_batch={args.input_period_batch} "
            f"overlap_periods={args.input_overlap_periods}; preview only opened {len(preview_paths)} files"
        )
    if do_normalize:
        print("normalize: statistics deferred until the guarded write phase")

    if not args.allow_write:
        print("\nDRY-RUN: nothing was written.")
        print("To execute, pass --allow-write --ack-risk I-understand-this-mutates-zarr")
        return 0

    if args.ack_risk != GUARD_ACK:
        raise SystemExit(f"Refusing write: --ack-risk must equal {GUARD_ACK}")

    if args.output_format == "zarr":
        run_guard(input_paths, output_path, args.overwrite, args.resume)
    else:
        guard_static_output(input_paths, output_path, args.overwrite)

    execution = {
        "input_period_batch": args.input_period_batch,
        "input_overlap_periods": args.input_overlap_periods,
    }
    fingerprint = conversion_fingerprint(input_paths, steps, input_chunks, output_chunks, execution)
    state_path = conversion_state_path(output_path)
    if args.resume:
        if not state_path.exists():
            raise SystemExit(f"resume refused: conversion state is missing: {state_path}")
        previous = json.loads(state_path.read_text(encoding="utf-8"))
        if previous.get("contract_sha256") != fingerprint["contract_sha256"]:
            raise SystemExit("resume refused: transformation, chunk, or batching configuration changed")
        if do_normalize and previous.get("inputs") != fingerprint["inputs"]:
            raise SystemExit(
                "resume refused: normalized incremental output requires the identical input set "
                "because its global mean/std must not change"
            )
    state = {**fingerprint, "output": str(output_path.resolve())}

    if stream_mode:
        def batch_factory():
            return prepared_file_batches(
                input_paths,
                period_batch_size=args.input_period_batch,
                overlap_periods=args.input_overlap_periods,
                chunks=input_chunks,
                steps=steps,
            )

        normalization = None
        if do_normalize:
            normalization = streaming_channel_stats(batch_factory, state_path, state)
            names, mean, std, coord = normalization
            preview, _ = next(batch_factory())
            lat, weight = latitude_weights(preview)
        write_streaming_file_batches(
            batch_factory,
            output_path,
            batch_time=args.batch_time,
            resume=args.resume,
            overwrite=args.overwrite,
            output_chunks=output_chunks,
            state_path=state_path,
            state=state,
            normalization=normalization,
        )
        if do_normalize:
            write_sidecars(output_path, names, mean, std, lat, weight, coord or "channel")
            print(f"sidecars: mean.nc / std.nc / weight.nc -> {output_path}")
        print(f"\nWROTE {output_path}")
        return 0

    # Guard passed; reopen the single dataset for the actual write.
    ds = open_inputs(input_paths, input_chunks)
    try:
        ds = apply_steps(ds, steps)
        ds = canonicalize_latlon(ds)
        if do_normalize:
            names, mean, std, coord = compute_channel_stats(ds)
            lat, weight = latitude_weights(ds)
            ds = normalize_ds(ds, mean, std, coord, names)
        if output_chunks and args.output_format == "zarr":
            applicable_chunks = {name: size for name, size in output_chunks.items() if name in ds.dims}
            ds = ds.chunk(applicable_chunks)
        if args.output_format == "zarr":
            write_incremental_zarr(
                ds,
                output_path,
                batch_time=args.batch_time,
                resume=args.resume,
                overwrite=args.overwrite,
                state_path=state_path,
                state=state,
            )
            if do_normalize:
                write_sidecars(output_path, names, mean, std, lat, weight, coord or "channel")
                print(f"sidecars: mean.nc / std.nc / weight.nc -> {output_path}")
        else:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            static_dataarray(ds).to_netcdf(output_path, mode="w")
    finally:
        ds.close()
    print(f"\nWROTE {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

