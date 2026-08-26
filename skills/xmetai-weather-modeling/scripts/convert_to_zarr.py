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
        {"regrid": {"target": "s2s_1.5deg", "method": "linear",
                    "variable_methods": {"lsm": "nearest"}}},
        {"merge_static": {"order": ["z", "lsm"], "coord": "channel",
                          "name": "const"}},
        {"units": {"q": 1000, "tp": 1000, "ttr": 1 / 3600}},
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
per-channel mean/std (and level-scaled weights) from the prepared data,
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
it by 1000 (``"tp": 1000``); rate-form precipitation (``kg m-2 s-1`` or
``mm/h`` averages) must first be multiplied by the accumulation length in
seconds. The accumulation window must follow the selected model contract
(for example daily totals for S2S, 6-hourly for IWC).
"""

from __future__ import annotations

import argparse
import glob
import json
import re
import subprocess
import sys
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
    return xr.open_dataset(path)


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


def channel_weights(names, land=(), ocean=()):
    """Level-scaled channel weights with optional land/ocean corrections (max 1)."""
    weights = np.ones(len(names), dtype=np.float32)
    for i, name in enumerate(names):
        match = LEVEL_RE.match(name)
        if match is not None:
            weights[i] = max(0.2, int(match.group(2)) / 1000.0)
    for i, name in enumerate(names):
        if name in land:
            weights[i] *= 0.33
        elif name in ocean:
            weights[i] *= 0.67
    weights /= weights.max()
    return weights


def normalize_land_ocean(steps):
    """Extract optional land/ocean channel names from a normalize step."""
    land, ocean = [], []
    for step in steps:
        if "normalize" in step and isinstance(step["normalize"], dict):
            land = step["normalize"].get("land_names", [])
            ocean = step["normalize"].get("ocean_names", [])
    return land, ocean


def write_sidecars(out_dir, names, mean, std, weight, coord_name):
    """Write mean/std/weight.nc next to the output Zarr (core reads from data dir)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dim = coord_name or "channel"
    for key, values in (("mean", mean), ("std", std), ("weight", weight)):
        da = xr.DataArray(values, dims=[dim], coords={dim: names}, attrs={"long_name": key})
        da.to_netcdf(out_dir / f"{key}.nc")


def describe(ds) -> dict[str, Any]:
    return {
        "dims": dict(ds.sizes),
        "variables": list(ds.data_vars),
        "coords": list(ds.coords),
        "units": {str(name): ds[name].attrs.get("units") for name in ds.data_vars},
    }


def run_guard(input_paths: list[Path], output_path: Path, overwrite: bool) -> None:
    guard = Path(__file__).resolve().parent / "zarr_write_guard.py"
    cmd = [
        sys.executable,
        str(guard),
        "--operation",
        "convert",
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
    parser.add_argument("--allow-write", action="store_true", help="confirm the write after guard approval")
    parser.add_argument("--overwrite", action="store_true", help="confirm replacing an existing output store")
    parser.add_argument("--ack-risk", default=None, help=f"must equal: {GUARD_ACK}")
    args = parser.parse_args(argv)

    input_paths = resolve_inputs(args.input, args.input_glob)
    input_path = input_paths[0]
    output_path = Path(args.output).expanduser()
    input_chunks = parse_chunks(args.input_chunks)
    output_chunks = parse_chunks(args.output_chunks)

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
    try:
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
        ds = apply_steps(ds, steps)
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
    if do_normalize:
        print("normalize: statistics deferred until the guarded write phase")

    if not args.allow_write:
        print("\nDRY-RUN: nothing was written.")
        print("To execute, pass --allow-write --ack-risk I-understand-this-mutates-zarr")
        return 0

    if args.ack_risk != GUARD_ACK:
        raise SystemExit(f"Refusing write: --ack-risk must equal {GUARD_ACK}")

    if args.output_format == "zarr":
        run_guard(input_paths, output_path, args.overwrite)
    else:
        guard_static_output(input_paths, output_path, args.overwrite)

    # Guard passed; reopen the dataset for the actual write.
    ds = open_inputs(input_paths, input_chunks)
    try:
        ds = apply_steps(ds, steps)
        ds = canonicalize_latlon(ds)
        if do_normalize:
            names, mean, std, coord = compute_channel_stats(ds)
            land_names, ocean_names = normalize_land_ocean(steps)
            weight = channel_weights(names, set(land_names), set(ocean_names))
            ds = normalize_ds(ds, mean, std, coord, names)
        if output_chunks and args.output_format == "zarr":
            applicable_chunks = {name: size for name, size in output_chunks.items() if name in ds.dims}
            ds = ds.chunk(applicable_chunks)
        if args.output_format == "zarr":
            ds.to_zarr(str(output_path), mode="w" if args.overwrite else "w-", consolidated=True)
            if do_normalize:
                write_sidecars(output_path, names, mean, std, weight, coord or "channel")
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

