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
        {"resample": {"freq": "6h", "operator": "mean"}},
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
            names = [str(n) for n in cfg.get("order", [])] or list(ds.data_vars)
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


def compute_channel_stats(ds):
    """Return (channel names, mean, std, channel coord) from the prepared dataset."""
    coord = next((c for c in ("level", "channel") if c in ds.coords), "")
    if coord and "data" in ds.data_vars:
        names = [str(v) for v in ds.coords[coord].values]
        da = ds["data"]
        mean = np.empty(len(names), dtype=np.float32)
        std = np.empty(len(names), dtype=np.float32)
        for i, name in enumerate(names):
            sub = da.sel({coord: name})
            dims = [d for d in sub.dims if d != coord]
            mean[i] = float(sub.mean(dim=dims, skipna=True).values)
            std[i] = float(sub.std(dim=dims, skipna=True).values)
        return names, mean, std, coord
    names = list(ds.data_vars)
    mean = np.empty(len(names), dtype=np.float32)
    std = np.empty(len(names), dtype=np.float32)
    for i, name in enumerate(names):
        da = ds[name]
        mean[i] = float(da.mean(skipna=True).values)
        std[i] = float(da.std(skipna=True).values)
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
    do_normalize = any("normalize" in step for step in steps)
    try:
        ds = open_input(input_path)
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
        if do_normalize:
            names, mean, std, coord = compute_channel_stats(ds)
            land_names, ocean_names = normalize_land_ocean(steps)
            weight = channel_weights(names, set(land_names), set(ocean_names))
            ds = normalize_ds(ds, mean, std, coord, names)
            preview = ", ".join(f"{n}={m:.3g}/{s:.3g}" for n, m, s in list(zip(names, mean, std))[:3])
            print(f"normalize: {len(names)} channels (e.g. {preview} ...); sidecars written on --allow-write")
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
        ds = canonicalize_latlon(ds)
        if do_normalize:
            names, mean, std, coord = compute_channel_stats(ds)
            land_names, ocean_names = normalize_land_ocean(steps)
            weight = channel_weights(names, set(land_names), set(ocean_names))
            ds = normalize_ds(ds, mean, std, coord, names)
        ds.to_zarr(str(output_path), mode="w" if args.overwrite else "w-", consolidated=True)
        if do_normalize:
            write_sidecars(output_path, names, mean, std, weight, coord or "channel")
            print(f"sidecars: mean.nc / std.nc / weight.nc -> {output_path}")
    finally:
        ds.close()
    print(f"\nWROTE {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

