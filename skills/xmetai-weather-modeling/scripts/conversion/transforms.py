"""Config loading and ordered dataset transformations."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np

try:
    import xarray as xr
except ImportError:  # pragma: no cover
    xr = None

from .grid import regrid_dataset
from .temporal import daily_aggregation, s2s_daily_accumulation


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

LEVEL_RE = re.compile(r"^([A-Za-z]+)_?(\d+)$")

CLA_76_CHANNELS = [
    f"{var}{level}"
    for var in ("z", "t", "u", "v", "q")
    for level in (1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50)
] + ["msl", "u10", "v10", "t2m", "d2m", "skt", "sp", "tcw", "tcc", "ttr", "tp"]
CLA_76_CHANNEL_SET = frozenset(CLA_76_CHANNELS)

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


def default_channel_order(names: list[str]) -> list[str]:
    """Order known channels like cla.zarr, then retain unknown channels."""
    if len(names) != len(set(names)):
        duplicates = sorted({name for name in names if names.count(name) > 1})
        raise SystemExit(f"duplicate channel names: {duplicates}")
    available = set(names)
    canonical = [name for name in CLA_76_CHANNELS if name in available]
    extras = [name for name in names if name not in CLA_76_CHANNEL_SET]
    return canonical + extras
