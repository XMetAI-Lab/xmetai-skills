# -*- coding: utf-8 -*-
"""两个空间分布取数脚本共用：常量、坐标归一、月内起报日、ERA5。"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import xarray as xr

from _bootstrap import bootstrap

STEP_HOURS = 6
TARGET_RES = 1.0                       # 1° → 181×360
GRAVITY = 9.80665
Z500_PHYS_THRESHOLD = 1e4              # 大于此值说明还是 m²/s²

ERA5_PL_STORE = (
    "/workspace/data/liujunjie/era5_foundation_store2/"
    "era5_pl_2025.01-2026.07.c84.p25.h6.zarr"
)
ERA5_VARIABLES = ("z500", "u850", "v850")


def find_coord(dataset: xr.Dataset, names) -> str:
    for name in names:
        if name in dataset.coords or name in dataset.dims:
            return name
    raise KeyError(
        f"找不到坐标 {names}，文件里有 {list(dataset.coords)} / {list(dataset.dims)}"
    )


def get_var(dataset: xr.Dataset, name: str):
    for key in dataset.data_vars:
        if str(key).lower() == name:
            return dataset[key]
    return None


def to_latlon(field, lat_name: str, lon_name: str):
    """统一成 lat 升序、lon 0–360 升序的二维场。"""
    field = field.squeeze()
    if field.ndim != 2:
        raise ValueError(f"变量不是二维场，维度是 {field.dims}")
    renames = {
        source: target
        for source, target in ((lat_name, "lat"), (lon_name, "lon"))
        if source != target
    }
    if renames:
        field = field.rename(renames)
    if float(field.lon.min()) < 0.0:
        field = field.assign_coords(lon=(field.lon % 360.0))
    return field.sortby("lat").sortby("lon")


def z500_to_gpm(field):
    if float(np.nanmedian(np.abs(field.values if hasattr(field, "values") else field))) \
            > Z500_PHYS_THRESHOLD:
        return field / GRAVITY
    return field


def month_days(root: Path, month: str) -> List[Path]:
    return [
        path for path in sorted(root.iterdir())
        if path.is_dir()
        and len(path.name) == 8
        and path.name.isdigit()
        and f"{path.name[:4]}-{path.name[4:6]}" == month
    ]


def target_grid() -> Tuple[np.ndarray, np.ndarray]:
    lat = np.arange(-90.0, 90.0 + 1e-9, TARGET_RES)
    lon = np.arange(0.0, 360.0 - 1e-9, TARGET_RES)
    return lat, lon


def read_era5(days, leads, lat_target, lon_target) -> Dict[Tuple[str, int], np.ndarray]:
    """同一批有效时刻的 ERA5 月平均。"""
    bootstrap()
    from reporting.sources.contracts import DataIndex, DataRequest
    from reporting.sources.era5 import Era5ZarrReader

    lead_times = {
        lead: [
            datetime.strptime(day.name, "%Y%m%d") + timedelta(hours=lead)
            for day in days
        ]
        for lead in leads
    }
    union = sorted({moment for times in lead_times.values() for moment in times})
    reader = Era5ZarrReader("era5", {"pl": ERA5_PL_STORE})
    request = DataRequest(
        source_id="era5", variables=list(ERA5_VARIABLES), init_times=union,
    )
    index = DataIndex(source_id="era5", available=[{"pl": ERA5_PL_STORE}])
    dataset = reader.read(request, index).payload
    print(f"    ERA5: {len(union)} 个有效时刻，网格 "
          f"{dataset.sizes['lat']}×{dataset.sizes['lon']}")

    fields = {}
    for lead, times in lead_times.items():
        subset = dataset.sel(valid_time=times)
        z = to_latlon(z500_to_gpm(subset["z500"].mean("valid_time")), "lat", "lon")
        # 风速逐时次合成后再平均
        ws = np.sqrt(subset["u850"] ** 2 + subset["v850"] ** 2).mean("valid_time")
        ws = to_latlon(ws, "lat", "lon")
        fields[("z500", lead)] = z.interp(lat=lat_target, lon=lon_target).values
        fields[("ws850", lead)] = ws.interp(lat=lat_target, lon=lon_target).values
    return fields


def grid_mean(accumulator: np.ndarray, count: np.ndarray, fill: float = np.nan) -> np.ndarray:
    return np.divide(
        accumulator, count,
        out=np.full_like(accumulator, fill),
        where=count > 0,
    )


def write_frame(columns: Dict[str, np.ndarray], lat_target: np.ndarray,
                lon_target: np.ndarray, out: Path) -> None:
    """宽表落盘：``lat`` 北 → 南、行主序。"""
    import pandas as pd

    columns = dict(columns)
    columns["lat"] = np.repeat(lat_target[::-1], lon_target.size)
    columns["lon"] = np.tile(lon_target, lat_target.size)
    frame = pd.DataFrame(columns)
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out, index=False, float_format="%.3f")
    size_mb = out.stat().st_size / 1024 / 1024
    print(f"\n写出 {out}：{len(frame):,} 行 × {len(frame.columns)} 列，{size_mb:.1f} MB")
