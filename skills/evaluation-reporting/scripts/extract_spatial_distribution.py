#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""空间分布取数（确定性）：月平均 × 时效 → 宽表 CSV。

档案布局 ``<ROOT>/<YYYYMMDD>/<NNN>.nc``，``NNN`` 是 6h 步长序号（``004`` = +24h）。

    python extract_spatial_distribution.py --root <预测结果根> \
        --months 2025-07,2025-11 --out spatial_monthly.csv

列：``lat, lon, {var}_{YYYYMM}_h{lead}, {var}_era5_{YYYYMM}_h{lead}``。
集合档案用 ``extract_spatial_distribution_ens.py``。
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import xarray as xr

from _bootstrap import bootstrap
from _spatial_common import (
    ERA5_PL_STORE,
    STEP_HOURS,
    find_coord,
    get_var,
    grid_mean,
    month_days,
    read_era5,
    target_grid,
    to_latlon,
    write_frame,
    z500_to_gpm,
)


def read_forecasts(days, leads, lat_target, lon_target):
    acc, cnt = {}, {}
    for name in ("z500", "ws850"):
        for lead in leads:
            acc[(name, lead)] = np.zeros((lat_target.size, lon_target.size))
            cnt[(name, lead)] = np.zeros_like(acc[(name, lead)])
    missing = 0
    for lead in leads:
        used = 0
        for day in days:
            path = day / f"{int(lead // STEP_HOURS):03d}.nc"
            if not path.is_file():
                missing += 1
                continue
            with xr.open_dataset(path) as dataset:
                lat_name = find_coord(dataset, ("latitude", "lat"))
                lon_name = find_coord(dataset, ("longitude", "lon"))

                z = get_var(dataset, "z500")
                if z is not None:
                    z = z500_to_gpm(to_latlon(z, lat_name, lon_name))
                    z = z.interp(lat=lat_target, lon=lon_target).values
                    acc[("z500", lead)] += np.nan_to_num(z)
                    cnt[("z500", lead)] += np.isfinite(z)

                u, v = get_var(dataset, "u850"), get_var(dataset, "v850")
                if u is not None and v is not None:
                    ws = np.sqrt(
                        to_latlon(u, lat_name, lon_name) ** 2
                        + to_latlon(v, lat_name, lon_name) ** 2
                    ).interp(lat=lat_target, lon=lon_target).values
                    acc[("ws850", lead)] += np.nan_to_num(ws)
                    cnt[("ws850", lead)] += np.isfinite(ws)
            used += 1
        print(f"    +{lead:03d}h: 读入 {used} 个起报日")
    return acc, cnt, missing


def main(argv=None) -> int:
    bootstrap()
    parser = argparse.ArgumentParser(description="月平均 × 时效 → 空间分布宽表 CSV")
    parser.add_argument("--root", type=Path, required=True, help="预报档案根目录")
    parser.add_argument("--months", required=True, help="起报月，逗号分隔，如 2025-07,2025-11")
    parser.add_argument("--leads", default="24,48,72,120", help="时效（小时），逗号分隔")
    parser.add_argument("--out", type=Path, default=Path("spatial_monthly.csv"))
    args = parser.parse_args(argv)

    months = [item.strip() for item in args.months.split(",") if item.strip()]
    leads = tuple(int(item) for item in args.leads.split(",") if item.strip())
    if not args.root.is_dir():
        sys.exit(f"预报目录不存在：{args.root}")
    if not Path(ERA5_PL_STORE).is_dir():
        sys.exit(f"ERA5 zarr store 不存在：{ERA5_PL_STORE}")

    lat_target, lon_target = target_grid()

    month_days_map = {}
    for month in months:
        days = month_days(args.root, month)
        if not days:
            sys.exit(f"{month} 在 {args.root} 下一个起报日都没有")
        month_days_map[month] = days
        print(f"{month}: {len(days)} 个起报日 "
              f"({days[0].name} … {days[-1].name})")

    columns = {}
    skipped = 0
    for month in months:
        print(f"  预报 {month}:")
        acc, cnt, missing = read_forecasts(
            month_days_map[month], leads, lat_target, lon_target
        )
        skipped += missing
        print(f"  ERA5 {month}:")
        for (name, lead), values in read_era5(
            month_days_map[month], leads, lat_target, lon_target
        ).items():
            acc[(name, lead)] = np.nan_to_num(values)
            cnt[(name, lead)] = np.isfinite(values)

        tag = month.replace("-", "")
        for (name, lead), count in cnt.items():
            mean = grid_mean(acc[(name, lead)], count)
            columns[f"{name}_{tag}_h{lead:03d}"] = mean[::-1].ravel()
            valid = int((count > 0).sum())
            print(f"    {name} +{lead:03d}h: {valid} 个格点有值 "
                  f"({valid / count.size:.1%})")

    if skipped:
        print(f"警告：{skipped} 个预报文件不存在，已跳过")

    # 自检：时效间必须不同，否则是累加器没分开
    tag = months[0].replace("-", "")
    for name in ("ws850", "z500", "ws850_era5", "z500_era5"):
        first = columns[f"{name}_{tag}_h{leads[0]:03d}"]
        last = columns[f"{name}_{tag}_h{leads[-1]:03d}"]
        if np.allclose(first, last, equal_nan=True):
            sys.exit(f"自检失败：{name} 各时效的场完全相同")

    write_frame(columns, lat_target, lon_target, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
