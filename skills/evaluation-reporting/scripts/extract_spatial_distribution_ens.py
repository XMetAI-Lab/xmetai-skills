#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""空间分布取数（集合）：月平均 × 时效 → 宽表 CSV。

档案布局 ``<ROOT>/<YYYYMMDD>/member_<NNN>/<NNN>.nc``（``004`` = +24h）。

    python extract_spatial_distribution_ens.py --root <预测结果根> \
        --months 2025-08,2025-09 --out spatial_monthly_ens.csv

离散度 = 成员月平均场的标准差（``ddof=0``），不是逐时次集合 std 再月平均。

列：``lat, lon, {var}_{YYYYMM}_h{lead}, {var}_spread_{YYYYMM}_h{lead},
{var}_era5_{YYYYMM}_h{lead}``。确定性档案用 ``extract_spatial_distribution.py``。
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


def member_dirs(day: Path) -> dict:
    found = {}
    for path in day.iterdir():
        if not path.is_dir() or not path.name.startswith("member_"):
            continue
        digits = path.name[len("member_"):]
        if digits.isdigit():
            found[int(digits)] = path
    return found


def read_ensemble_lead(entries, members, lead, lat_target, lon_target):
    """一个时效：逐成员月平均 → 成员均值与离散度。"""
    step = int(lead // STEP_HOURS)
    shape = (lat_target.size, lon_target.size)
    acc = {(member, name): np.zeros(shape) for member in members
           for name in ("z500", "ws850")}
    cnt = {(member, name): np.zeros(shape) for member in members
           for name in ("z500", "ws850")}
    day_count = {member: 0 for member in members}
    missing = 0

    for day, dirs in entries:
        for member, directory in dirs.items():
            if member not in day_count:
                continue
            path = directory / f"{step:03d}.nc"
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
                    acc[(member, "z500")] += np.nan_to_num(z)
                    cnt[(member, "z500")] += np.isfinite(z)

                u, v = get_var(dataset, "u850"), get_var(dataset, "v850")
                if u is not None and v is not None:
                    ws = np.sqrt(
                        to_latlon(u, lat_name, lon_name) ** 2
                        + to_latlon(v, lat_name, lon_name) ** 2
                    ).interp(lat=lat_target, lon=lon_target).values
                    acc[(member, "ws850")] += np.nan_to_num(ws)
                    cnt[(member, "ws850")] += np.isfinite(ws)
            day_count[member] += 1

    result = {}
    for name in ("z500", "ws850"):
        stack = np.full((len(members), *shape), np.nan, dtype=float)
        for index, member in enumerate(members):
            stack[index] = grid_mean(acc[(member, name)], cnt[(member, name)])

        finite = np.isfinite(stack)
        count = finite.sum(axis=0)
        total = np.nansum(stack, axis=0)
        mean = np.divide(total, count,
                         out=np.full_like(total, np.nan), where=count > 0)
        squares = np.nansum(np.where(finite, stack, 0.0) ** 2, axis=0)
        variance = np.divide(squares, count,
                             out=np.full_like(squares, np.nan),
                             where=count > 0) - mean ** 2
        spread = np.sqrt(np.maximum(variance, 0.0))
        spread = np.where(count >= 2, spread, np.nan)
        result[name] = (mean, spread, count)

    return result, day_count, missing


def main(argv=None) -> int:
    bootstrap()
    parser = argparse.ArgumentParser(
        description="集合档案：月平均 × 时效 → 空间分布宽表 CSV")
    parser.add_argument("--root", type=Path, required=True, help="预报档案根目录")
    parser.add_argument("--months", required=True, help="起报月，逗号分隔，如 2025-08,2025-09")
    parser.add_argument("--leads", default="24,48,72,120", help="时效（小时），逗号分隔")
    parser.add_argument("--out", type=Path,
                        default=Path("spatial_monthly_ens.csv"))
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
        days = month_days_map[month]
        entries = [(day, member_dirs(day)) for day in days]
        members = sorted({member for _day, dirs in entries for member in dirs})
        if not members:
            sys.exit(f"{month} 在 {args.root} 下一个 member_* 目录都没有")
        print(f"  集合 {month}: {len(members)} 个成员 "
              f"({members[0]:03d}–{members[-1]:03d})")

        for lead in leads:
            result, day_count, missing = read_ensemble_lead(
                entries, members, lead, lat_target, lon_target
            )
            skipped += missing
            count = result["z500"][2]
            per_cell = float(np.nansum(np.where(np.isfinite(count), count, 0.0))
                             / max(int((count > 0).sum()), 1))
            print(f"    +{lead:03d}h: 缺 {missing} 个文件，"
                  f"每格点平均 {per_cell:.1f} 个成员")
            short = [member for member, used in day_count.items()
                     if used < len(days)]
            if short:
                print(f"    警告：{len(short)} 个成员缺起报日 "
                      f"（如 {short[:5]}），其月平均用了更少的天数")

            tag = month.replace("-", "")
            for name in ("z500", "ws850"):
                mean, spread, _count = result[name]
                columns[f"{name}_{tag}_h{lead:03d}"] = mean[::-1].ravel()
                columns[f"{name}_spread_{tag}_h{lead:03d}"] = spread[::-1].ravel()

        print(f"  ERA5 {month}:")
        for (name, lead), values in read_era5(
            days, leads, lat_target, lon_target
        ).items():
            tag = month.replace("-", "")
            columns[f"{name}_era5_{tag}_h{lead:03d}"] = values[::-1].ravel()

    if skipped:
        print(f"警告：{skipped} 个预报文件不存在，已跳过")

    # 自检：时效间必须不同，且离散度不能全空
    tag = months[0].replace("-", "")
    for name in ("z500", "ws850"):
        first = columns[f"{name}_{tag}_h{leads[0]:03d}"]
        last = columns[f"{name}_{tag}_h{leads[-1]:03d}"]
        if np.allclose(first, last, equal_nan=True):
            sys.exit(f"自检失败：{name} 各时效的集合均值完全相同")
        first = columns[f"{name}_spread_{tag}_h{leads[0]:03d}"]
        last = columns[f"{name}_spread_{tag}_h{leads[-1]:03d}"]
        if not np.isfinite(first).any() or np.allclose(first, last, equal_nan=True):
            sys.exit(f"自检失败：{name} 的离散度为空或各时效完全相同")

    write_frame(columns, lat_target, lon_target, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
