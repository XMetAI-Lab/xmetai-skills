#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""空间分布出图（本地）：月平均 × 时效 → 模型 / ERA5 / 偏差（/ 离散度）。

吃 ``extract_spatial_distribution*.py`` 产出的宽表 CSV；CSV 里有 ``spread`` 列时
自动多画一行离散度，否则是确定性三行。只出 SVG。

    python render_spatial_distribution.py --csv spatial_monthly.csv \
        --out report/spatial --model Tongyun
"""

from __future__ import annotations

import argparse
from pathlib import Path

from _bootstrap import bootstrap


def main(argv=None) -> int:
    bootstrap()
    from reporting.reports.spatial_report import (
        load_spatial_csv,
        render_spatial,
    )

    parser = argparse.ArgumentParser(description="空间分布出图（模型 / ERA5 / 偏差）")
    parser.add_argument("--csv", type=Path, default=Path("spatial_monthly.csv"))
    parser.add_argument("--out", type=Path, default=Path("report/spatial"))
    parser.add_argument("--model", default="model", help="图例里的预报名称")
    args = parser.parse_args(argv)

    data = load_spatial_csv(args.csv)
    print(f"读入 {args.csv}：网格 {data.lat.size}×{data.lon.size}，"
          f"变量 {'、'.join(data.variables)}，"
          f"月份 {'、'.join(label for _tag, label in data.months)}，"
          f"时效 {'、'.join(f'+{lead}h' for lead in data.leads)}，"
          f"{'含' if data.has_spread else '不含'}离散度")
    paths = render_spatial(data, args.out, model=args.model, mode="auto")
    for path in paths.values():
        print(f"  写出 {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
