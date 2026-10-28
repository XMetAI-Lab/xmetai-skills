#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""多个集合评估产物目录 → 集合预报多模型评估检验报告。

    python render_ens_report.py --model FuXi=<目录> --model AIFS=<目录> \
        --result outputs/.temp/ens_2models

``--model NAME=目录`` 可重复、至少两个；``NAME`` 是报告里显示的模型名。

    --result DIR       报告输出目录（必给）
    --archive TEXT     归档名，写进报告抬头
    --members N        集合成员数（产物里没记，不给渲染成「—」）
    --own-dates        每个模型按自己的全部起报日统计（默认只用共同日期）
    --spatial-csv FILE §4.6 空间分布宽表（集合版取数，必须含 spread 列）
    --spatial-model NAME 空间图里的预报名称；缺省用第一个 --model

集合产物的 rmse / acc 是集合平均场口径，谱偏差用 ln；喂给
``render_det_report.py`` 不报错，但章节口径是 ``_single`` 的。
本脚本只做入口，统计与渲染在 ``scripts/reporting/reports/ens_report.py``。
"""

from __future__ import annotations

import argparse
from pathlib import Path

from _bootstrap import bootstrap, parse_model


def main(argv=None) -> int:
    bootstrap()

    parser = argparse.ArgumentParser(
        description="按多个集合批次归档目录出图并写集合预报评估检验报告",
        epilog="示例: python skills/evaluation-reporting/scripts/render_ens_report.py "
        "--model FuXi=outputs/results/weather_rmse_ens_fuxi/ens_fuxi "
        "--model AIFS=outputs/results/weather_rmse_ens_aifs/ens_aifs "
        "--result outputs/.temp/ens_2models",
    )
    parser.add_argument(
        "--model", action="append", default=None, metavar="NAME=目录", required=True,
        help="集合批次归档目录（可重复，至少两个）",
    )
    parser.add_argument("--result", type=Path, required=True, help="报告输出目录")
    parser.add_argument("--archive", default=None, help="归档名，写进报告抬头")
    parser.add_argument(
        "--members", type=int, default=None,
        help="集合成员数。产物的长表里没有这个字段，不给就渲染成「—」。",
    )
    parser.add_argument(
        "--own-dates", action="store_true",
        help="每个模型按自己的全部起报日统计，而不是只用共同日期；"
        "配对检验（§6）永远只在共同日期上做。",
    )
    parser.add_argument(
        "--spatial-csv", type=Path, default=None, metavar="文件",
        help="§4.6 空间分布的月平均宽表"
        "（extract_spatial_distribution_ens.py 的产物，必须含 spread 列）",
    )
    parser.add_argument(
        "--spatial-model", default=None, metavar="NAME",
        help="空间图里的预报名称；不给用第一个 --model",
    )
    args = parser.parse_args(argv)

    models = []
    for raw in args.model:
        try:
            models.append(parse_model(raw))
        except argparse.ArgumentTypeError as error:
            parser.error(str(error))
    if len(models) < 2:
        parser.error(f"多模型报告至少要两个 --model，现在只有 {len(models)} 个")
    names = [name for name, _ in models]
    if len(set(names)) != len(names):
        parser.error(f"模型名重复了: {names}")
    if args.spatial_csv is not None and not args.spatial_csv.is_file():
        parser.error(f"--spatial-csv 指向的 CSV 不存在：{args.spatial_csv}")
    if args.spatial_model is not None and args.spatial_csv is None:
        parser.error("--spatial-model 必须和 --spatial-csv 一起给")

    from reporting.reports.det_report import load_archive
    from reporting.reports.ens_report import build_ens_report

    archives = [load_archive(name, directory) for name, directory in models]
    artifacts = build_ens_report(
        archives, args.result, archive=args.archive,
        members=args.members, own_dates=args.own_dates,
        spatial_csv=args.spatial_csv, spatial_model=args.spatial_model,
    )

    print(f"\n产出 {len(artifacts)} 个文件：")
    for name, path in artifacts.items():
        print(f"  {name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
