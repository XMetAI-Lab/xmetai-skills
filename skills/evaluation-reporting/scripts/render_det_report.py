#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""多个评估产物目录 → 多模型对比图 + Markdown 报告（确定性单成员）。

    python render_det_report.py --model FGVP=<目录> --model FuXi=<目录> \
        --result reports/det_2models

``--model NAME=目录`` 可重复、至少两个；``NAME`` 是报告里显示的模型名。

    --result DIR               报告输出目录（必给）
    --title TEXT               覆盖报告标题
    --change TEXT              本次改动说明，写进第一节
    --archive TEXT             归档名，写进报告抬头
    --declared NAME=v1,v2,...  声明变量清单（可重复）
    --primary NAME             §4.5.3 配对检验的主模型；缺省用第一个 --model
    --spatial-csv FILE         §4.6 空间分布宽表（单模型，确定性取数）
    --spatial-model NAME       空间图里的预报名称；缺省用 --primary / 第一个模型

本脚本只做入口，统计与渲染在 ``scripts/reporting/reports/det_report.py``。
"""

from __future__ import annotations

import argparse
from pathlib import Path

from _bootstrap import bootstrap, parse_model


def parse_declared(value: str):
    if "=" not in value:
        raise argparse.ArgumentTypeError(
            f"--declared 要写成 NAME=v1,v2,...，收到的是 {value!r}"
        )
    name, variables = value.split("=", 1)
    return name.strip(), [v.strip() for v in variables.split(",") if v.strip()]


def main(argv=None) -> int:
    bootstrap()

    parser = argparse.ArgumentParser(
        description="按多个批次归档目录出图并写多模型确定性评估报告",
        epilog="示例: python skills/evaluation-reporting/scripts/render_det_report.py "
        "--model FGVP=evaluation_results/weather_rmse_single_fgvp "
        "--model FuXi=evaluation_results/weather_rmse_single_fuxi --result reports/det_2models",
    )
    parser.add_argument(
        "--model", action="append", default=None, metavar="NAME=目录", required=True,
        help="模型的批次归档目录（可重复，至少两个）",
    )
    parser.add_argument("--result", type=Path, required=True, help="报告输出目录")
    parser.add_argument("--title", default=None, help="覆盖报告标题那一行")
    parser.add_argument("--change", default=None, help="本次改动说明，写进报告第一节")
    parser.add_argument("--archive", default=None, help="归档名，写进报告抬头的「归档：」")
    parser.add_argument(
        "--declared", action="append", default=None, metavar="NAME=v1,v2,...",
        help="声明变量清单（可重复），用于第 2 节的变量覆盖说明",
    )
    parser.add_argument(
        "--primary", default=None, metavar="NAME",
        help="§4.5.3 配对检验的主模型；不给用第一个 --model",
    )
    parser.add_argument(
        "--spatial-csv", type=Path, default=None, metavar="文件",
        help="§4.6 空间分布的月平均宽表（extract_spatial_distribution.py 的产物）",
    )
    parser.add_argument(
        "--spatial-model", default=None, metavar="NAME",
        help="空间图里的预报名称；不给用 --primary（或第一个 --model）",
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
    if args.primary is not None and args.primary not in names:
        parser.error(f"--primary {args.primary!r} 不在模型列表 {names} 里")
    if args.spatial_csv is not None and not args.spatial_csv.is_file():
        parser.error(f"--spatial-csv 指向的 CSV 不存在：{args.spatial_csv}")
    if args.spatial_model is not None and args.spatial_csv is None:
        parser.error("--spatial-model 必须和 --spatial-csv 一起给")

    declared = {}
    for raw in args.declared or []:
        try:
            name, variables = parse_declared(raw)
        except argparse.ArgumentTypeError as error:
            parser.error(str(error))
        declared[name] = variables

    from reporting.reports.det_report import build_det_report, load_archive

    archives = [load_archive(name, directory) for name, directory in models]
    artifacts = build_det_report(
        archives,
        args.result,
        title=args.title,
        change=args.change,
        archive=args.archive,
        declared=declared or None,
        primary=args.primary,
        spatial_csv=args.spatial_csv,
        spatial_model=args.spatial_model,
    )
    print(f"\n产出 {len(artifacts)} 个文件：")
    for name, path in artifacts.items():
        print(f"  {name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
