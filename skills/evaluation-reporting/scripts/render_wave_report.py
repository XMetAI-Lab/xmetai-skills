#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""多个评估产物目录 → 纬向 FFT 与球谐谱补充分析报告（legacy）。

    python render_wave_report.py --model AIFS=<目录> --model FuXi=<目录> \
        --result report/wave_2models

``--model NAME=目录`` 可重复、至少两个；``NAME`` 是报告里显示的模型名。

    --result DIR     报告输出目录（必给）
    --archive TEXT   归档名，写进报告抬头与附录
    --own-dates      每个模型按自己的全部起报日统计（默认只用共同日期）

球谐带已并入 ``render_det_report.py`` / ``render_ens_report.py``，本入口只作
legacy 谱专题保留。纬向谱 log-RMS 用 ln（det 报告用 log10）。
本脚本只做入口，统计与渲染在 ``scripts/reporting/reports/wave_report.py``。
"""

from __future__ import annotations

import argparse
from pathlib import Path

from _bootstrap import bootstrap, parse_model


def main(argv=None) -> int:
    bootstrap()

    parser = argparse.ArgumentParser(
        description="按多个批次归档目录出图并写纬向 FFT / 球谐谱补充分析报告",
        epilog="示例: python skills/evaluation-reporting/scripts/render_wave_report.py "
        "--model FGVP=outputs/results/weather_rmse_single_fgvp/single_fgvp "
        "--model FGVP_pinpu=outputs/results/weather_rmse_single_fgvp_pinpu/dc849c75feb7d393 "
        "--result report/wave_2models",
    )
    parser.add_argument(
        "--model", action="append", default=None, metavar="NAME=目录", required=True,
        help="模型的批次归档目录（可重复，至少两个）",
    )
    parser.add_argument("--result", type=Path, required=True, help="报告输出目录")
    parser.add_argument("--archive", default=None,
                        help="归档名，写进报告抬头与附录的「归档文件」")
    parser.add_argument(
        "--own-dates", action="store_true",
        help="每个模型按自己的全部起报日统计，而不是只用共同日期；"
        "配对检验永远只在共同日期上做。",
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

    from reporting.reports.det_report import load_archive
    from reporting.reports.wave_report import build_wave_report

    archives = [load_archive(name, directory) for name, directory in models]
    artifacts = build_wave_report(
        archives, args.result, archive=args.archive, own_dates=args.own_dates,
    )

    print(f"\n产出 {len(artifacts)} 个文件：")
    for name, path in artifacts.items():
        print(f"  {name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
