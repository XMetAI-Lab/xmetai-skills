#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""多个暴雨过程个例产物根 → 多模型对比图 + Markdown 报告。

    python render_rainstorm_report.py --model FGVP=<输出根> --model AIFS=<输出根> \
        --model FuXi=<输出根> --model FengQing=<输出根> --model Tongyun=<输出根> \
        --process 202503,202521,202531 --result report/rainstorm

``--model NAME=目录`` 可重复、至少两个，第一个是主模型（Δ = 主模型 − 第二个）。
目录是 ``rainstorm_ts_single_*`` 的输出根，不是单个过程的产物目录。

    --process 编号[,编号...]  必给，先跑 render_rainstorm_summary.py 挑
    --result DIR              报告输出目录（必给）
    --title TEXT              覆盖报告标题
    --change TEXT             本次改动说明
    --archive TEXT            归档名，写进报告抬头

口径 A（逐日 24h）与口径 B（过程累积）基准率不同，只能在同口径内比。
本脚本只做入口，统计与渲染在 ``scripts/reporting/reports/rainstorm_report.py``。
"""

from __future__ import annotations

import argparse
from pathlib import Path

from _bootstrap import bootstrap, parse_model


def main(argv=None) -> int:
    bootstrap()

    parser = argparse.ArgumentParser(
        description="按多个暴雨过程个例产物根出图并写多模型对比报告",
        epilog="示例: python skills/evaluation-reporting/scripts/render_rainstorm_report.py "
        "--model FGVP=evaluation_results/rainstorm_ts_single_fgvp "
        "--model AIFS=evaluation_results/rainstorm_ts_single_aifs "
        "--model FuXi=evaluation_results/rainstorm_ts_single_fuxi "
        "--model FengQing=evaluation_results/rainstorm_ts_single_fengqing "
        "--model Tongyun=evaluation_results/rainstorm_ts_single_tongyun "
        "--process 202503,202521 --result reports/rainstorm",
    )
    parser.add_argument(
        "--model", action="append", default=None, metavar="NAME=目录", required=True,
        help="rainstorm_ts_single_* 的输出根（可重复，至少两个，第一个是主模型）",
    )
    parser.add_argument(
        "--process", default=None, metavar="编号[,编号...]", required=True,
        help="要详析的过程编号，逗号分隔（202501–202543）；编目全表见 rainstorm_catalog.py",
    )
    parser.add_argument("--result", type=Path, required=True, help="报告输出目录")
    parser.add_argument("--title", default=None, help="覆盖报告标题那一行")
    parser.add_argument("--change", default=None, help="本次改动说明，写进报告第一节")
    parser.add_argument("--archive", default=None, help="归档名，写进报告抬头的「归档：」")
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

    processes = [item.strip() for item in args.process.split(",") if item.strip()]
    if not processes:
        parser.error("--process 不能是空的：给至少一个过程编号，如 --process 202503,202521")

    from reporting.reports.rainstorm_report import (
        build_rainstorm_report,
        load_archive,
    )

    try:
        archives = [load_archive(name, directory) for name, directory in models]
        artifacts = build_rainstorm_report(
            archives,
            args.result,
            processes=processes,
            title=args.title or "2025 年暴雨过程个例检验报告",
            archive=args.archive,
            change=args.change,
        )
    except ValueError as error:
        parser.error(str(error))

    print(f"\n产出 {len(artifacts)} 个文件：")
    for name, path in artifacts.items():
        print(f"  {name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
