#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""评测产物目录 → 图表 + Markdown 报告（skill 的通用入口）。

    python render_report.py <产物目录> [选项]

    --result DIR       报告输出目录（默认 reports/<run_id>）
    --model NAME       报告标题里的模型名（默认取 manifest 的 run_id）
    --change TEXT      本次改动说明，写进报告第一节
    --compare 名字=CSV路径
                       TS：对比模型的宽表（可重复），在各产物目录的
                       diagnostics/categorical_wide.csv；台风：名字=产物目录。
    --members N        集合成员数，只在集合产物上有意义（manifest 不记这个字段）
    --leads ...        台风报告时效；省略时使用全部可用时效
    --fixed-leads ...  台风固定完整路径样本窗口
    --bootstrap N      台风分组 bootstrap 次数（默认 2000）
    --seed N           bootstrap 随机种子（默认 20250910）
    --small-cases N    台风少样本起报阈值（默认 10）
    --small-storms N   台风少样本台风阈值（默认 3）
    --forecast-type    显式声明旧元数据缺少的 det/ens
    --verify-legacy-members  只读核验旧集合归档的成员明细

本脚本是 agent 使用的唯一报告入口；认能力、选模板、出图、写报告在
``reporting.dispatch``。
"""

from __future__ import annotations

import argparse
import shlex
import sys
from pathlib import Path

from _bootstrap import bootstrap


def main(argv=None) -> int:
    bootstrap()

    parser = argparse.ArgumentParser(
        description="按评测产物目录自动选模板，出图并写 Markdown 报告",
        epilog="示例: python skills/evaluation-reporting/scripts/render_report.py "
        "evaluation_results/weather_ts_single_fgvp --result reports/weather_ts_single_fgvp",
    )
    parser.add_argument("output_dir", type=Path, help="评测产物目录（含 manifest.json 与结果表）")
    parser.add_argument("--result", type=Path, default=None, help="报告输出目录（默认 reports/<run_id>）")
    parser.add_argument("--model", default=None, help="模型名（默认取 manifest 的 run_id）")
    parser.add_argument("--primary-model", help="台风分析主模型")
    parser.add_argument("--previous-model", help="台风主模型的旧版本显示名")
    parser.add_argument("--primary-leads", type=float, nargs="+", help="台风主要讨论时效")
    parser.add_argument("--exploratory-leads", type=float, nargs="+", help="台风探索性讨论时效")
    parser.add_argument("--leads", type=float, nargs="+", help="台风报告时效；省略使用全部可用时效")
    parser.add_argument("--fixed-leads", type=float, nargs="+", help="台风固定完整路径样本窗口")
    parser.add_argument("--bootstrap", type=int, default=None, help="台风分组 bootstrap 次数；默认 2000")
    parser.add_argument("--seed", type=int, default=None, help="bootstrap 随机种子；默认 20250910")
    parser.add_argument("--small-cases", type=int, default=None, help="台风少样本起报阈值；默认 10")
    parser.add_argument("--small-storms", type=int, default=None, help="台风少样本台风阈值；默认 3")
    parser.add_argument("--forecast-type", choices=["det", "ens"], help="显式声明旧元数据缺少的预报类型")
    parser.add_argument("--verify-legacy-members", action="store_true", help="只读核验旧集合归档的成员明细")
    parser.add_argument("--change", default=None, help="本次改动说明，写进报告第一节")
    parser.add_argument(
        "--compare", action="append", default=None, metavar="名字=CSV路径",
        help="可重复：TS 为 NAME=宽表CSV；台风为 NAME=产物目录",
    )
    parser.add_argument(
        "--spectrum-variable", default=None,
        help="要点名看某一个起报的谱曲线的变量（读 diagnostics/spectrum_by_init.csv），"
        "只在连续场链路上有意义；该变量得真算过纬向谱",
    )
    parser.add_argument(
        "--spectrum-init", default=None,
        help="谱曲线的起报时刻，按子串匹配 init_time（如 20250102），省略取最后一个起报",
    )
    parser.add_argument(
        "--members", type=int, default=None,
        help="集合成员数（只在 weather_ts_ens 上有意义）。产物里没有这个字段，"
        "报告里凡要写成员数的地方都靠它；不给就省略数字",
    )
    args = parser.parse_args(argv)

    from reporting.dispatch import build_report

    invoked = [sys.executable, *sys.argv] if argv is None else ["python", *argv]
    artifacts = build_report(
        args.output_dir,
        out_dir=args.result,
        model_name=args.model,
        spectrum_variable=args.spectrum_variable,
        spectrum_init=args.spectrum_init,
        change_description=args.change,
        compare=args.compare,
        members=args.members,
        primary_model=args.primary_model, previous_model=args.previous_model,
        primary_leads=args.primary_leads, exploratory_leads=args.exploratory_leads,
        leads=args.leads, fixed_leads=args.fixed_leads,
        n_bootstrap=args.bootstrap, seed=args.seed,
        small_cases=args.small_cases, small_storms=args.small_storms,
        forecast_type=args.forecast_type,
        verify_legacy_members=args.verify_legacy_members,
        command=shlex.join(invoked),
    )
    print(f"\n产出 {len(artifacts)} 个文件：")
    for name, path in artifacts.items():
        print(f"  {name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
