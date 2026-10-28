#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""43 个暴雨过程 × 2 口径 × N 模型 → 挑过程对比总表。

    python render_rainstorm_summary.py --model FGVP=<输出根> --model AIFS=<输出根> \
        --model FuXi=<输出根> --model FengQing=<输出根> --model Tongyun=<输出根> \
        --result evaluation_results/rainstorm_summary

``--model`` 缺省用 DEFAULT_MODELS，``--result`` 缺省用 DEFAULT_RESULT。

产物：
    summary_long.csv   一行 = (过程, 口径, [起报日], 阈值, 指标)，模型列并排 + TS 差值
    screening.md       口径 A 逐起报日、口径 B 逐过程的两张宽表 + 挑过程排序

过程级 TS 一律汇总重算（hits/misses/false_alarms 加总），不是逐日起报 TS 的平均。
两种口径基准率不同，只能同口径内比较。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from _bootstrap import bootstrap, parse_model

bootstrap()

from reporting.reports.rainstorm_catalog import codes, expected_days, get
from reporting.reports.rainstorm_report import (
    RANK_THRESHOLD,
    SCREENING_THRESHOLDS,
    daily_pooled,
    load_archive,
    load_process,
)

#: 模型名 → rainstorm_ts_single_* 输出根（换机器改这里）
DEFAULT_MODELS = [
    ("FGVP", "/workspace/szwCode/evaluation_results/rainstorm_ts_single_fgvp"),
    ("AIFS", "/workspace/szwCode/evaluation_results/rainstorm_ts_single_aifs"),
    ("FuXi", "/workspace/szwCode/evaluation_results/rainstorm_ts_single_fuxi"),
    ("FengQing", "/workspace/szwCode/evaluation_results/rainstorm_ts_single_fengqing"),
    ("Tongyun", "/workspace/szwCode/evaluation_results/rainstorm_ts_single_tongyun"),
]

DEFAULT_RESULT = "/workspace/szwCode/evaluation_results/rainstorm_summary"

METRICS = ["ts", "pod", "far", "miss_rate", "frequency_bias"]

ALL_CODES = codes()


def read_scores(path: Path):
    """读一份 scores.csv，只留全球行。"""
    if not path.is_file():
        return None
    frame = pd.read_csv(path)
    if frame.empty:
        return None
    frame["region"] = frame["region"].fillna("全球")
    return frame[frame["region"] == "全球"]


def load_total(root: Path, code: str):
    """口径 B 的产物表；缺产物返回 None。"""
    return read_scores(root / f"{code}_total" / "scores.csv")


def load_days(root: Path, code: str):
    """口径 A：{起报日: 表}。"""
    base = root / f"{code}_24h"
    days = {}
    if not base.is_dir():
        return days
    for day_dir in sorted(p for p in base.iterdir() if p.is_dir()):
        frame = read_scores(day_dir / "scores.csv")
        if frame is not None:
            days[day_dir.name] = frame
    return days


def segment_view(frame, metric: str = "ts"):
    """(阈值 -> value)。"""
    if frame is None:
        return None
    part = frame[frame["metric"] == metric]
    return dict(zip(part["threshold"], part["value"]))


def fmt_ts(view, threshold: float) -> str:
    if view is None:
        return "缺"
    value = view.get(threshold)
    if value is None or pd.isna(value):
        return "—"
    return f"{value:.3f}"


def frame_keys(frames):
    """长表要出的 (阈值, 指标) 组合。"""
    pairs = set()
    for frame in frames:
        if frame is not None:
            part = frame[frame["metric"].isin(METRICS)]
            pairs.update(zip(part["threshold"], part["metric"]))
    return sorted(pairs)


def fill_model_cells(row, names, frames, threshold, metric):
    """把各模型的 value / n_valid / status 填进一行；TS 再加差值。"""
    values = []
    for name, frame in zip(names, frames):
        part = (
            frame[(frame["metric"] == metric) & (frame["threshold"] == threshold)]
            if frame is not None
            else None
        )
        if part is None or part.empty:
            row[f"{name}.value"], row[f"{name}.n_valid"] = None, None
            row[f"{name}.status"] = "缺"
            values.append(None)
        else:
            first = part.iloc[0]
            row[f"{name}.value"] = float(first["value"])
            row[f"{name}.n_valid"] = int(first["n_valid"])
            row[f"{name}.status"] = first["status"]
            values.append(row[f"{name}.value"])
    if metric == "ts" and len(values) >= 2 and all(v is not None for v in values):
        row["delta_ts"] = values[0] - values[1]


def build_long(models, codes):
    names = [name for name, _ in models]
    rows = []
    for code in codes:
        process = get(code)

        per_model = {name: load_days(root, code) for name, root in models}
        all_days = (
            sorted(set().union(*[set(d) for d in per_model.values()]))
            if any(per_model.values())
            else []
        )
        for day in all_days:
            frames = [per_model[name].get(day) for name in names]
            for threshold, metric in frame_keys(frames):
                row = {
                    "编号": code,
                    "起止": process.span,
                    "等级": process.grade,
                    "口径": "逐日24h",
                    "起报日": day,
                    "threshold": threshold,
                    "metric": metric,
                }
                fill_model_cells(row, names, frames, threshold, metric)
                rows.append(row)

        totals = [load_total(root, code) for _, root in models]
        if any(f is not None for f in totals):
            for threshold, metric in frame_keys(totals):
                row = {
                    "编号": code,
                    "起止": process.span,
                    "等级": process.grade,
                    "口径": "过程累积",
                    "起报日": "",
                    "threshold": threshold,
                    "metric": metric,
                }
                fill_model_cells(row, names, totals, threshold, metric)
                rows.append(row)
    return pd.DataFrame(rows)


def _screening_row(cells, names, views, threshold):
    values = []
    for name in names:
        text = fmt_ts(views.get(name), threshold)
        cells.append(text)
        values.append(float(text) if text not in ("缺", "—") else None)
    cells.append(
        f"{values[0] - values[1]:+.3f}"
        if len(values) >= 2 and all(v is not None for v in values)
        else "—"
    )


def build_screening_markdown(models, codes):
    names = [name for name, _ in models]
    main_root = dict(models)[names[0]]
    main_archive = load_archive(names[0], main_root)
    lines = [
        "# 2025 年暴雨过程个例检验 · 挑过程总表",
        "",
        "TS 均为**全球行**（zd_sta_10285 全部站）。两种口径互不可比：",
        "「逐日24h」= 每个起报日的 day-1 窗（起报日 08 时 → 次日 08 时），**每天一个结果**；",
        "「过程累积」= 过程起点前一天的一根预报把整段降水累加后评总量。",
        "「缺」= 该段产物缺失（多半是那几天的预报数据缺）。",
        "",
    ]

    lines += ["## 口径：逐日24h（每行一个起报日）", ""]
    header = ["编号", "起止", "等级", "起报日"]
    for threshold in SCREENING_THRESHOLDS:
        header += [f"≥{threshold:g} {name}" for name in names] + [f"Δ({names[0]}-{names[1]})"]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "---|" * len(header))
    for code in codes:
        process = get(code)
        per_model = {name: load_days(root, code) for name, root in models}
        all_days = (
            sorted(set().union(*[set(d) for d in per_model.values()]))
            if any(per_model.values())
            else []
        )
        for day in all_days:
            cells = [code, process.span, process.grade, day]
            for threshold in SCREENING_THRESHOLDS:
                views = {name: segment_view(per_model[name].get(day)) for name in names}
                _screening_row(cells, names, views, threshold)
            lines.append("| " + " | ".join(cells) + " |")
    lines.append("")

    lines += ["## 口径：过程累积", ""]
    header = ["编号", "起止", "等级"]
    for threshold in SCREENING_THRESHOLDS:
        header += [f"≥{threshold:g} {name}" for name in names] + [f"Δ({names[0]}-{names[1]})"]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "---|" * len(header))
    for code in codes:
        process = get(code)
        views = {name: segment_view(load_total(root, code)) for name, root in models}
        cells = [code, process.span, process.grade]
        for threshold in SCREENING_THRESHOLDS:
            _screening_row(cells, names, views, threshold)
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")

    lines += [
        f"## 排序参考（逐日24h 汇总，{names[0]} ≥{RANK_THRESHOLD:g}mm TS，前 10）",
        "",
        "「汇总」= 把该过程各起报日的 hits / misses / false_alarms 加总后重算。",
        "",
    ]
    entries = []
    for code in codes:
        value = daily_pooled(load_process(main_archive, code), RANK_THRESHOLD)
        if value is not None:
            entries.append((value, code))
    entries.sort(reverse=True)
    for rank, (value, code) in enumerate(entries[:10], start=1):
        process = get(code)
        lines.append(f"{rank}. **{code}**（{process.span}，{process.grade}）汇总 TS = {value:.3f}")
    lines.append("")

    lines += [f"## 排序参考（过程累积，{names[0]} ≥{RANK_THRESHOLD:g}mm TS，前 10）", ""]
    entries = []
    for code in codes:
        view = segment_view(load_total(main_root, code))
        if view is not None:
            value = view.get(RANK_THRESHOLD)
            if value is not None and not pd.isna(value):
                entries.append((float(value), code))
    entries.sort(reverse=True)
    for rank, (value, code) in enumerate(entries[:10], start=1):
        process = get(code)
        lines.append(f"{rank}. **{code}**（{process.span}，{process.grade}）TS = {value:.3f}")
    lines.append("")
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="43 个暴雨过程 × 2 口径 × N 模型 → 挑过程对比总表",
    )
    parser.add_argument(
        "--model",
        action="append",
        type=parse_model,
        metavar="NAME=DIR",
        help="模型名=rainstorm_ts_single_* 输出根，可重复；缺省用 DEFAULT_MODELS",
    )
    parser.add_argument(
        "--result",
        default=DEFAULT_RESULT,
        help="汇总输出目录（缺省用 DEFAULT_RESULT）",
    )
    args = parser.parse_args(argv)

    models = args.model if args.model else [parse_model(f"{n}={d}") for n, d in DEFAULT_MODELS]
    if len(models) < 2:
        parser.error("至少两个 --model（对比表没有意义），或改脚本内 DEFAULT_MODELS")
    all_codes = ALL_CODES

    result_dir = Path(args.result)
    result_dir.mkdir(parents=True, exist_ok=True)

    long_table = build_long(models, all_codes)
    long_path = result_dir / "summary_long.csv"
    long_table.to_csv(long_path, index=False, encoding="utf-8-sig")

    markdown = build_screening_markdown(models, all_codes)
    md_path = result_dir / "screening.md"
    md_path.write_text(markdown, encoding="utf-8")

    print(f"长表 {len(long_table)} 行 -> {long_path}")
    print(f"宽表 -> {md_path}")
    for name, root in models:
        missing = []
        for code in all_codes:
            got = set(load_days(root, code))
            missing += [f"{code}_24h_{day}" for day in expected_days(code) if day not in got]
            if load_total(root, code) is None:
                missing.append(f"{code}_total")
        if missing:
            print(f"注意：{name} 缺 {len(missing)} 段产物：{', '.join(missing)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
