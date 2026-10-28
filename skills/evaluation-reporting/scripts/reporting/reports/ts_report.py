#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""TS 结果表 → Markdown 报告（多模型对比骨架）。

命令行入口 ``skills/evaluation-reporting/scripts/render_report.py``（按产物目录
自动认能力、选模板）；本模块只提供实现。

报告固定六章 + 附录，结构参考 ``weather_rmse_single.md``：

``# 1. 摘要`` / ``# 2. 样本口径与数据质量`` / ``# 3. 评估指标与解释`` /
``# 4. 总体评估结果``（4.1 TS、4.2 POD、4.3 FAR、4.4 BIAS、4.5 跨时效稳定性、
4.6 性能图、4.7 概率评分[集合]） / ``# 5. 分时效表现`` / ``# 6. 结论与建议`` /
``# 附录``。第 4 章的四个指标各自独占一节、各配一张随时效曲线图。

表**不编号**（每张表由上方粗体小标题引导），图按在正文中出现的顺序编号
``图 N：``。概率评分只在集合产物里出现，AROC 与 BSS 各占一张（固定的
``图 9``、``图 10`` 两位），因此 §5 分时效点图在集合报告里是图 11、在确定性
报告里是图 9。图号由 :class:`_Numbering` 跟着实际出图现算，缺图不占号；
集合报告即使没有概率表也会**保留节位与图号**。

``analysis_*`` 只吃 ``summarize()`` / ``box_stats()`` 算好的数字，不重新读表、
不重算指标——口径只有一处，表、图、文字就不会各说各话。
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from reporting.core.common import (
    bullet_lines,
    cn_number,
    fmt_number,
    source_of,
    table_lines,
)

#: §4.5（跨时效稳定性）里逐字固定的三行。``assets/templates/weather_ts_single.md``
#: 与 ``weather_ts_ens.md`` 的骨架要抄同一份字节，两边差一个全角/半角字符就会漂。
_BOX_HEADER = "| 模型 | 降水等级 | 中位数 | IQR | Q1–Q3 | 须线范围 | 离群时效 | 时效数 |"
_BOX_SEP = "|---|---|---|---|---|---|---|---|"

_BOX_NOTE = (
    "> 每个箱体的样本是**同模型、同量级在全部时效上的 TS 取值**"
    "（无有效 TS 的时效不计入，样本数见「时效数」列）：箱体为四分位距（IQR），中线为中位数，"
    "须线延伸到 1.5×IQR 内最远的点，空心圆为离群时效。**TS 越高越优**，"
    "所以箱体**越高**表示该量级整体技巧越好、**越窄**表示跨时效越稳定；"
    "这里的统计量与 4.1 节的「全时效平均」不是同一个量（等权平均 vs 中位数/IQR），不可互相替代。"
)

#: 概率评分宽表的必需列。缺任一列就不是这份表——**不猜、不补**。
PROBABILITY_COLUMNS = (
    "lead_h", "grade", "threshold_mm", "AROC", "BS", "BS_ref", "BSS",
    "base_rate", "n_points",
)

#: 强降水那一档：不低于这个阈值（mm）的**最小**一档；全批取同一个门槛。
HEAVY_RAIN_THRESHOLD = 25.0

#: 「最低可用技巧」门槛：有效时效上限按综合 TS 首次跌破它来判。
USABLE_TS = 0.10

#: 分带小节的图名：既是 ``artifacts`` 里的键，也是报告里 ``![…]()`` 的标签。
REGION_FIGURE = "region_TS_vs_lead"

#: 这份产物**一条带都没落数**时的说明。
_REGION_MISSING_NOTE = (
    "这份产物的宽表里没有 `region` 行：本次评测没配 `options[\"regions\"]`"
    "（或者配了但没有任何一条带算出结果）。"
)


class _Numbering:
    """图按在正文里出现的顺序编号。

    图会缺（确定性产物没有概率评分图），缺图时**连图注一起不出现**、序号仍然
    连续；集合报告的概率评分节虽然可能没有数据，但**节位与图号都保留**——由
    :meth:`reserve` 占号，这样 §5 的点图在两种报告里分别落到 9 / 11。
    """

    def __init__(self) -> None:
        self._figures: Dict[str, int] = {}

    def figure(self, name: str) -> int:
        number = len(self._figures) + 1
        self._figures[name] = number
        return number

    def reserve(self, name: str) -> int:
        """给一个**本节可能出现但本批没有图**的位置占号。"""
        return self.figure(name)


def _figure_block(
    artifacts: Optional[Dict[str, Path]],
    numbering: _Numbering,
    name: str,
    caption: str,
) -> List[str]:
    """图片 + 图注。产物里没有这张图（或不是图片）时返回空列表，编号也不占位。"""
    path = (artifacts or {}).get(name)
    if path is None or Path(path).suffix.lower() not in (".png", ".jpg", ".svg"):
        return []
    number = numbering.figure(name)
    return [f"![{name}]({Path(path).name})", "", f"图 {number}：{caption}", ""]


def _table_title(text: str) -> List[str]:
    """表题**在表上方**，不编号——条件章节增删时不用重排全篇表号。"""
    return [f"**{text}**", ""]


def _threshold_cell(value: object) -> str:
    """阈值显示成 ``0.1`` / ``13``，不补小数位。"""
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return "—"
    return f"{number:g}" if np.isfinite(number) else "—"


def _grade_sort_key(grade: str, threshold: Optional[float]) -> float:
    if threshold is not None and np.isfinite(threshold):
        return float(threshold)
    text = str(grade).replace("≥", "").strip()
    try:
        return float(text)
    except ValueError:
        return 1e9


def summarize(data: pd.DataFrame) -> Dict[str, object]:
    """把 TS 表压成报告需要的统计量。"""
    grades: List[str] = []
    per_grade: List[Dict[str, object]] = []
    for grade, subset in data.groupby("grade", observed=True):
        thresholds = subset["threshold_mm"].dropna() if "threshold_mm" in subset else pd.Series(dtype=float)
        threshold = float(thresholds.iloc[0]) if len(thresholds) else None
        totals = {
            key: float(subset[key].sum()) if key in subset.columns else float("nan")
            for key in ("hits", "misses", "false_alarms", "n_pairs")
        }
        hits = totals["hits"]
        misses = totals["misses"]
        false_alarms = totals["false_alarms"]

        def _ratio(numerator: float, denominator: float) -> float:
            if not np.isfinite(denominator) or denominator <= 0:
                return float("nan")
            return numerator / denominator

        entry = {
            "grade": str(grade),
            "threshold": threshold,
            # 主口径：先合计列联表再算比率；计数与之一致（都是合计）
            "TS_sum": _ratio(hits, hits + misses + false_alarms),
            "POD_sum": _ratio(hits, hits + misses),
            "FAR_sum": _ratio(false_alarms, hits + false_alarms),
            "漏报率_sum": _ratio(misses, hits + misses),
            "BIAS_sum": _ratio(hits + false_alarms, hits + misses),
            # 参考口径：各时效先算比率再等权平均
            "TS_mean": float(subset["TS"].mean()),
            "TS_last": float(subset.sort_values("lead_h")["TS"].iloc[-1]),
            "hits": int(hits) if np.isfinite(hits) else 0,
            "misses": int(misses) if np.isfinite(misses) else 0,
            "false_alarms": int(false_alarms) if np.isfinite(false_alarms) else 0,
            "n_pairs": int(totals["n_pairs"]) if np.isfinite(totals["n_pairs"]) else 0,
            "n_leads": int(subset["lead_h"].nunique()),
            "events": int(hits + misses) if np.isfinite(hits) else 0,
        }
        entry["TS"] = entry["TS_sum"]
        entry["POD"] = entry["POD_sum"]
        entry["FAR"] = entry["FAR_sum"]
        entry["漏报率"] = entry["漏报率_sum"]
        entry["BIAS"] = entry["BIAS_sum"]
        grades.append(str(grade))
        per_grade.append(entry)
    per_grade.sort(key=lambda item: _grade_sort_key(item["grade"], item["threshold"]))

    leads = sorted(float(value) for value in pd.unique(data["lead_h"]))
    first, last = leads[0], leads[-1]
    overall = float(np.nanmean([entry["TS_sum"] for entry in per_grade]))
    overall_mean = float(np.nanmean([entry["TS_mean"] for entry in per_grade]))
    weakest = per_grade[0] if per_grade else None
    strongest = per_grade[-1] if per_grade else None

    decay = None
    if weakest and strongest:
        first_ts = float(data[data["lead_h"] == first]["TS"].mean())
        last_ts = float(data[data["lead_h"] == last]["TS"].mean())
        decay = {
            "first_lead": first,
            "last_lead": last,
            "first_ts": first_ts,
            "last_ts": last_ts,
            "drop": first_ts - last_ts,
            "relative": (first_ts - last_ts) / first_ts if first_ts else np.nan,
            "per_day": (first_ts - last_ts) / max((last - first) / 24.0, 1e-9),
        }

    pivot = data.pivot_table(index="lead_h", columns="grade", values="TS", observed=True)
    return {
        "grades": per_grade,
        "leads": leads,
        "overall_ts": overall,
        "overall_ts_mean": overall_mean,
        "decay": decay,
        "pivot": pivot,
        "n_rows": int(len(data)),
    }


def diagnose(summary: Dict[str, object]) -> List[str]:
    """规则化诊断（主模型）：每条结论都带具体数值。"""
    grades = summary["grades"]
    findings: List[str] = []
    if not grades:
        return ["数据里没有可用的降水等级"]

    weak = grades[0]
    strong = grades[-1]
    overall = summary["overall_ts"]
    level = "较好" if overall >= 0.3 else ("中等" if overall >= 0.2 else "偏弱")
    findings.append(
        f"整体技巧 **{level}**：各等级全时效平均 TS = {fmt_number(overall)}"
        f"（参考经验分级：≥0.3 较好 / 0.2–0.3 中等 / <0.2 偏弱）。"
    )

    if len(grades) > 1:
        steps = [
            grades[index]["TS"] - grades[index + 1]["TS"]
            for index in range(len(grades) - 1)
        ]
        trend = "**单调衰减**" if all(step >= -1e-12 for step in steps) else "总体随量级下降"
        findings.append(
            f"技巧随量级{trend}：{weak['grade']}mm 的 TS {fmt_number(weak['TS'])}"
            f" → {strong['grade']}mm 的 TS {fmt_number(strong['TS'])}"
            f"（高量级命中数 {strong['hits']}，样本 {strong['n_pairs']}）。"
        )

    bias_entries = [entry for entry in grades if np.isfinite(entry.get("BIAS", np.nan))]
    if bias_entries:
        worst_bias = max(bias_entries, key=lambda entry: abs(np.log(max(entry["BIAS"], 1e-6))))
        direction = "系统性高报" if worst_bias["BIAS"] > 1 else "系统性漏报"
        findings.append(
            f"偏差结构：偏离理想值 1 最多的是 {worst_bias['grade']}mm，"
            f"BIAS = {fmt_number(worst_bias['BIAS'])}（{direction}）；"
            f"BIAS 是频率偏差（预报事件数/观测事件数），不是平均误差。"
        )

    far_entries = [
        entry
        for entry in grades
        if np.isfinite(entry.get("FAR", np.nan)) and np.isfinite(entry.get("漏报率", np.nan))
    ]
    if far_entries:
        entry = far_entries[-1]
        far_text, miss_text = fmt_number(entry["FAR"]), fmt_number(entry["漏报率"])
        if far_text == miss_text:
            findings.append(
                f"误差形态：{entry['grade']}mm 的 FAR 与漏报率均为 {far_text}，"
                f"空报与漏报比重相当，无主导方向。"
            )
        elif entry["FAR"] > entry["漏报率"]:
            findings.append(
                f"误差形态偏**空报**：{entry['grade']}mm 的 FAR {far_text}"
                f" 高于漏报率 {miss_text}。"
            )
        else:
            findings.append(
                f"误差形态偏**漏报**：{entry['grade']}mm 的漏报率 {miss_text}"
                f" 高于 FAR {far_text}。"
            )

    decay = summary.get("decay")
    if decay:
        findings.append(
            f"时效衰减：综合 TS 从 {decay['first_lead']:g}h 的 {fmt_number(decay['first_ts'])}"
            f" 降到 {decay['last_lead']:g}h 的 {fmt_number(decay['last_ts'])}"
            f"（绝对 -{fmt_number(decay['drop'])}，约 {fmt_number(decay['per_day'])}/天）。"
        )

    zero_hit = [entry for entry in grades if entry["hits"] == 0 and entry["events"] > 0]
    if zero_hit:
        names = "、".join(entry["grade"] for entry in zero_hit)
        findings.append(
            f"{names}mm 在全部时效中**命中为 0**（观测事件确实存在、只是没有报中），"
            f"因此 TS = 0.000；这与缺测的「—」含义不同，解读时注意区分。"
        )
    no_event = [entry for entry in grades if entry["events"] == 0]
    if no_event:
        names = "、".join(entry["grade"] for entry in no_event)
        findings.append(
            f"{names}mm 在评估期内**观测事件合计为 0**，属于样本库的物理上限："
            f"该量级技巧**不可评定**（不具备统计学意义），不是模型缺陷，也不应当作结论使用。"
        )
    sparse = [
        entry
        for entry in grades
        if 0 < entry["events"] < 50 and entry not in no_event
    ]
    if sparse:
        names = "、".join(
            f"{entry['grade']}（{entry['events']} 次事件）" for entry in sparse
        )
        findings.append(f"样本量过少、方差大，仅供参照：{names}。")
    return findings


def business_conclusion(grades: Sequence[Dict[str, object]]) -> str:
    """把技巧水平翻译成业务语言（基于数据自动生成，不是套话）。"""
    def _entry(threshold: float):
        for item in grades:
            if item.get("threshold") is not None and abs(item["threshold"] - threshold) < 1e-6:
                return item
        return None

    light = _entry(0.1)
    moderate = _entry(10.0) or _entry(13.0)
    heavy = _entry(25.0) or _entry(50.0)

    parts: List[str] = []
    if light:
        usable = "可作为有无降水的定性参考" if light["TS"] >= 0.3 else "仅能给出粗略的有雨/无雨趋势"
        parts.append(f"小雨（≥0.1mm）TS {fmt_number(light['TS'])}，{usable}")
    if moderate:
        parts.append(
            f"中雨（≥{fmt_number(moderate['threshold'], 0)}mm）TS {fmt_number(moderate['TS'])}，"
            + ("可作量级参考" if moderate["TS"] >= 0.2 else "已基本失去量级判别能力")
        )
    if heavy:
        parts.append(f"强降水（≥{fmt_number(heavy['threshold'], 0)}mm）TS {fmt_number(heavy['TS'])}")

    conclusion = "；".join(parts)
    if heavy and np.isfinite(heavy["TS"]) and heavy["TS"] < 0.15:
        conclusion += "。**结论定性：当前模型不宜直接用于强降水预警，需先做后处理订正。**"
    elif conclusion:
        conclusion += "。"
    return conclusion or "数据不足以给出业务定性。"


def recommendations(grades: Sequence[Dict[str, object]]) -> List[str]:
    """按数据条件给出的改进建议（只在条件成立时出现）。"""
    items: List[str] = []
    light = next((item for item in grades if item.get("threshold") == 0.1), None)
    heavy = next(
        (item for item in grades if item.get("threshold") in (25.0, 50.0)), None
    )

    if light and np.isfinite(light["BIAS"]) and light["BIAS"] > 1.3 and light["FAR"] > 0.45:
        items.append(
            f"小雨量级 BIAS {fmt_number(light['BIAS'])}、FAR {fmt_number(light['FAR'])} → "
            f"模型倾向「到处报小雨」、空间上过度平滑。损失函数可引入**焦点损失（Focal Loss）**"
            f"降低无雨背景点权重，并在频次上做**频率匹配**订正。"
        )
    if heavy and np.isfinite(heavy["TS"]) and heavy["TS"] < 0.15:
        items.append(
            f"强降水（≥{fmt_number(heavy['threshold'], 0)}mm）TS {fmt_number(heavy['TS'])} → "
            f"高频细节丢失。可引入**空间谱损失（Spectral Loss）**恢复小尺度结构，"
            f"并对极端事件做**样本重加权**以缓解长尾稀疏。"
        )
    sparse = [item for item in grades if item["events"] < 50]
    if sparse:
        names = "、".join(item["grade"] for item in sparse)
        items.append(
            f"{names}mm 的事件数不足 50（合计 {sum(item['events'] for item in sparse)} 次）→ "
            f"先延长评估期或扩大统计区域补齐样本，再评该量级技巧；"
            f"在样本补齐前不要据此调参。"
        )
    items.append(
        "后处理路径：分位数映射（QM）或频率匹配（FM）可同时压缩 BIAS 与空报；"
        "对强降水优先保证不漏报（提高 POD），再压 FAR。"
    )
    return items


# --------------------------------------------------------------------------- #
# 共用统计
# --------------------------------------------------------------------------- #
def _coverage_sentence(coverage: Optional[Dict[str, int]]) -> List[str]:
    """§2 与各节共用的实测覆盖句：各份给了多少对有效样本。

    对比表只写分数、不写样本量，而各份来源的起报日数可以差一个量级。不写出来，
    读者会把样本最少的那份的名次当成结论。样本数取**最小阈值档的全时效合计**，
    与 §2「最小阈值档配对样本」列同口径。
    """
    entries = [(str(name), int(value)) for name, value in (coverage or {}).items() if value]
    if len(entries) < 2:
        return ["本批只有一份来源，样本量见第 2 节的样本覆盖表；不存在跨来源的样本量差异。"]
    values = [value for _, value in entries]
    if max(values) == min(values):
        return [
            f"各模型的配对样本一致（{values[0]:,}，最小阈值档全时效合计），"
            f"样本量不构成名次差异的来源。"
        ]
    detail = "、".join(f"{name} {value:,}" for name, value in entries)
    lowest = min(entries, key=lambda item: item[1])
    highest = max(entries, key=lambda item: item[1])
    ratio = lowest[1] / highest[1] if highest[1] else float("nan")
    if ratio >= 0.9:
        return [
            f"各模型的配对样本**基本一致**（{detail}；最少的是最多的 {ratio:.1%}），"
            f"样本量不构成名次差异的来源。"
        ]
    text = (
        f"**各模型的实测覆盖并不齐**（最小阈值档的全时效配对样本数）：{detail}。"
        f"站点数每天大体固定，这组数的比值基本就是各份起报日数的比值"
        f"（最少的是最多的 {ratio:.1%}）。"
    )
    if ratio < 0.5:
        text += (
            f"{lowest[0]} 的抽样波动明显更大，它的名次不宜当结论用。"
        )
    return [text]


def _consistency_sentences(models: Dict[str, pd.DataFrame]) -> List[str]:
    """各模型的时效/等级/阈值集合是否一致；不一致时逐条写明差在哪。"""
    leads = {
        name: tuple(sorted(float(v) for v in pd.unique(frame["lead_h"])))
        for name, frame in models.items()
    }
    grades = {
        name: tuple(str(v) for v in ordered_grades([frame])) for name, frame in models.items()
    }
    thresholds = {
        name: tuple(
            sorted({
                float(value)
                for value in pd.to_numeric(
                    frame.get("threshold_mm", pd.Series(dtype=float)), errors="coerce"
                ).dropna()
            })
        )
        for name, frame in models.items()
    }
    lines: List[str] = []
    if len(set(leads.values())) == 1:
        values = next(iter(leads.values()))
        lines.append(
            f"各模型的时效集合一致（{values[0]:g}–{values[-1]:g}h，共 {len(values)} 个），"
            f"横向比较不受时效口径差影响。"
        )
    else:
        detail = "；".join(
            f"{name} {len(value)} 个（{value[0]:g}–{value[-1]:g}h）"
            for name, value in leads.items()
        )
        lines.append(f"**各模型的时效集合并不一致**：{detail}。横向比较只在共同时效上有意义。")
    if len(set(grades.values())) == 1:
        values = next(iter(grades.values()))
        lines.append(
            f"各模型的降水等级一致（{'、'.join(values)}），逐等级对比同源。"
        )
    else:
        detail = "；".join(
            f"{name} {'、'.join(value)}" for name, value in grades.items()
        )
        lines.append(f"**各模型的降水等级集合并不一致**：{detail}。缺档的模型在该等级上记 —。")
    if len(set(thresholds.values())) > 1:
        detail = "；".join(
            f"{name} {'/'.join(f'{v:g}' for v in value) or '—'}mm"
            for name, value in thresholds.items()
        )
        lines.append(f"阈值集合不一致：{detail}。")
    return lines


def _heavy_threshold(models: Dict[str, pd.DataFrame]) -> Optional[float]:
    """强降水档的门槛，**全批取同一个**；缺这一档的模型按缺值处理。"""
    thresholds = set()
    for frame in models.values():
        if "threshold_mm" not in frame.columns:
            continue
        values = pd.to_numeric(frame["threshold_mm"], errors="coerce").dropna()
        thresholds |= {float(value) for value in values if float(value) >= HEAVY_RAIN_THRESHOLD}
    return min(thresholds) if thresholds else None


def _heavy_entry(summary: Dict[str, object], heavy: Optional[float]) -> Optional[Dict[str, object]]:
    if heavy is None:
        return None
    for entry in summary["grades"]:
        if entry["threshold"] is not None and np.isclose(float(entry["threshold"]), heavy):
            return entry
    return None


def _mean_iqr(frame: pd.DataFrame, metric: str = "TS") -> float:
    """各等级跨时效 IQR 的等权平均。"""
    values = [
        float(box_stats(frame[frame["grade"].astype(str) == grade], metric)["iqr"])
        for grade in ordered_grades([frame])
    ]
    values = [value for value in values if np.isfinite(value)]
    return float(np.mean(values)) if values else float("nan")


def overview_metrics(models: Dict[str, pd.DataFrame]) -> Dict[str, Dict[str, float]]:
    """总体表的四个统计量（逐模型）：综合 TS、强降水 TS、|BIAS−1|、跨时效 IQR。"""
    heavy = _heavy_threshold(models)
    result: Dict[str, Dict[str, float]] = {}
    for name, frame in models.items():
        summary = summarize(frame)
        grades = summary["grades"]
        heavy_entry = _heavy_entry(summary, heavy)
        biases = [
            abs(float(entry["BIAS"]) - 1.0)
            for entry in grades
            if np.isfinite(entry.get("BIAS", np.nan))
        ]
        result[name] = {
            "ts": float(summary["overall_ts"]),
            "heavy": float("nan") if heavy_entry is None else float(heavy_entry["TS_sum"]),
            "bias": float(np.mean(biases)) if biases else float("nan"),
            "iqr": _mean_iqr(frame),
        }
    return result


def average_ranks(models: Dict[str, pd.DataFrame]) -> Dict[str, float]:
    """按「等级 × 时效」逐格竞争排名（1 = 该格最好），再对各格取平均。

    只在同一格有 ≥2 个模型有有效值时计名次；缺值不计、不拿 0 顶替。
    """
    collected: Dict[str, List[float]] = {name: [] for name in models}
    grades = ordered_grades(list(models.values()))
    for grade in grades:
        series = {
            name: frame[frame["grade"].astype(str) == grade]
            .set_index("lead_h")["TS"]
            .apply(pd.to_numeric, errors="coerce")
            for name, frame in models.items()
        }
        leads = sorted({float(lead) for value in series.values() for lead in value.index})
        for lead in leads:
            values = {
                name: float(value.get(lead, np.nan))
                for name, value in series.items()
            }
            values = {name: value for name, value in values.items() if np.isfinite(value)}
            if len(values) < 2:
                continue
            for name, value in values.items():
                collected[name].append(1 + sum(1 for other in values.values() if other > value + 1e-12))
    return {
        name: float(np.mean(values)) if values else float("nan")
        for name, values in collected.items()
    }


def _leaders(values: Dict[str, float], higher: bool = True) -> set:
    """按**显示精度**判并列：表上一样的两家不会在文字里分高下。"""
    finite = {name: float(value) for name, value in values.items() if np.isfinite(value)}
    if not finite:
        return set()
    best = max(finite.values()) if higher else min(finite.values())
    key = fmt_number(best)
    return {name for name, value in finite.items() if fmt_number(value) == key}


def _closest_to_one(values: Dict[str, float]) -> set:
    finite = {name: float(value) for name, value in values.items() if np.isfinite(value)}
    if not finite:
        return set()
    best = min(abs(value - 1.0) for value in finite.values())
    key = fmt_number(abs(best))
    return {name for name, value in finite.items() if fmt_number(abs(value - 1.0)) == key}


def _best_cell(name: str, leaders: set, value: float) -> str:
    text = fmt_number(value)
    return f"**{text}**" if name in leaders else text


def _best_names(leaders: set) -> str:
    return "、".join(sorted(leaders)) if leaders else "—"


# --------------------------------------------------------------------------- #
# 表格
# --------------------------------------------------------------------------- #
def _grade_metric_table(
    models: Dict[str, pd.DataFrame],
    summaries: Dict[str, Dict[str, object]],
    *,
    key: str,
    best_label: str,
    closer: bool = False,
    lower: bool = False,
) -> str:
    """行 = 降水等级，列 = 各模型 + 最佳。

    ``closer`` 用于 BIAS（按离 1 的距离判），``lower`` 用于 FAR（越低越好）。
    """
    names = list(models)
    grades = ordered_grades(list(models.values()))
    rows: List[List[str]] = []
    for grade in grades:
        values: Dict[str, float] = {}
        threshold = None
        for name in names:
            entry = next(
                (item for item in summaries[name]["grades"] if item["grade"] == grade),
                None,
            )
            if entry is None:
                continue
            threshold = entry["threshold"]
            values[name] = float(entry.get(key, np.nan))
        if closer:
            leaders = _closest_to_one(values)
        else:
            leaders = _leaders(values, higher=not lower)
        cells = [_best_cell(name, leaders, values.get(name, np.nan)) for name in names]
        rows.append([grade, _threshold_cell(threshold), *cells, _best_names(leaders)])
    header = (
        ["降水等级", "阈值(mm)"]
        + names
        + [best_label]
    )
    return "\n".join(table_lines(header, rows))


def _grade_counts_table(summary: Dict[str, object]) -> str:
    rows = [
        [
            entry["grade"],
            str(entry["hits"]),
            str(entry["misses"]),
            str(entry["false_alarms"]),
            f"{entry['n_pairs']:,}",
        ]
        for entry in summary["grades"]
    ]
    return "\n".join(
        table_lines(["降水等级", "命中", "漏报", "空报", "样本合计"], rows)
    )


def _lead_mean_values(models: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """index = 时效，columns = 模型，值 = 该时效各降水等级 TS 的等权平均。"""
    series = {}
    for name, frame in models.items():
        values = (
            frame.groupby("lead_h")["TS"].mean().sort_index()
        )
        series[name] = values
    table = pd.DataFrame(series)
    table.index = [float(index) for index in table.index]
    return table.sort_index()


def _lead_mean_table(models: Dict[str, pd.DataFrame]) -> str:
    table = _lead_mean_values(models)
    rows: List[List[str]] = []
    for lead, row in table.iterrows():
        values = {name: float(row.get(name, np.nan)) for name in table.columns}
        leaders = _leaders(values)
        cells = [_best_cell(name, leaders, values[name]) for name in table.columns]
        rows.append([f"{float(lead):g}", *cells, _best_names(leaders)])
    return "\n".join(
        table_lines(["时效(h)", *list(table.columns), "最佳"], rows)
    )


def _lead_decay_table(models: Dict[str, pd.DataFrame]) -> str:
    """逐模型的首末时效技巧与衰减幅度（**不列全时效**，全表在 §5）。"""
    table = _lead_mean_values(models)
    if table.empty:
        return "\n".join(table_lines(["模型", "有效时效"], [["—", "0"]]))
    rows: List[List[str]] = []
    for name in table.columns:
        series = table[name].dropna()
        if series.empty:
            rows.append([name, "—", "—", "—", "—", "—"])
            continue
        start, end = float(series.iloc[0]), float(series.iloc[-1])
        drop = start - end
        relative = drop / start if start else float("nan")
        below = series[series < USABLE_TS]
        first_below = f"{float(below.index[0]):g}" if len(below) else "—"
        rows.append([
            name,
            f"{fmt_number(start)}（{float(series.index[0]):g}h）",
            f"{fmt_number(end)}（{float(series.index[-1]):g}h）",
            fmt_number(drop),
            f"−{relative:.0%}" if np.isfinite(relative) else "—",
            first_below,
        ])
    return "\n".join(
        table_lines(
            ["模型", "首时效TS（时效）", "末时效TS（时效）", "绝对下降",
             "相对下降", f"首次低于{USABLE_TS:g}的时效(h)"],
            rows,
        )
    )


def _performance_table(models: Dict[str, pd.DataFrame], lead: float) -> str:
    rows: List[List[str]] = []
    for name, frame in models.items():
        subset = frame[pd.to_numeric(frame["lead_h"], errors="coerce") == lead]
        pod = pd.to_numeric(subset["POD"], errors="coerce").dropna() if "POD" in subset else pd.Series(dtype=float)
        far = pd.to_numeric(subset["FAR"], errors="coerce").dropna() if "FAR" in subset else pd.Series(dtype=float)
        pod_value = float(pod.mean()) if len(pod) else float("nan")
        far_value = float(far.mean()) if len(far) else float("nan")
        distance = (
            float(np.hypot(1.0 - pod_value, far_value))
            if np.isfinite(pod_value) and np.isfinite(far_value)
            else float("nan")
        )
        rows.append([
            name,
            fmt_number(pod_value),
            fmt_number(far_value),
            fmt_number(1.0 - far_value if np.isfinite(far_value) else float("nan")),
            fmt_number(distance),
        ])
    return "\n".join(
        table_lines(["模型", "平均POD", "平均FAR", "平均1−FAR", "距右上角"], rows)
    )


def box_stats(data: pd.DataFrame, metric: str = "TS") -> Dict[str, object]:
    """一个箱体的 Tukey 统计量——**与 matplotlib 的 boxplot 同口径**。

    样本是"同一模型、同一量级在全部时效上的取值"，一行一个时效。
    放进 ``ts_report`` 而不是画图模块，是为了让**报告表格和图上箱体同源**。
    """
    if metric not in data.columns:
        raise KeyError(f"结果表里没有指标列 '{metric}'")

    values = pd.to_numeric(data[metric], errors="coerce").to_numpy(dtype=float)
    leads = (
        pd.to_numeric(data["lead_h"], errors="coerce").to_numpy(dtype=float)
        if "lead_h" in data.columns
        else np.full(values.shape, np.nan)
    )
    keep = np.isfinite(values)
    values, leads = values[keep], leads[keep]

    if values.size == 0:
        nan = float("nan")
        return {
            "n": 0, "q1": nan, "median": nan, "q3": nan, "iqr": nan,
            "whisker_lo": nan, "whisker_hi": nan,
            "outlier_leads": [], "outlier_values": [],
        }

    # 分位数用 np.percentile 的默认参数（线性插值）——matplotlib 的 cbook.boxplot_stats
    # 用的也是它，所以口径天然一致；千万别传 method=，一传两边就可能对不上。
    q1, median, q3 = (float(value) for value in np.percentile(values, [25, 50, 75]))
    iqr = q3 - q1
    low_fence, high_fence = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    inside = values[(values >= low_fence) & (values <= high_fence)]
    outliers = (values < low_fence) | (values > high_fence)
    return {
        "n": int(values.size),
        "q1": q1,
        "median": median,
        "q3": q3,
        "iqr": iqr,
        "whisker_lo": float(inside.min()) if inside.size else q1,
        "whisker_hi": float(inside.max()) if inside.size else q3,
        "outlier_leads": [float(value) for value in leads[outliers]],
        "outlier_values": [float(value) for value in values[outliers]],
    }


def box_stats_by_grade(data: pd.DataFrame, metric: str = "TS") -> List[Dict[str, object]]:
    """``[{"grade", "threshold", **box_stats(...)}]``，按阈值升序。"""
    rows: List[Dict[str, object]] = []
    for grade, subset in data.groupby("grade", observed=True):
        thresholds = (
            subset["threshold_mm"].dropna()
            if "threshold_mm" in subset.columns
            else pd.Series(dtype=float)
        )
        rows.append(
            {
                "grade": str(grade),
                "threshold": float(thresholds.iloc[0]) if len(thresholds) else None,
                **box_stats(subset, metric=metric),
            }
        )
    rows.sort(key=lambda row: _grade_sort_key(row["grade"], row["threshold"]))
    return rows


def ordered_grades(frames: Sequence[pd.DataFrame]) -> List[str]:
    """多张结果表里出现过的降水等级**并集**，按 ``threshold_mm`` 升序。"""
    thresholds: Dict[str, float] = {}
    for frame in frames:
        if "grade" not in frame.columns:
            continue
        first = (
            frame.groupby("grade", observed=True)["threshold_mm"].first()
            if "threshold_mm" in frame.columns
            else pd.Series(dtype=float)
        )
        for grade in frame["grade"].astype(str):
            if grade in thresholds:
                continue
            value = first.get(grade, float("nan")) if len(first) else float("nan")
            thresholds[grade] = float(value) if pd.notna(value) else float("nan")
    return sorted(thresholds, key=lambda grade: _grade_sort_key(grade, thresholds[grade]))


def collect_box_models(
    model_name: str,
    data: pd.DataFrame,
    *,
    baseline_df: Optional[pd.DataFrame] = None,
    baseline_name: Optional[str] = None,
    baselines: Optional[Dict[str, pd.DataFrame]] = None,
) -> Dict[str, pd.DataFrame]:
    """报告里参与对比的模型：本模型在前，然后是单基准、各对比模型。"""
    models: Dict[str, pd.DataFrame] = {model_name: data}
    if baseline_df is not None and baseline_name:
        models[baseline_name] = baseline_df
    models.update(baselines or {})
    return models


def _box_table(models: Dict[str, pd.DataFrame], metric: str = "TS") -> str:
    """一行一个（模型 × 降水等级）：量级升序，同量级内按 ``models`` 的顺序。"""
    names = list(models)
    grades = ordered_grades(list(models.values()))
    stats = {
        name: {row["grade"]: row for row in box_stats_by_grade(frame, metric=metric)}
        for name, frame in models.items()
    }

    lines = [_BOX_HEADER, _BOX_SEP]
    for grade in grades:
        for name in names:
            row = stats[name].get(grade)
            if row is None or row["n"] == 0:
                # 五格 —— 对应中位数/IQR/Q1–Q3/须线/离群时效，末格才是时效数 0。
                cells = ["—"] * 5 + ["0"]
            else:
                leads = "、".join(f"{lead:g}h" for lead in sorted(row["outlier_leads"])) or "—"
                cells = [
                    fmt_number(row["median"]),
                    fmt_number(row["iqr"]),
                    f"{fmt_number(row['q1'])}–{fmt_number(row['q3'])}",
                    f"{fmt_number(row['whisker_lo'])}–{fmt_number(row['whisker_hi'])}",
                    leads,
                    str(row["n"]),
                ]
            lines.append(f"| {name} | {grade} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


# ------------------------------------------------------------- 概率评分（§4.7）
def _members_phrase(members: Optional[int]) -> str:
    """``" 51 个成员"`` / ``"集合成员"``（与 §1「集合每例 51 个成员」同一排版）。

    产物里**没有**成员数（``manifest.json`` 不写这个字段），拿不到时老老实实
    写「集合成员」，不编一个数字出来。数字前带一个空格，是因为两处调用都直接
    接在中文动词后面（「均在…」「都是…」），中英文之间要留白；没有数字时不留。
    """
    return f" {int(members)} 个成员" if members else "集合成员"


def load_probability(frame: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    """校验概率宽表；``None`` 原样返回，缺列则报错。

    Raises:
        ValueError: 表里缺 :data:`PROBABILITY_COLUMNS` 里的列。
    """
    if frame is None:
        return None
    missing = [column for column in PROBABILITY_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(
            f"概率宽表缺列 {missing}；现有列 {list(frame.columns)}。"
            f"这份表由 weather_ts_ens_prob 能力写出，配置里写 "
            f'pipeline=["weather_ts_ens", "weather_ts_ens_prob"] 才会落到 '
            f"diagnostics/probability_wide.csv。"
        )
    return frame


def probability_summary(frame: pd.DataFrame) -> Dict[str, object]:
    """单模型的概率评分汇总：逐档一行 + 一行 ``AVG`` + ``by_lead``。"""
    if "grade" in frame.columns:
        frame = frame[frame["grade"].astype(str) != "AVG"]

    rows: List[Dict[str, object]] = []
    for grade, subset in frame.groupby("grade", observed=True):
        thresholds = pd.to_numeric(subset["threshold_mm"], errors="coerce").dropna()
        entry: Dict[str, object] = {
            "grade": str(grade),
            "threshold": float(thresholds.iloc[0]) if len(thresholds) else None,
            "n_leads": int(pd.to_numeric(subset["lead_h"], errors="coerce").nunique()),
        }
        for key in ("AROC", "BS", "BS_ref", "BSS", "base_rate"):
            values = pd.to_numeric(subset[key], errors="coerce").dropna()
            entry[key] = float(values.mean()) if len(values) else float("nan")
        points = pd.to_numeric(subset["n_points"], errors="coerce").dropna()
        entry["n_points"] = int(points.sum()) if len(points) else 0
        rows.append(entry)
    rows.sort(key=lambda item: _grade_sort_key(item["grade"], item["threshold"]))

    average: Dict[str, object] = {
        "grade": "AVG", "threshold": None, "base_rate": None, "n_points": None,
    }
    for key in ("AROC", "BS", "BS_ref", "BSS"):
        values = [float(item[key]) for item in rows if np.isfinite(item[key])]
        average[key] = float(np.mean(values)) if values else float("nan")
    average["n_grades"] = len(
        [item for item in rows if np.isfinite(item["AROC"]) and np.isfinite(item["BSS"])]
    )
    average["n_leads"] = max((item["n_leads"] for item in rows), default=0)

    numeric = frame.assign(
        lead_h=pd.to_numeric(frame["lead_h"], errors="coerce"),
        AROC=pd.to_numeric(frame["AROC"], errors="coerce"),
        BSS=pd.to_numeric(frame["BSS"], errors="coerce"),
    )
    by_lead = numeric.groupby("lead_h")[["AROC", "BSS"]].mean().dropna(how="all")
    return {"rows": rows, "average": average, "by_lead": by_lead}


def _probability_window(frame: pd.DataFrame) -> Optional[float]:
    """概率宽表的累积窗口（``window_h`` 列）——表题与引用块都要写它。"""
    if "window_h" not in frame.columns:
        return None
    values = pd.to_numeric(frame["window_h"], errors="coerce").dropna()
    return float(values.iloc[0]) if len(values) else None


def _threshold_list(frame: pd.DataFrame) -> str:
    """``0.1/4/13/25mm``——引用块里要和 TS 的阈值列表并排写。"""
    values = sorted(
        {float(value) for value in pd.to_numeric(frame["threshold_mm"], errors="coerce").dropna()}
    )
    return "/".join(f"{value:g}" for value in values) + "mm" if values else "—"


def _probability_note(window: Optional[float]) -> str:
    """概率评分表下的口径引用块。"""
    span = f"{window:g}h" if window else "概率"
    return (
        f"> BS_ref 是气候基准的 Brier 分数，**只有外部给了气候概率参考的 {span} 窗才有值**，"
        f"其余窗口整列写 —；BSS = 1 − BS/BS_ref。AROC 在**逐成员插值后**按超越式阈值"
        f"（x ≥ 阈值）计算，与前面各节「插值到站再配对」的做法不是一回事。"
        f"AVG 行是对**可算阈值**的等权平均（该档无事件的跳过），不是按样本数加权。"
    )


def _probability_table(
    summaries: Dict[str, Dict[str, object]], *, with_model: bool
) -> str:
    """概率评分表：一行一个「模型 × 降水等级」，每个模型末尾再加一行 ``AVG``。"""
    header = (["模型"] if with_model else []) + [
        "降水等级", "阈值(mm)", "AROC", "BS", "BS_ref", "BSS", "基准率", "样本点数",
    ]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for name, summary in summaries.items():
        for entry in list(summary["rows"]) + [summary["average"]]:
            cells = [
                _threshold_cell(entry.get("threshold")),
                fmt_number(entry.get("AROC")),
                fmt_number(entry.get("BS")),
                fmt_number(entry.get("BS_ref")),
                fmt_number(entry.get("BSS")),
                fmt_number(entry.get("base_rate")),
                (
                    f"{entry['n_points']:,}" if entry.get("n_points")
                    else (f"{entry['n_grades']} 档" if entry.get("n_grades") else "—")
                ),
            ]
            prefix = [str(name)] if with_model else []
            lines.append("| " + " | ".join(prefix + [str(entry["grade"])] + cells) + " |")
    return "\n".join(lines)


def analysis_probability(summaries: Dict[str, Dict[str, object]]) -> List[str]:
    """概率评分判读：区分能力、Brier 技巧的正负与走向、高阈值档的样本。

    一条一行，由调用方套 ``• `` ——与其余各节的判读同一种排版（``_analysis_*``）。
    """
    if not summaries:
        return ["没有概率评分结果，本节无法给出定量分析。"]
    names = list(summaries)
    primary = names[0]
    rows = summaries[primary]["rows"]
    average = summaries[primary]["average"]
    parts: List[str] = []

    usable = [entry for entry in rows if np.isfinite(entry.get("AROC", np.nan))]
    if usable:
        best = max(usable, key=lambda item: item["AROC"])
        worst = min(usable, key=lambda item: item["AROC"])
        below = [entry for entry in usable if entry["AROC"] <= 0.5]
        head = f"{primary} 各档 AROC 在 {fmt_number(worst['AROC'])}–{fmt_number(best['AROC'])} 之间，"
        if below:
            head += (
                "其中 " + "、".join(entry["grade"] for entry in below)
                + " 未超过 0.5 的无技巧线，这几档的概率预报分不开事件与非事件；"
            )
        else:
            head += (
                f"全部高于 0.5 的无技巧线（最低的 {worst['grade']} 档高出 "
                f"{fmt_number(worst['AROC'] - 0.5)}）；"
            )
        parts.append(head + f"区分能力最强的是 {best['grade']}（{fmt_number(best['AROC'])}）。")

    scored = [entry for entry in rows if np.isfinite(entry.get("BSS", np.nan))]
    if scored:
        positive = [entry for entry in scored if entry["BSS"] > 0]
        best = max(scored, key=lambda item: item["BSS"])
        worst = min(scored, key=lambda item: item["BSS"])
        if positive:
            parts.append(
                f"BSS 有 {len(positive)}/{len(scored)} 档为正：最好的是 {best['grade']}"
                f"（{fmt_number(best['BSS'])}），最差的是 {worst['grade']}"
                f"（{fmt_number(worst['BSS'])}）。"
            )
        else:
            parts.append(
                f"BSS **一档都没有为正**（最好的是 {best['grade']}，"
                f"{fmt_number(best['BSS'])}），整批概率预报都不及气候基准。"
            )
        if worst["BSS"] < 0 and positive:
            parts.append(
                f"{worst['grade']} 档为负，说明这一档的概率预报**还不如气候基准**——"
                f"它的 BS {fmt_number(worst['BS'])} 高于 BS_ref {fmt_number(worst['BS_ref'])}。"
            )
        elif not positive:
            parts.append(
                f"最差的 {worst['grade']} 档 BS {fmt_number(worst['BS'])} "
                f"远高于 BS_ref {fmt_number(worst['BS_ref'])}。"
            )

    if rows:
        lightest, heaviest = rows[0], rows[-1]
        if (
            lightest is not heaviest
            and lightest.get("n_points")
            and np.isfinite(lightest.get("base_rate", np.nan))
            and np.isfinite(heaviest.get("base_rate", np.nan))
            and heaviest["base_rate"] > 0
        ):
            ratio = lightest["base_rate"] / heaviest["base_rate"]
            extra = f"（约 {ratio:.0f} 分之一）" if ratio >= 2 else ""
            parts.append(
                f"各档的样本点数相同（{lightest['n_points']:,} 对——阈值只改事件的定义、"
                f"不改配对点），但**事件次数**随量级急剧收缩：{lightest['grade']} 档的"
                f"基准率 {fmt_number(lightest['base_rate'])}、{heaviest['grade']} 档只剩 "
                f"{fmt_number(heaviest['base_rate'])}{extra}——高阈值档的 AROC/BSS 波动更大。"
            )

    by_lead = summaries[primary].get("by_lead")
    if by_lead is not None and len(by_lead) >= 2:
        first_lead, last_lead = float(by_lead.index[0]), float(by_lead.index[-1])
        aroc_first, aroc_last = float(by_lead["AROC"].iloc[0]), float(by_lead["AROC"].iloc[-1])
        bss_first, bss_last = float(by_lead["BSS"].iloc[0]), float(by_lead["BSS"].iloc[-1])
        parts.append(
            f"AVG 档随时效：AROC 从 {first_lead:g}h 的 {fmt_number(aroc_first)} 变到 "
            f"{last_lead:g}h 的 {fmt_number(aroc_last)}（{aroc_last - aroc_first:+.3f}），"
            f"BSS 从 {fmt_number(bss_first)} 变到 {fmt_number(bss_last)}（{bss_last - bss_first:+.3f}）——"
            + (
                "区分能力随时效退化，长时效的结论同样要限定范围。"
                if aroc_last < aroc_first
                else "区分能力并不随时效单调下降，这时段内的分档主要由样本波动支配。"
            )
        )

    if len(names) > 1:
        pool = {
            name: summary["average"]
            for name, summary in summaries.items()
            if np.isfinite(summary["average"].get("AROC", np.nan))
            and np.isfinite(summary["average"].get("BSS", np.nan))
        }
        if len(pool) >= 2:
            best_aroc = max(pool, key=lambda name: pool[name]["AROC"])
            best_bss = max(pool, key=lambda name: pool[name]["BSS"])
            parts.append(
                f"AVG 档上 AROC 最高的是 {best_aroc}（{fmt_number(pool[best_aroc]['AROC'])}）、"
                f"BSS 最高的是 {best_bss}（{fmt_number(pool[best_bss]['BSS'])}）。"
                f"各模型的概率样本量并不相同（见「样本点数」列），样本少的那个名次波动更大。"
            )

    if np.isfinite(average.get("AROC", np.nan)):
        parts.append(
            f"{primary} 的 AVG 档：AROC {fmt_number(average['AROC'])}、BSS {fmt_number(average['BSS'])}"
            f"（对 {average['n_grades']} 个可算阈值等权平均）。"
        )
    return parts or ["概率评分的各档取值都不可用，本节无法给出定量分析。"]


# --------------------------------------------------------------------------- #
# 判读（只读已算好的统计量）
# --------------------------------------------------------------------------- #
def _sort_value(value: float) -> float:
    return float(value) if np.isfinite(value) else -np.inf


def _comparison_sentence(
    models: Dict[str, pd.DataFrame],
    overview: Dict[str, Dict[str, float]],
    ranks: Dict[str, float],
) -> str:
    if len(models) < 2:
        return ""
    order = sorted(overview, key=lambda name: _sort_value(overview[name]["ts"]), reverse=True)
    best, worst = order[0], order[-1]
    if not np.isfinite(overview[best]["ts"]):
        return ""
    text = (
        f"本批 {len(models)} 个模型中，综合 TS 最高的是 **{best}**"
        f"（{fmt_number(overview[best]['ts'])}），最低的是 **{worst}**"
        f"（{fmt_number(overview[worst]['ts'])}）；"
    )
    rank_pool = {name: value for name, value in ranks.items() if np.isfinite(value)}
    if rank_pool:
        rank_best = min(rank_pool, key=rank_pool.get)
        text += f"按等级×时效逐格排名的平均名次最好是 **{rank_best}**（{fmt_number(rank_pool[rank_best], 2)}）。"
    return text


def _analysis_overview(
    models: Dict[str, pd.DataFrame],
    overview: Dict[str, Dict[str, float]],
    ranks: Dict[str, float],
    heavy: Optional[float],
) -> List[str]:
    if len(models) < 2:
        name = next(iter(models))
        values = overview[name]
        return [
            f"单模型评估：综合 TS {fmt_number(values['ts'])}、强降水 TS "
            f"{fmt_number(values['heavy'])}、|BIAS−1| {fmt_number(values['bias'])}、"
            f"跨时效 IQR {fmt_number(values['iqr'])}。"
        ]
    lines: List[str] = []
    order = sorted(overview, key=lambda name: _sort_value(overview[name]["ts"]), reverse=True)
    best, worst = order[0], order[-1]
    if np.isfinite(overview[best]["ts"]) and np.isfinite(overview[worst]["ts"]):
        lines.append(
            f"综合 TS 最高的是 {best}（{fmt_number(overview[best]['ts'])}），"
            f"最低的是 {worst}（{fmt_number(overview[worst]['ts'])}），"
            f"相差 {fmt_number(overview[best]['ts'] - overview[worst]['ts'])}。"
        )
    # 综合 TS 在显示精度上恒为 0 的模型，柱状图里几乎看不见——先说明是数值趋零。
    zero_skill = {
        name for name, values in overview.items() if fmt_number(values["ts"]) == "0.000"
    }
    if zero_skill:
        lines.append(
            f"{'、'.join(sorted(zero_skill))} 的全等级平均 TS 为 0.000"
            f"（各时效都接近 0），本章开头的基准时效柱状图里它的柱高几乎不可见——"
            f"这是数值趋零，不是缺数据；建议先核查该份宽表的口径。"
        )
    heavy_values = {
        name: values["heavy"] for name, values in overview.items() if np.isfinite(values["heavy"])
    }
    if heavy_values:
        leaders = _leaders(heavy_values)
        top = max(heavy_values.values())
        bottom = min(heavy_values.values())
        label = f"≥{heavy:g}mm" if heavy is not None else "强降水档"
        lines.append(
            f"{label} 的 TS 最高的是 {'、'.join(sorted(leaders))}（{fmt_number(top)}），"
            f"最低 {fmt_number(bottom)}；高量级样本稀疏，这一项的小数位不宜过度解读。"
        )
    bias_values = {
        name: values["bias"] for name, values in overview.items() if np.isfinite(values["bias"])
    }
    if bias_values:
        closest = _leaders(bias_values, higher=False)
        lines.append(
            f"|BIAS−1| 最小的是 {'、'.join(sorted(closest))}"
            f"（{fmt_number(min(bias_values.values()))}），频率偏差在全等级上最接近无偏。"
        )
    # 综合 TS 在显示精度上恒为 0 的模型，IQR 也恒为 0——不能读成「最均衡」。
    iqr_values = {
        name: values["iqr"] for name, values in overview.items()
        if np.isfinite(values["iqr"]) and name not in zero_skill
    }
    if iqr_values:
        stable = _leaders(iqr_values, higher=False)
        text = (
            f"跨时效 IQR 中位数最小的是 {'、'.join(sorted(stable))}"
            f"（{fmt_number(min(iqr_values.values()))}），各时效的技巧最均衡"
        )
        if zero_skill:
            text += f"（{'、'.join(sorted(zero_skill))} 的 TS 恒为 0，不参与这一项）"
        lines.append(text + "。")
    rank_pool = {name: value for name, value in ranks.items() if np.isfinite(value)}
    if rank_pool:
        rank_best = min(rank_pool, key=rank_pool.get)
        detail = "、".join(
            f"{name} {fmt_number(value, 2)}" for name, value in sorted(rank_pool.items(), key=lambda item: item[1])
        )
        lines.append(
            f"按等级×时效逐格排名的平均名次（1 = 每格最好）：{detail}；"
            f"最好的是 {rank_best}。"
        )
    leaders_all = {
        "综合 TS": set(_leaders({n: v["ts"] for n, v in overview.items()})),
        "强降水 TS": set(_leaders(heavy_values)) if heavy_values else set(),
        "无偏": set(_leaders(bias_values, higher=False)) if bias_values else set(),
        "稳定": set(_leaders(iqr_values, higher=False)) if iqr_values else set(),
    }
    leaders_all = {key: value for key, value in leaders_all.items() if value}
    if leaders_all:
        union = set().union(*leaders_all.values())
        if len(union) == 1:
            lines.append(
                f"四个维度由同一个模型 {'、'.join(sorted(union))} 领先，模型之间没有互补性。"
            )
        else:
            lines.append(
                "四个维度的领先者不是同一个模型，说明各模型的能力形状不同，"
                "单看综合 TS 会掩盖这些差别。"
            )
    return lines


def _analysis_grades(
    models: Dict[str, pd.DataFrame],
    summaries: Dict[str, Dict[str, object]],
    primary: str,
) -> List[str]:
    names = list(models)
    grades = ordered_grades(list(models.values()))
    lines: List[str] = []
    for grade in grades:
        values: Dict[str, float] = {}
        for name in names:
            entry = next(
                (item for item in summaries[name]["grades"] if item["grade"] == grade), None
            )
            if entry is not None:
                values[name] = float(entry["TS_sum"])
        finite = {name: value for name, value in values.items() if np.isfinite(value)}
        if not finite:
            lines.append(f"{grade}mm：各模型都没有有效 TS，该档不可评定。")
            continue
        leaders = _leaders(values)
        top = max(finite.values())
        others = [value for name, value in finite.items() if name not in leaders]
        if len(finite) > 1 and others:
            runner = max(others)
            lines.append(
                f"{grade}mm：{'、'.join(sorted(leaders))} 领先（TS {fmt_number(top)}），"
                f"比次优高 {fmt_number(top - runner)}。"
            )
        elif len(leaders) > 1:
            lines.append(
                f"{grade}mm：{'、'.join(sorted(leaders))} 并列领先（TS {fmt_number(top)}）。"
            )
        else:
            lines.append(f"{grade}mm：只有 {next(iter(finite))} 有有效 TS（{fmt_number(top)}）。")

    entries = summaries[primary]["grades"]
    usable = [entry for entry in entries if np.isfinite(entry.get("TS", np.nan))]
    if len(usable) > 1:
        steps = [
            (usable[index], usable[index + 1], usable[index]["TS"] - usable[index + 1]["TS"])
            for index in range(len(usable) - 1)
        ]
        left, right, drop = max(steps, key=lambda step: step[2])
        if drop > 0:
            lines.append(
                f"{primary} 的相邻量级落差最大的是 {left['grade']}→{right['grade']}"
                f"（{fmt_number(left['TS'])} → {fmt_number(right['TS'])}，掉 {fmt_number(drop)}）。"
            )
        rises = [step for step in steps if step[2] < -0.005]
        if rises:
            where = "、".join(f"{a['grade']}→{b['grade']}" for a, b, _ in rises)
            lines.append(
                f"{primary} 的 TS 并非随量级单调下降：{where} 处出现回升，"
                f"该位置的样本波动已经盖过真实差异。"
            )
        first, last = usable[0], usable[-1]
        if first["events"] > 0 and 0 < last["events"] < first["events"]:
            ratio = first["events"] / last["events"]
            lines.append(
                f"{primary} 的观测事件数从 {first['grade']} 的 {first['events']:,} 次"
                f"收缩到 {last['grade']} 的 {last['events']:,} 次"
                + (f"（约 1/{ratio:.0f}）" if ratio >= 10 else "")
                + "，高量级的 TS 小数位不宜过度解读。"
            )
    return lines


def _analysis_leads(models: Dict[str, pd.DataFrame]) -> List[str]:
    table = _lead_mean_values(models)
    if table.empty:
        return ["没有可用的时效数据。"]
    leads = [float(lead) for lead in table.index]
    lines: List[str] = []
    for name in table.columns:
        series = table[name].dropna()
        if len(series) < 2:
            continue
        first, last = float(series.index[0]), float(series.index[-1])
        start, end = float(series.iloc[0]), float(series.iloc[-1])
        drop = start - end
        relative = drop / start if start else float("nan")
        below = series[series < USABLE_TS]
        effective = f"在 {float(below.index[0]):g}h 首次跌破 {USABLE_TS:g}" if len(below) else f"全程未跌破 {USABLE_TS:g}"
        lines.append(
            f"{name}：综合 TS 从 {first:g}h 的 {fmt_number(start)} 降到 {last:g}h 的 "
            f"{fmt_number(end)}（−{fmt_number(drop)}，相对 −{relative:.0%}），{effective}。"
        )
    if len(leads) >= 2:
        first_values = {
            name: float(table.loc[leads[0], name])
            for name in table.columns
            if np.isfinite(table.loc[leads[0], name])
        }
        last_values = {
            name: float(table.loc[leads[-1], name])
            for name in table.columns
            if np.isfinite(table.loc[leads[-1], name])
        }
        if first_values and last_values:
            first_leaders = _leaders(first_values)
            last_leaders = _leaders(last_values)
            lines.append(
                f"{leads[0]:g}h 由 {'、'.join(sorted(first_leaders))} 领先"
                f"（{fmt_number(max(first_values.values()))}），"
                f"{leads[-1]:g}h 由 {'、'.join(sorted(last_leaders))} 领先"
                f"（{fmt_number(max(last_values.values()))}）；"
                + (
                    "领先者随时效发生变化。"
                    if first_leaders != last_leaders
                    else "领先者在首末时效上一致。"
                )
            )
    decays = {}
    for name in table.columns:
        series = table[name].dropna()
        if len(series) >= 2 and series.iloc[0]:
            decays[name] = float((series.iloc[0] - series.iloc[-1]) / series.iloc[0])
    if len(decays) > 1:
        fastest = max(decays, key=decays.get)
        slowest = min(decays, key=decays.get)
        if f"{decays[fastest]:.0%}" == f"{decays[slowest]:.0%}":
            lines.append(
                f"各模型的相对衰减相当（都在 {decays[slowest]:.0%} 左右），"
                f"看不出谁抗衰减。"
            )
        else:
            lines.append(
                f"相对衰减最快的是 {fastest}（{decays[fastest]:.0%}），"
                f"最慢的是 {slowest}（{decays[slowest]:.0%}）。"
            )
    return lines


def _analysis_metric(
    models: Dict[str, pd.DataFrame],
    summaries: Dict[str, Dict[str, object]],
    *,
    key: str,
    label: str,
    higher: bool = True,
    closer: bool = False,
) -> List[str]:
    """单个分等级指标：领先次数 + 首末量级对照（POD / FAR / BIAS 共用）。"""
    names = list(models)
    if not names:
        return []
    grades = [str(entry["grade"]) for entry in summaries[names[0]]["grades"]]
    values: Dict[str, Dict[str, float]] = {
        name: {
            str(entry["grade"]): float(entry.get(key, np.nan))
            for entry in summaries[name]["grades"]
        }
        for name in names
    }
    counts = {name: 0 for name in names}
    total = 0
    for grade in grades:
        row = {
            name: values[name].get(grade, np.nan) for name in names
        }
        row = {name: value for name, value in row.items() if np.isfinite(value)}
        if not row:
            continue
        total += 1
        for name in (_closest_to_one(row) if closer else _leaders(row, higher=higher)):
            counts[name] += 1
    if total == 0:
        return [f"各等级都没有有效 {label}，本节无法比较。"]

    lines: List[str] = []
    best_count = max(counts.values())
    best = [name for name in names if counts[name] == best_count]
    if len(best) == len(names):
        lines.append(f"{label}：各模型在 {total} 个等级上互有胜负，没有一致领先者。")
    else:
        direction = "最接近 1 的" if closer else ("最高" if higher else "最低")
        detail = "、".join(f"{name} {counts[name]}/{total}" for name in names)
        lines.append(f"{label} {direction}次数：{detail}。")

    def describe(grade: str) -> Optional[str]:
        row = {
            name: values[name].get(grade, np.nan) for name in names
        }
        row = {name: value for name, value in row.items() if np.isfinite(value)}
        if not row:
            return None
        if closer:
            winners = _closest_to_one(row)
            value = min(row.values(), key=lambda item: abs(item - 1.0))
            worst = max(row, key=lambda name: abs(row[name] - 1.0))
            return (
                f"{grade}mm 上最接近无偏的是 {'、'.join(sorted(winners))}"
                f"（BIAS {fmt_number(value)}），最偏的是 {worst}"
                f"（{fmt_number(row[worst])}）。"
            )
        winners = _leaders(row, higher=higher)
        value = max(row.values()) if higher else min(row.values())
        word = "最高" if higher else "最低"
        return (
            f"{grade}mm 上 {label} {word}的是 {'、'.join(sorted(winners))}"
            f"（{fmt_number(value)}）。"
        )

    for grade in ([grades[0], grades[-1]] if len(grades) > 1 else grades):
        line = describe(grade)
        if line:
            lines.append(line)
    return lines


def _analysis_far_bias(
    models: Dict[str, pd.DataFrame], summaries: Dict[str, Dict[str, object]]
) -> List[str]:
    """FAR 与漏报率的对照：误差到底偏向哪一侧。"""
    lines: List[str] = []
    for name in models:
        entries = [
            entry
            for entry in summaries[name]["grades"]
            if np.isfinite(entry.get("FAR", np.nan)) and np.isfinite(entry.get("漏报率", np.nan))
        ]
        if not entries:
            lines.append(f"{name}：各等级的 FAR / 漏报率都不可用，无法判断误差形态。")
            continue
        entry = entries[-1]
        far_text, miss_text = fmt_number(entry["FAR"]), fmt_number(entry["漏报率"])
        if far_text == miss_text:
            kind = "空报与漏报相当"
        elif entry["FAR"] > entry["漏报率"]:
            kind = "以空报为主"
        else:
            kind = "以漏报为主"
        lines.append(
            f"{name}：{entry['grade']}mm 的 FAR {far_text}、漏报率 {miss_text}，{kind}。"
        )
    return lines


def _analysis_stability(
    models: Dict[str, pd.DataFrame], primary: str
) -> List[str]:
    per_model: Dict[str, List[Dict[str, object]]] = {}
    for name, frame in models.items():
        rows = [row for row in box_stats_by_grade(frame, metric="TS") if row["n"]]
        if rows:
            per_model[name] = rows
    if not per_model:
        return ["各模型在各量级上都没有有效样本，无法评估跨时效稳定性。"]

    lines: List[str] = []
    # 各量级中位数在显示精度上仍是 0.000 的模型（几乎不报事件）IQR 也恒为 0，
    # 不能读成「跨时效最稳定」。
    skilled = {
        name: rows
        for name, rows in per_model.items()
        if any(
            np.isfinite(row["median"]) and fmt_number(row["median"]) != "0.000"
            for row in rows
        )
    }
    if skilled and len(skilled) < len(per_model):
        skipped = "、".join(name for name in per_model if name not in skilled)
        lines.append(
            f"{skipped} 的 TS 在各量级上恒为 0（几乎不报事件），IQR 也恒为 0，"
            f"不参与稳定性排序。"
        )
    pool = skilled or per_model
    spread = {
        name: float(np.median([row["iqr"] for row in rows]))
        for name, rows in pool.items()
    }
    ranked = sorted(spread, key=lambda name: spread[name])
    if len(ranked) == 1:
        if len(models) == 1:
            lines.append(
                f"单模型评估：{ranked[0]} 各量级 IQR 的中位数为 "
                f"{fmt_number(spread[ranked[0]])}。"
            )
        else:
            lines.append(
                f"参与稳定性排序的只有 {ranked[0]}，各量级 IQR 的中位数为 "
                f"{fmt_number(spread[ranked[0]])}。"
            )
    elif spread[ranked[-1]] - spread[ranked[0]] < 0.01:
        lines.append(
            f"各模型的跨时效稳定性接近（IQR 中位数 {fmt_number(spread[ranked[0]])}–"
            f"{fmt_number(spread[ranked[-1]])}，极差不足 0.01），稳定性不构成区分度。"
        )
    else:
        lines.append(
            f"按各量级 IQR 的中位数排序，跨时效最稳定的是 {ranked[0]}"
            f"（{fmt_number(spread[ranked[0]])}），最不稳定的是 {ranked[-1]}"
            f"（{fmt_number(spread[ranked[-1]])}）。"
        )

    outliers = sorted(
        {float(lead) for row in per_model.get(primary, []) for lead in row["outlier_leads"]}
    )
    if outliers and primary in models:
        all_leads = sorted(float(value) for value in pd.unique(models[primary]["lead_h"]))
        middle = (all_leads[0] + all_leads[-1]) / 2
        early = sum(1 for lead in outliers if lead <= middle)
        where = (
            "短时效段" if early * 2 > len(outliers)
            else ("长时效段" if early * 2 < len(outliers) else "首尾两端")
        )
        shown = "、".join(f"{lead:g}h" for lead in outliers[:6])
        lines.append(
            f"{primary} 的离群时效共 {len(outliers)} 个（{shown}"
            + ("…" if len(outliers) > 6 else "")
            + f"），偏向{where}。"
        )
    elif primary in models:
        lines.append(f"{primary} 的各量级都没有离群时效，TS 的跨时效分布连续。")

    flips = []
    for grade in ordered_grades(list(models.values())):
        medians: Dict[str, float] = {}
        means: Dict[str, float] = {}
        for name, frame in models.items():
            subset = frame[frame["grade"].astype(str) == grade]
            values = pd.to_numeric(subset["TS"], errors="coerce").dropna()
            if values.empty:
                continue
            medians[name] = float(values.median())
            means[name] = float(values.mean())
        if len(medians) < 2:
            continue
        median_leaders = _leaders(medians)
        mean_leaders = _leaders(means)
        # 按显示精度并列时不算翻转：两个量级都只剩噪声的档（各家显示值一样）
        # 不该被写成「谁领先」。
        if len(median_leaders) != 1 or len(mean_leaders) != 1:
            continue
        by_median = next(iter(median_leaders))
        by_mean = next(iter(mean_leaders))
        if by_median != by_mean:
            flips.append((grade, by_median, by_mean))
    if flips:
        detail = "、".join(
            f"{grade}（中位数 {left} 领先、等权平均 {right} 领先）"
            for grade, left, right in flips[:3]
        )
        lines.append(
            f"**注意口径**：{detail}。本节表里的统计量是中位数/IQR，4.1 节表里的是等权平均，"
            f"两者给出不同的领先者是正常现象，不可互相替代。"
        )
    return lines


def _analysis_performance(models: Dict[str, pd.DataFrame], lead: float) -> List[str]:
    values: Dict[str, Tuple[float, float]] = {}
    for name, frame in models.items():
        subset = frame[pd.to_numeric(frame["lead_h"], errors="coerce") == lead]
        pod = pd.to_numeric(subset["POD"], errors="coerce").dropna() if "POD" in subset else pd.Series(dtype=float)
        far = pd.to_numeric(subset["FAR"], errors="coerce").dropna() if "FAR" in subset else pd.Series(dtype=float)
        if len(pod) and len(far):
            values[name] = (float(pod.mean()), float(far.mean()))
    if not values:
        return [f"{lead:g}h 时效上没有可用的 POD/FAR，无法画出性能图。"]
    distance = {
        name: float(np.hypot(1.0 - pod, far)) for name, (pod, far) in values.items()
    }
    lines: List[str] = []
    ranked = sorted(distance, key=lambda name: distance[name])
    if len(ranked) == 1:
        name = ranked[0]
        lines.append(
            f"单模型评估：{name} 在 {lead:g}h 的等级平均 POD {fmt_number(values[name][0])}、"
            f"FAR {fmt_number(values[name][1])}，距右上角 {fmt_number(distance[name])}。"
        )
        return lines
    if distance[ranked[-1]] - distance[ranked[0]] < 0.01:
        lines.append(
            f"各模型到右上角的距离接近（{fmt_number(distance[ranked[0]])}–"
            f"{fmt_number(distance[ranked[-1]])}，极差不足 0.01），性能图的相对位置不构成区分度。"
        )
    else:
        lines.append(
            f"按到右上角的距离（√((1−POD)² + FAR²)）排序，{ranked[0]} 最近"
            f"（{fmt_number(distance[ranked[0]])}），{ranked[-1]} 最远"
            f"（{fmt_number(distance[ranked[-1]])}）。"
        )
    best_pod = max(values, key=lambda name: values[name][0])
    best_far = min(values, key=lambda name: values[name][1])
    lines.append(
        f"POD 最高的是 {best_pod}（{fmt_number(values[best_pod][0])}），"
        f"FAR 最低的是 {best_far}（{fmt_number(values[best_far][1])}）；"
        + (
            "同一模型同时拿下这两项，命中与空报没有此消彼长。"
            if best_pod == best_far
            else "两项不是同一个模型，说明命中与空报之间存在权衡。"
        )
    )
    detail = "、".join(
        f"{name}（POD {fmt_number(pod)} / FAR {fmt_number(far)}）"
        for name, (pod, far) in sorted(values.items(), key=lambda item: distance[item[0]])
    )
    lines.append(f"各模型的等级平均位置：{detail}。")
    return lines


def _analysis_lead_rank(models: Dict[str, pd.DataFrame]) -> List[str]:
    table = _lead_mean_values(models)
    if table.empty:
        return ["没有可用的时效数据。"]
    lines: List[str] = []
    winners: List[Tuple[float, Tuple[str, ...]]] = []
    for lead, row in table.iterrows():
        values = {name: float(row.get(name, np.nan)) for name in table.columns}
        leaders = _leaders(values)
        if leaders:
            winners.append((float(lead), tuple(sorted(leaders))))
    counts: Dict[str, int] = {}
    for _, leaders in winners:
        for name in leaders:
            counts[name] = counts.get(name, 0) + 1
    if len(table.columns) == 1:
        name = str(table.columns[0])
        lines.append(
            f"单模型评估：{name} 在 {len(winners)} 个时效上都有有效 TS，"
            f"各时效综合 TS 见上表。"
        )
        return lines
    if counts:
        detail = "、".join(
            f"{name} {count} 个" for name, count in sorted(counts.items(), key=lambda item: -item[1])
        )
        lines.append(
            f"在 {len(winners)} 个有时效可比的行里，各模型拿到头名的次数：{detail}。"
        )
    switches = [
        index for index in range(1, len(winners)) if winners[index][1] != winners[index - 1][1]
    ]
    if switches:
        index = switches[0]
        before, after = winners[index - 1], winners[index]
        lines.append(
            f"领先者在 {before[0]:g}h 与 {after[0]:g}h 之间由 {'、'.join(before[1])} 变为 "
            f"{'、'.join(after[1])}（整段共 {len(switches)} 处变化）。"
        )
    else:
        lines.append("领先者在全部时效上保持不变。")
    return lines


# --------------------------------------------------------------------------- #
# 分节渲染
# --------------------------------------------------------------------------- #
def _section_summary(
    model_name: str,
    models: Dict[str, pd.DataFrame],
    summaries: Dict[str, Dict[str, object]],
    overview: Dict[str, Dict[str, float]],
    ranks: Dict[str, float],
    *,
    sources: Optional[Dict[str, str]],
    coverage: Optional[Dict[str, int]],
    change_description: Optional[str],
    ensemble: bool,
    members: Optional[int],
    has_probability: bool,
) -> List[str]:
    summary = summaries[model_name]
    leads = summary["leads"]
    window = None
    frame = models[model_name]
    if "window_h" in frame.columns and frame["window_h"].notna().any():
        window = float(frame["window_h"].dropna().iloc[0])
    names = "、".join(models)
    lines = ["# 1. 摘要", ""]
    lines.append(
        f"本报告评估 {names} 共 {len(models)} 个模型"
        f"（{model_name} 的来源：{source_of(sources, model_name)}），"
        + (f"集合每例 {int(members)} 个成员，" if (ensemble and members) else "")
        + "".join(_coverage_sentence(coverage))
        + "评估口径为站点分类检验（"
        + (f"{window:g}h 累积窗口，" if window else "")
        + f"时效 {leads[0]:g}–{leads[-1]:g}h 共 {len(leads)} 个，"
        + f"每份 {summary['n_rows']} 条记录），指标为 TS、POD、FAR、漏报率与 BIAS。"
        + (f"本次改动：{change_description}。" if change_description else "")
        + f"报告生成时间：{datetime.now():%Y-%m-%d %H:%M}。"
    )
    lines.append("")
    lines.append("".join(diagnose(summary)) + _comparison_sentence(models, overview, ranks))
    lines.append("")
    lines.append(f"**业务定性**：{business_conclusion(summary['grades'])}")
    lines.append("")
    if ensemble:
        lines.append(
            f"集合口径：TS / POD / FAR / 漏报率 / BIAS 均在{_members_phrase(members)}的"
            f"**平均场**上计算（确定性口径），不是逐成员评分后再平均；"
            + (
                "成员离散信息只在 4.7 节的概率评分里体现。"
                if has_probability
                else "本批没有概率评分表，成员离散信息在这份报告里没有体现。"
            )
        )
        lines.append("")
    lines.append(
        "本报告中的BIAS指频率偏差（预报事件数/观测事件数），BIAS > 1 表示预报事件偏多、"
        "< 1 表示偏少，与平均误差不是同一个量。"
    )
    lines.append("")
    return lines


def _section_samples(
    models: Dict[str, pd.DataFrame],
    summaries: Dict[str, Dict[str, object]],
    coverage: Optional[Dict[str, int]],
) -> List[str]:
    lines = ["# 2. 样本口径与数据质量", ""]
    rows: List[List[str]] = []
    for name, frame in models.items():
        summary = summaries[name]
        leads = summary["leads"]
        window = None
        if "window_h" in frame.columns and frame["window_h"].notna().any():
            window = float(frame["window_h"].dropna().iloc[0])
        thresholds = sorted({
            float(value)
            for value in pd.to_numeric(
                frame.get("threshold_mm", pd.Series(dtype=float)), errors="coerce"
            ).dropna()
        })
        pairs = coverage.get(name) if coverage else None
        rows.append([
            name,
            fmt_number(window, 0),
            f"{leads[0]:g}–{leads[-1]:g}",
            str(len(leads)),
            str(len(summary["grades"])),
            "/".join(f"{value:g}" for value in thresholds) or "—",
            str(summary["n_rows"]),
            f"{int(pairs):,}" if pairs else "—",
        ])
    lines += _table_title(f"{cn_number(len(models))}模型样本覆盖")
    lines += table_lines(
        ["模型", "累积窗口(h)", "时效范围(h)", "时效数", "降水等级数",
         "阈值(mm)", "记录数", "最小阈值档配对样本"],
        rows,
    )
    lines.append("")
    lines += bullet_lines(_coverage_sentence(coverage) + _consistency_sentences(models))
    lines.append("")
    lines.append(
        "> 宽表不记录起报日期，本节无法给出日期跨度；样本量一律以**配对数**为准，"
        "各份来源的起报日数差异只能从配对样本的比值反推。"
    )
    lines.append("")
    return lines


def _section_metrics(ensemble: bool = False) -> List[str]:
    rows = [
        ["TS", "hits/(hits+misses+false_alarms)", "越高越好"],
        ["POD", "hits/(hits+misses)", "越高越好"],
        ["FAR", "false_alarms/(hits+false_alarms)", "越低越好"],
        ["漏报率", "1−POD", "越低越好"],
        ["BIAS", "(hits+false_alarms)/(hits+misses)，频率偏差", "越接近 1 越好"],
        ["综合TS", "各降水等级的 TS 等权平均（先合计列联表再算比率）", "越高越好"],
    ]
    if ensemble:
        # 集合专属：成员离散信息只在这三个指标里出现。
        rows += [
            ["AROC", "概率评分：ROC 曲线下面积（逐成员概率）", "越高越好，0.5 为无技巧"],
            ["BS", "Brier 分数（概率预报的均方误差）", "越低越好"],
            ["BSS", "1 − BS/BS_ref，相对气候基准的 Brier 技巧", "越高越好，≤ 0 无正技巧"],
        ]
    lines = ["# 3. 评估指标与解释", ""]
    lines += _table_title("指标口径")
    lines += table_lines(["指标", "定义/口径", "方向"], rows)
    lines.append("")
    lines.append(
        "「全时效平均」指**各时效等权平均**，不是按样本数加权；TS 为 “—” 表示该等级"
        "没有有效配对或观测事件为 0，与「评分为 0.000」含义不同。"
    )
    lines.append("")
    lines.append(
        "TS = f(POD, FAR)（1/TS = 1/POD + 1/(1−FAR) − 1），三者是同一件事的三种写法；"
        "总体表只保留综合 TS，POD / FAR 的绝对水平分别放在 4.2 / 4.3 节读。"
    )
    lines.append("")
    if ensemble:
        lines.append(
            "概率评分使用独立的累积窗口与阈值，与上面 TS 那套不可混比；"
            "AROC / BS / BSS 的口径与数值见 4.7 节。"
        )
        lines.append("")
    return lines


def _section_overall(
    models: Dict[str, pd.DataFrame],
    overview: Dict[str, Dict[str, float]],
    ranks: Dict[str, float],
    heavy: Optional[float],
    artifacts: Optional[Dict[str, Path]],
    numbering: _Numbering,
    caption: str,
    grades_caption: str,
) -> List[str]:
    lines = ["# 4. 总体评估结果", ""]
    lines.append(
        "本章先给总体排序与基准时效上的分等级对照，再按 TS / POD / FAR / BIAS 四个"
        "指标各占一节展开，最后是跨时效稳定性、性能图与（集合报告的）概率评分。"
    )
    lines.append("")
    labels = {
        "ts": "综合TS",
        "heavy": f"强降水TS（≥{heavy:g}mm）" if heavy is not None else "强降水TS",
        "bias": "平均BIAS偏离度",
        "iqr": "跨时效IQR（全等级平均）",
        "rank": "平均名次",
    }
    order = sorted(
        models,
        key=lambda name: (
            _sort_value(overview[name]["ts"]),
            -_sort_value(ranks.get(name, float("nan"))),
        ),
        reverse=True,
    )
    rank_numbers: Dict[str, int] = {}
    for name in order:
        rank_numbers[name] = 1 + sum(
            1 for other in order
            if np.isfinite(overview[other]["ts"])
            and np.isfinite(overview[name]["ts"])
            and overview[other]["ts"] > overview[name]["ts"] + 1e-12
        )
    rows: List[List[str]] = []
    for name in order:
        values = overview[name]
        rows.append([
            f"{name}（第{rank_numbers[name]}）",
            fmt_number(values["ts"]),
            fmt_number(values["heavy"]),
            fmt_number(values["bias"]),
            fmt_number(values["iqr"]),
            fmt_number(ranks.get(name, float("nan")), 2),
        ])
    lines += _table_title(f"{cn_number(len(models))}模型总体指标")
    lines += table_lines(
        ["模型（相对排序）", labels["ts"], labels["heavy"], labels["bias"],
         labels["iqr"], labels["rank"]],
        rows,
    )
    lines.append("")
    lines += bullet_lines(_analysis_overview(models, overview, ranks, heavy))
    lines.append("")
    lines += _figure_block(artifacts, numbering, "overall_heatmap", caption)
    lines += _figure_block(artifacts, numbering, "metrics_at_lead", grades_caption)
    return lines


def _section_ts(
    models: Dict[str, pd.DataFrame],
    summaries: Dict[str, Dict[str, object]],
    primary: str,
    artifacts: Optional[Dict[str, Path]],
    numbering: _Numbering,
    caption: str,
) -> List[str]:
    lines = ["## 4.1 TS（技巧评分）", ""]
    lines.append(
        "TS 同时惩罚漏报与空报，是这一批的主排序指标。下表按降水等级展开，"
        "其后是主模型的列联表计数与各模型的时效衰减——技巧掉到不可用之前能推多远。"
    )
    lines.append("")
    lines += _table_title("各模型各降水等级的全时效平均TS")
    lines.append(
        _grade_metric_table(
            models, summaries, key="TS_sum", best_label="最佳"
        )
    )
    lines.append("")
    lines += _table_title(f"{primary}的逐等级列联表计数（全时效合计）")
    lines.append(_grade_counts_table(summaries[primary]))
    lines.append("")
    lines += _table_title("各模型的时效衰减（等级等权平均TS）")
    lines.append(_lead_decay_table(models))
    lines.append("")
    lines += _figure_block(artifacts, numbering, "ts_vs_lead", caption)
    lines += bullet_lines(
        _analysis_grades(models, summaries, primary) + _analysis_leads(models)
    )
    lines.append("")
    return lines


def _section_pod(
    models: Dict[str, pd.DataFrame],
    summaries: Dict[str, Dict[str, object]],
    artifacts: Optional[Dict[str, Path]],
    numbering: _Numbering,
    caption: str,
) -> List[str]:
    lines = ["## 4.2 POD（命中率）", ""]
    lines.append(
        "POD = hits/(hits+misses)，回答「实际下了的雨，预报到了多少」。POD 高不代表"
        "技巧好——把所有格点都报成有雨同样能把 POD 做高，必须与 4.3 的 FAR 一起读。"
    )
    lines.append("")
    lines += _table_title("各模型各降水等级的POD（全时效平均，越高越好）")
    lines.append(
        _grade_metric_table(models, summaries, key="POD_sum", best_label="最佳")
    )
    lines.append("")
    lines += _figure_block(artifacts, numbering, "pod_vs_lead", caption)
    lines += bullet_lines(
        _analysis_metric(models, summaries, key="POD", label="POD", higher=True)
    )
    lines.append("")
    return lines


def _section_far(
    models: Dict[str, pd.DataFrame],
    summaries: Dict[str, Dict[str, object]],
    artifacts: Optional[Dict[str, Path]],
    numbering: _Numbering,
    caption: str,
) -> List[str]:
    lines = ["## 4.3 FAR（空报率）", ""]
    lines.append(
        "FAR = false_alarms/(hits+false_alarms)，回答「报了雨的格点里，有多少其实没下」，"
        "越低越好。把它与漏报率（1−POD）放在一起看，才知道误差偏向空报还是漏报。"
    )
    lines.append("")
    lines += _table_title("各模型各降水等级的FAR（全时效平均，越低越好）")
    lines.append(
        _grade_metric_table(
            models, summaries, key="FAR_sum", best_label="最佳", lower=True
        )
    )
    lines.append("")
    lines += _figure_block(artifacts, numbering, "far_vs_lead", caption)
    lines += bullet_lines(
        _analysis_metric(models, summaries, key="FAR", label="FAR", higher=False)
        + _analysis_far_bias(models, summaries)
    )
    lines.append("")
    return lines


def _section_bias(
    models: Dict[str, pd.DataFrame],
    summaries: Dict[str, Dict[str, object]],
    artifacts: Optional[Dict[str, Path]],
    numbering: _Numbering,
    caption: str,
) -> List[str]:
    lines = ["## 4.4 BIAS（频率偏差）", ""]
    lines.append(
        "BIAS = (hits+false_alarms)/(hits+misses)，1 为无偏、>1 为报得偏多、<1 为偏少。"
        "它只看次数不看强度，量级正确与否由 TS 与 FAR 负责。"
    )
    lines.append("")
    lines += _table_title("各模型各降水等级的BIAS（全时效平均，越接近1越好）")
    lines.append(
        _grade_metric_table(
            models, summaries, key="BIAS_sum", best_label="最接近1", closer=True
        )
    )
    lines.append("")
    lines += _figure_block(artifacts, numbering, "bias_vs_lead", caption)
    lines += bullet_lines(
        _analysis_metric(
            models, summaries, key="BIAS", label="BIAS", closer=True
        )
    )
    lines.append("")
    return lines


def _section_stability(
    models: Dict[str, pd.DataFrame],
    primary: str,
    artifacts: Optional[Dict[str, Path]],
    numbering: _Numbering,
    caption: str,
) -> List[str]:
    lines = ["## 4.5 跨时效稳定性", ""]
    lines.append(
        "把同一模型、同一量级在各时效上的 TS 视为一个样本，用箱线图刻画其离散程度——"
        "平均技巧相同的两个模型，稳定性可能差很远。"
    )
    lines.append("")
    lines += _table_title("各模型 × 各降水等级 TS 的跨时效分布统计量")
    lines.append(_box_table(models))
    lines.append("")
    lines.append(_BOX_NOTE)
    lines.append("")
    lines += _figure_block(artifacts, numbering, "TS_boxplot", caption)
    lines += bullet_lines(_analysis_stability(models, primary))
    lines.append("")
    return lines


def _section_performance(
    models: Dict[str, pd.DataFrame],
    lead: float,
    artifacts: Optional[Dict[str, Path]],
    numbering: _Numbering,
    caption: str,
) -> List[str]:
    lines = ["## 4.6 性能图", ""]
    lines.append(
        f"性能图把 POD 与 1−FAR 放在同一平面上：点越靠近右上角，命中越多、空报越少。"
        f"下表取 {lead:g}h 时效、各等级等权平均。"
    )
    lines.append("")
    lines += _table_title(f"{lead:g}h 时效各模型的命中率与空报率")
    lines.append(_performance_table(models, lead))
    lines.append("")
    lines += _figure_block(artifacts, numbering, "performance_diagram", caption)
    lines += bullet_lines(_analysis_performance(models, lead))
    lines.append("")
    return lines


def _section_probability(
    probability: Optional[Dict[str, pd.DataFrame]],
    artifacts: Optional[Dict[str, Path]],
    numbering: _Numbering,
    captions: Dict[str, str],
    *,
    members: Optional[int],
    ts_window: Optional[float],
    ts_thresholds: str,
) -> List[str]:
    """「4.7 概率评分（AROC / BSS）」——集合报告固定占位，确定性不调用。

    AROC 与 BSS 各占一张图（两张分开出），缺表时两个图号一起保留。
    """
    lines = ["## 4.7 概率评分（AROC / BSS）", ""]
    if not probability:
        numbering.reserve("aroc_vs_lead")
        numbering.reserve("bss_vs_lead")
        lines += [
            "**本批未出。** 产物里没有 `diagnostics/probability_wide.csv`；"
            "这份表由独立能力 `weather_ts_ens_prob` 写出，只有配置里写 "
            '`pipeline=["weather_ts_ens", "weather_ts_ens_prob"]` 才会落盘。',
            "",
            "因此本节无表、无图，保留节位与图号以便将来补数后直接填入。",
            "",
        ]
        return lines

    summaries = {name: probability_summary(frame) for name, frame in probability.items()}
    primary = next(iter(probability))
    window = _probability_window(probability[primary])
    window_text = f"{window:g}h" if window else "概率"
    lines[0] = f"## 4.7 概率评分（AROC / BSS，{window_text} 累积窗口）"

    leads = sorted(
        float(value)
        for value in pd.to_numeric(probability[primary]["lead_h"], errors="coerce").dropna()
    )
    span = f"{leads[0]:g}–{leads[-1]:g}h" if leads else "各时效"

    lines += [
        f"前面各节评的都是{_members_phrase(members)}的**平均场**，成员之间的离散信息"
        f"一点没进指标——一个成员高度一致、另一个成员四散分布的集合，前面的分数"
        f"可以完全相同。本节改用全体成员的概率分布：AROC 衡量预报概率把事件与非事件"
        f"分开的能力（0.5 为无技巧），BSS 衡量相对气候基准的 Brier 技巧（>0 才算有正技巧）。",
        "",
        f"**本节与前面各节不同口径**：概率评分用 {window_text} 累积窗口、"
        f"阈值 {_threshold_list(probability[primary])}，与前面 "
        + (f"{ts_window:g}h / " if ts_window else "")
        + f"{ts_thresholds} 既不可比，也不要画进同一张图。",
        "",
    ]
    lines += _table_title("各降水等级的概率评分（各时效等权平均；AVG 为可算阈值的等权平均）")
    lines.append(_probability_table(summaries, with_model=len(summaries) > 1))
    lines.append("")
    lines.append(_probability_note(window))
    lines.append("")
    lines += _figure_block(artifacts, numbering, "aroc_vs_lead", captions["aroc_vs_lead"])
    lines += _figure_block(artifacts, numbering, "bss_vs_lead", captions["bss_vs_lead"])
    lines += bullet_lines(analysis_probability(summaries))
    lines.append("")
    return lines


def _section_lead_rank(
    models: Dict[str, pd.DataFrame],
    artifacts: Optional[Dict[str, Path]],
    numbering: _Numbering,
    caption: str,
) -> List[str]:
    lines = ["# 5. 分时效表现", ""]
    lines.append(
        "把每个时效、各降水等级的 TS 做等权平均，得到该时效的综合 TS——"
        "它回答「换一个预报时效，领先者会不会换人」。"
    )
    lines.append("")
    lines += _table_title("各时效各模型的综合TS")
    lines.append(_lead_mean_table(models))
    lines.append("")
    lines += _figure_block(artifacts, numbering, "lead_dot", caption)
    lines += bullet_lines(_analysis_lead_rank(models))
    lines.append("")
    return lines


def _section_conclusions(
    models: Dict[str, pd.DataFrame],
    summaries: Dict[str, Dict[str, object]],
    primary: str,
    *,
    ensemble: bool,
    has_probability: bool,
    coverage: Optional[Dict[str, int]],
) -> List[str]:
    lines = ["# 6. 结论与建议", ""]
    lines += _table_title("改进建议")
    for index, item in enumerate(recommendations(summaries[primary]["grades"]), start=1):
        lines.append(f"{index}. {item}")
    lines.append("")
    lines += _table_title("口径与注意事项")
    lines.append("- **值缺失的含义**：TS 为 “—” 表示该等级没有有效配对或无事件，与“评分为 0”不同；")
    lines.append("- **BIAS 不是平均误差**：它是频率偏差，>1 表示预报事件偏多，<1 表示偏少；")
    if ensemble:
        lines.append(
            "- **前面各节评的是集合平均场**：TS / POD / FAR / 漏报率 / BIAS 都不含成员离散信息，"
            + ("成员分布要看 4.7 节；" if has_probability else "本批未跑概率评分；")
        )
        if has_probability:
            lines.append(
                "- **概率评分与 TS 不可混比**：两者的累积窗口与阈值都不同，AROC 还是逐成员概率口径；"
            )
    if coverage and len(coverage) > 1:
        values = [int(value) for value in coverage.values() if value]
        if values and min(values) < 0.9 * max(values):
            lines.append(
                "- **各模型的样本量并不相同**：第二节列的是各份实测配对数，"
                "差值来自起报日数而非口径；样本最少的那份名次波动最大，横向比高低前先看那一行；"
            )
    lines.append("- **跨时效不要直接平均后比较模型**：正式对比应使用共同有效样本与固定验证口径；")
    lines.append(
        "- 结果来自统一长表（`scores.csv` / `categorical_wide.csv`"
        + (
            "，概率评分另见同目录的 `probability_wide.csv`（老归档叫 `aroc_bss*.csv`）"
            if has_probability
            else ""
        )
        + "），逐行可追溯到具体时效与阈值。"
    )
    lines.append("")
    return lines


# ------------------------------------------------------------- 分纬度带（附 L）
def region_labels_from(root: Path) -> Dict[str, str]:
    """``{带名: 边界串}``，边界取自产物 ``manifest["regions"]``。"""
    from reporting.reports.det_report import region_labels

    try:
        return region_labels(Path(root))
    except ValueError:  # 没有 manifest.json，或它不是一个合法 JSON
        return {}


def _band_order(names: Sequence[str], labels: Optional[Dict[str, str]]) -> List[str]:
    """带名按**南界从低到高**排；取不到边界时退回按名字升序。"""
    def key(name: str):
        head = str((labels or {}).get(name, "")).split(",")[0].strip()
        try:
            return (0, float(head), name)
        except ValueError:
            return (1, 0.0, name)

    return sorted(names, key=key)


def _region_table(
    band_frames: Dict[str, pd.DataFrame], labels: Optional[Dict[str, str]]
) -> str:
    """带 × 降水等级 的 TS 表，末列是各带自己的「全等级等权平均」。"""
    from reporting.reports.det_report import region_display

    grades = [str(grade) for grade in ordered_grades(list(band_frames.values()))]
    header = ["纬度带"] + grades + ["综合"]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for name in _band_order(list(band_frames), labels):
        frame = band_frames[name]
        summary = summarize(frame)
        by_grade = {entry["grade"]: entry for entry in summary["grades"]}
        cells = [fmt_number(by_grade[grade]["TS"]) if grade in by_grade else "—" for grade in grades]
        lines.append(
            "| "
            + " | ".join(
                [region_display(name, labels or {})] + cells + [fmt_number(summary["overall_ts"])]
            )
            + " |"
        )
    return "\n".join(lines)


def _region_reading(
    band_frames: Dict[str, pd.DataFrame], labels: Optional[Dict[str, str]]
) -> str:
    """分带判读句：最强的带、最弱的带、差多少，以及头名是否随预报时效翻转。"""
    overall = {
        name: float(summarize(frame)["overall_ts"]) for name, frame in band_frames.items()
    }
    finite = {name: value for name, value in overall.items() if np.isfinite(value)}
    if len(finite) < 2:
        return "各带的有效样本不足以横向比较（至少要有两条带各自算出 TS）。"

    ranked = sorted(finite.items(), key=lambda item: item[1], reverse=True)

    def _tied(value: float) -> List[str]:
        key = fmt_number(value)
        return [name for name, item_value in ranked if fmt_number(item_value) == key]

    tops, bottoms = _tied(ranked[0][1]), _tied(ranked[-1][1])
    if tops == bottoms:
        text = (
            f"各带的全等级等权平均 TS 都是 {fmt_number(ranked[0][1])}，"
            f"带间分不出高下（差距小于表里的显示精度）。"
        )
        best_names = set(tops)
    else:
        text = (
            f"各带之中全等级等权平均 TS 最高的是 {_join_bands(tops, labels)}"
            f"（{fmt_number(ranked[0][1])}），最低的是 {_join_bands(bottoms, labels)}"
            f"（{fmt_number(ranked[-1][1])}），相差 {fmt_number(ranked[0][1] - ranked[-1][1])}"
            f"{'——并列。' if len(tops) > 1 else '。'}"
        )
        best_names = set(tops)

    by_lead = {
        name: frame.groupby("lead_h")["TS"].mean() for name, frame in band_frames.items()
    }
    leads = sorted({float(lead) for series in by_lead.values() for lead in series.index})
    flips = 0
    for lead in leads:
        values = {
            name: float(series.get(lead, np.nan))
            for name, series in by_lead.items()
            if lead in series.index
        }
        values = {name: value for name, value in values.items() if np.isfinite(value)}
        if not values:
            continue
        lead_best = max(values.values())
        lead_tops = {name for name, value in values.items() if fmt_number(value) == fmt_number(lead_best)}
        if not lead_tops & best_names:
            flips += 1
    if flips:
        text += (
            f" 名次并不固定：{len(leads)} 个时效里有 {flips} 个时效的头名是别的带——"
            f"带间差异随时效变化，不能只拿全时效平均下结论。"
        )
    else:
        text += f" {len(leads)} 个时效的头名都在上面那几条带里，带间差异在全时效上一致。"
    return text


def _join_bands(names: Sequence[str], labels: Optional[Dict[str, str]]) -> str:
    """并列的带名连起来写（``south [0, 15]、north [15, 30]``）。"""
    from reporting.reports.det_report import region_display

    return "、".join(region_display(name, labels or {}) for name in names)


def _section_regions(
    band_frames: Dict[str, pd.DataFrame],
    labels: Optional[Dict[str, str]],
    artifacts: Optional[Dict[str, Path]],
    figure: Optional[Path],
) -> List[str]:
    """「附 L 分纬度带结果（可选）」——空数据时保留节位并写「本批未出」。"""
    lines = ["## 附 L 分纬度带结果（可选）", ""]
    if not band_frames:
        lines += [f"**本块本批未出。** {_REGION_MISSING_NOTE}", ""]
        return lines

    lines += [
        "本块把宽表里 `region` 非空的行**单独汇总**，全球行**不参与**——全球平均与",
        "纬度带平均是两个量，混在一起算会把带间差异整个抹平，这一块也就没有意义了。",
        "带名与边界照搬评测配置 `options[\"regions\"]`：**报告不翻译、也不写死任何带名**。",
        "表里的口径与 4.1 节完全一致（先合计列联表再算比率，再对等级等权平均），",
        "末列的「综合」就是该带自己的全等级等权平均。",
        "",
    ]
    lines += _table_title("分纬度带 TS（各降水等级，全时效平均）")
    lines.append(_region_table(band_frames, labels))
    lines.append("")
    lines.append(_region_reading(band_frames, labels))
    lines.append("")
    lines.append(
        "> 这一块**只汇总主模型**：按带切和按模型切是两个正交的维度，硬塞进同一张表，"
        "会出现「同一个名字既是带名又是模型名」的歧义。另外，各带的站点数差得远时"
        "TS 的抽样噪声也不一样，差一两个百分点不必当结论。"
    )
    lines.append("")
    if figure is not None and REGION_FIGURE in (artifacts or {}):
        lines += [
            f"![{REGION_FIGURE}]({figure.name})",
            "",
            "图 L1：各纬度带 TS 随时效变化（每个降水等级一个子图，纵轴 0–1）。",
            "",
        ]
    return lines


# --------------------------------------------------------------------------- #
# 主入口
# --------------------------------------------------------------------------- #
def _captions(
    models: Dict[str, pd.DataFrame],
    *,
    window: Optional[float],
    lead: float,
    heavy: Optional[float],
) -> Dict[str, str]:
    summary = summarize(next(iter(models.values())))
    span = f"{summary['leads'][0]:g}–{summary['leads'][-1]:g}h"
    window_text = f"{window:g}h 累积窗口" if window else "本次评估窗口"
    return {
        "overall_heatmap": (
            f"{cn_number(len(models))}模型总体指标热力图。列为指标（综合TS、强降水TS "
            f"越高越好，|BIAS−1|、跨时效IQR、平均名次越低越好），行按综合 TS 从好到差排列；"
            f"格内数字为原始值。颜色是该列在本批模型内的归一化相对表现（1 = 最优、0 = 最差），"
            f"各列量纲不同，颜色只在同一列内可比，不能跨列比较。"
        ),
        "metrics_at_lead": (
            f"基准时效 {lead:g}h（{window_text}）各模型在各降水等级上的 TS、POD、FAR 与 BIAS 对比；"
            f"每格一组柱、按模型着色，BIAS 格内虚线为理想值 1（纵轴自适应、不截断）。"
        ),
        "ts_vs_lead": (
            f"{span} 各模型分降水等级的 TS 曲线，一个等级一个子图、每个模型一条实线；"
            f"纵轴按各子图数据范围缩放，便于分辨模型间差异，"
            f"绝对数值见 §4.1 的逐等级表与时效衰减表。"
        ),
        "pod_vs_lead": (
            f"{span} 各模型分降水等级的 POD 曲线，一个等级一个子图；"
            f"纵轴锁定 0–1，绝对数值见 §4.2 的逐等级表。"
        ),
        "far_vs_lead": (
            f"{span} 各模型分降水等级的 FAR 曲线，一个等级一个子图；"
            f"纵轴锁定 0–1、越低越好，绝对数值见 §4.3 的逐等级表。"
        ),
        "bias_vs_lead": (
            f"{span} 各模型分降水等级的 BIAS 曲线，一个等级一个子图；"
            f"虚线为理想值 1，纵轴自适应、不截断，绝对数值见 §4.4 的逐等级表。"
        ),
        "TS_boxplot": (
            "各模型 TS 的跨时效分布；箱体越高表示该量级整体技巧越好、越窄表示跨时效越稳定，"
            "空心圆为离群时效。纵轴在 0.6 处断开、上段只标 1 的天花板，"
            "下段刻度每 0.1 一格。"
        ),
        "performance_diagram": (
            f"{lead:g}h 时效的性能图：横轴为成功比 1−FAR、纵轴为命中率 POD，"
            f"灰色虚线为 TS 等值线、点线为 BIAS 参考线；每个模型一条线按阈值由小到大连接，"
            f"点越靠近右上角技巧越好。"
        ),
        "aroc_vs_lead": (
            "概率评分的 AROC 随时效变化，2×2 每格一个阈值；各模型一律实线，"
            "纵轴从各模型各阈值的最低值稍下起、刻度每 0.1 一格，"
            "越靠上区分能力越强，0.5 为无技巧（不画辅助线，读数看纵轴）。"
        ),
        "bss_vs_lead": (
            "概率评分的 BSS 随时效变化，2×2 每格一个阈值；各模型一律实线，"
            "纵轴按各档数据范围自适应，越靠上越优于气候基准，"
            "0 为无正技巧（不画辅助线，读数看纵轴）。"
        ),
        "lead_dot": (
            "分时效综合 TS 点图（每行一个时效、每个模型一个点，点越靠右技巧越好）；"
            "竖虚线为 0.1 参考线，点落到线左侧表示该时效综合 TS 已低于 0.1。"
        ),
    }


def build_ts_report(
    data: pd.DataFrame,
    output_path: Path,
    *,
    model_name: str = "model",
    baseline_df: Optional[pd.DataFrame] = None,
    baseline_name: Optional[str] = None,
    baselines: Optional[Dict[str, pd.DataFrame]] = None,
    sources: Optional[Dict[str, str]] = None,
    change_description: Optional[str] = None,
    artifacts: Optional[Dict[str, Path]] = None,
    lead_h: Optional[float] = None,
    box_models: Optional[Dict[str, pd.DataFrame]] = None,
    region_frames: Optional[Dict[str, pd.DataFrame]] = None,
    region_labels: Optional[Dict[str, str]] = None,
    probability: Optional[Dict[str, pd.DataFrame]] = None,
    members: Optional[int] = None,
    coverage: Optional[Dict[str, int]] = None,
    ensemble: bool = False,
) -> Path:
    """写一份 Markdown 报告（六章 + 附录，表不编号、图连续编号）。

    ``box_models`` 是参与对比的模型集合（主模型在前），``create_report`` 传进来
    的那一份同时用于画图，两边必须同一份；不传则按 ``model_name`` + ``baseline_*``
    自拼。``region_frames`` / ``region_labels`` 进「附 L」。``probability`` 只在
    集合报告里占 §4.7 的固定图号；确定性报告整节省略、图号顺延。
    """
    output_path = Path(output_path)
    # 全球行与带行分开算：``data`` 里混着带行时，各节会把同一天算好几遍。
    from reporting.reports.precipitation_plots import split_regions

    data, derived_regions = split_regions(data)
    if region_frames is None:
        region_frames = derived_regions

    models = box_models if box_models is not None else collect_box_models(
        model_name,
        data,
        baseline_df=baseline_df,
        baseline_name=baseline_name,
        baselines=baselines,
    )
    if model_name not in models:
        models = {model_name: data, **models}
    summaries = {name: summarize(frame) for name, frame in models.items()}
    overview = overview_metrics(models)
    ranks = average_ranks(models)
    heavy = _heavy_threshold(models)
    primary_summary = summaries[model_name]
    leads = primary_summary["leads"]
    lead = float(lead_h) if lead_h is not None else leads[0]
    window = None
    if "window_h" in data.columns and data["window_h"].notna().any():
        window = float(data["window_h"].dropna().iloc[0])

    # 概率宽表统一在这里过一遍校验；``None`` 与空 dict 等价。
    loaded_tables: Dict[str, pd.DataFrame] = {}
    for name, frame in (probability or {}).items():
        loaded = load_probability(frame)
        if loaded is not None:
            loaded_tables[str(name)] = loaded
    probability = loaded_tables
    ts_thresholds = _threshold_list(data)

    numbering = _Numbering()
    captions = _captions(models, window=window, lead=lead, heavy=heavy)

    # 图 1 与 §5 点图由本函数出（它们依赖 overview / models，画在这里最省一次重算）。
    if artifacts is not None:
        from reporting.reports.det_plots import plot_overall_heatmap
        from reporting.reports.precipitation_plots import PrecipitationPlotter

        output_path.parent.mkdir(parents=True, exist_ok=True)
        heatmap_path = output_path.parent / "overall_heatmap.svg"
        # 列名进 SVG，必须是英文（图里不放中文）。
        columns = [
            ("Composite TS", {name: overview[name]["ts"] for name in models}, True, 3),
            (
                f"Heavy TS (≥{heavy:g}mm)" if heavy is not None else "Heavy TS",
                {name: overview[name]["heavy"] for name in models},
                True,
                3,
            ),
            ("|BIAS−1|", {name: overview[name]["bias"] for name in models}, False, 3),
            ("IQR across leads", {name: overview[name]["iqr"] for name in models}, False, 3),
            ("Avg rank", {name: ranks.get(name, float("nan")) for name in models}, False, 2),
        ]
        order = sorted(
            models, key=lambda name: _sort_value(overview[name]["ts"]), reverse=True
        )
        figure = plot_overall_heatmap(order, columns)
        figure.savefig(heatmap_path)
        import matplotlib.pyplot as plt

        plt.close(figure)
        artifacts["overall_heatmap"] = heatmap_path

        dot_path = output_path.parent / "lead_dot.svg"
        PrecipitationPlotter().plot_lead_dots(models, save_path=dot_path)
        plt.close("all")
        artifacts["lead_dot"] = dot_path

    # 分带图：要画才画，只有真的落盘了才登记进 artifacts。
    region_figure: Optional[Path] = None
    if region_frames and artifacts is not None:
        region_figure = output_path.parent / "region_ts_vs_lead.svg"
        from reporting.reports.precipitation_plots import PrecipitationPlotter

        PrecipitationPlotter().plot_multi_model_vs_lead(
            dict(region_frames),
            metric="TS",
            mode="absolute",
            legend_title="Latitude band",
            suptitle="TS by latitude band vs lead time",
            save_path=region_figure,
        )
        if artifacts is not None and region_figure.exists():
            artifacts[REGION_FIGURE] = region_figure
        import matplotlib.pyplot as plt

        plt.close("all")

    title = (
        (
            f"集合预报{cn_number(len(models))}模型降水评估报告"
            if len(models) > 1
            else "集合预报降水评估报告"
        )
        if ensemble
        else (
            f"{cn_number(len(models))}模型降水评估报告"
            if len(models) > 1
            else f"{model_name}降水评估报告"
        )
    )
    lines: List[str] = [
        f"# {title}",
        "",
        f"报告日期：{datetime.now():%Y%m%d}    评估层级：站点分类检验",
        "",
    ]
    lines += _section_summary(
        model_name, models, summaries, overview, ranks,
        sources=sources,
        coverage=coverage,
        change_description=change_description,
        ensemble=ensemble,
        members=members,
        has_probability=bool(probability),
    )
    lines += _section_samples(models, summaries, coverage)
    lines += _section_metrics(ensemble=ensemble)
    lines += _section_overall(
        models, overview, ranks, heavy, artifacts, numbering,
        captions["overall_heatmap"], captions["metrics_at_lead"],
    )
    lines += _section_ts(
        models, summaries, model_name, artifacts, numbering, captions["ts_vs_lead"]
    )
    lines += _section_pod(
        models, summaries, artifacts, numbering, captions["pod_vs_lead"]
    )
    lines += _section_far(
        models, summaries, artifacts, numbering, captions["far_vs_lead"]
    )
    lines += _section_bias(
        models, summaries, artifacts, numbering, captions["bias_vs_lead"]
    )
    lines += _section_stability(
        models, model_name, artifacts, numbering, captions["TS_boxplot"]
    )
    lines += _section_performance(
        models, lead, artifacts, numbering, captions["performance_diagram"]
    )
    if ensemble:
        lines += _section_probability(
            probability or None,
            artifacts,
            numbering,
            captions,
            members=members,
            ts_window=window,
            ts_thresholds=ts_thresholds,
        )
    lines += _section_lead_rank(models, artifacts, numbering, captions["lead_dot"])
    lines += _section_conclusions(
        models, summaries, model_name,
        ensemble=ensemble,
        has_probability=bool(probability),
        coverage=coverage,
    )
    lines += ["# 附录：补充诊断", ""]
    lines += [
        "本附录只放**不参与主排序**的分纬度带结果，图号走独立命名空间 `图 L*`，"
        "整块可缺——缺时写「本批未出」及原因，**不留空号、也不指不存在的文件**。",
        "",
    ]
    lines += _section_regions(region_frames, region_labels, artifacts, region_figure)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path
