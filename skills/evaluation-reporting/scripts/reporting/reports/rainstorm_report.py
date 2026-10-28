# -*- coding: utf-8 -*-
"""多个暴雨过程个例产物根 → 多模型对比图 + Markdown 报告。

与 ``ts_report.py`` 不同：它吃一次连续时段评测的宽表，本模块吃一整年的
43 个过程 × 2 种口径，每个过程一段独立产物，所以按「过程」分节。产物形状::

    <根>/{编号}_24h/{起报日}/diagnostics/categorical_wide.csv   口径 A，一天一份
    <根>/{编号}_total/diagnostics/categorical_wide.csv          口径 B，一个过程一份

两种口径互不可比，只在同一口径内做跨模型比较。读 ``categorical_wide.csv``
（带 hits / misses / false_alarms / n_pairs 列联计数），与 ``scores.csv`` 的 TS 同值。
读不到的段记「缺」并计入末尾「口径与注意事项」，不静默跳过。

**不出降水空间分布**：产物只有站点列联表的聚合统计，没有逐站/逐格降水场。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from reporting.reports import rainstorm_catalog
from reporting.reports.rainstorm_catalog import expected_days, get, window_days
from reporting.core.common import finite_or_none, fmt_number

#: 宽表在产物目录里的相对位置（三个 TS 流程都写在这里）。
WIDE_RELATIVE_PATH = Path("diagnostics") / "categorical_wide.csv"

#: 「挑选依据」两张宽表里展示的降水等级（mm）。
SCREENING_THRESHOLDS: Tuple[float, ...] = (10.0, 25.0, 50.0)

#: 排序 / 结论里点名的等级（mm）——强降水看这一档。
RANK_THRESHOLD = 25.0

#: 差值小于这个数一律视为并列（SKILL.md 硬性规则 14）。
TIE_THRESHOLD = 0.005

#: 列联宽表必须有的列。
REQUIRED_COLUMNS = ("grade", "threshold_mm", "TS")

#: 图在报告目录下的子目录。
FIGURES_DIR = "figures"


# ============================================================== 数据结构


@dataclass
class RainstormArchive:
    """一个模型的产物根，外加报告里显示的名字。

    ``name`` **由调用方给**，不从目录名、也不从 ``manifest.run_id`` 猜——
    本仓库的规矩是身份不靠目录名，而这里连 manifest 都不读（一个产物根下有
    255 份 manifest，认哪一份都不对）。
    """

    name: str
    root: Path

    def path_of(self, code: str, *, day: Optional[str] = None) -> Path:
        """该模型某个过程（可选某个起报日）的宽表路径。"""
        if day is None:
            return Path(self.root) / f"{code}_total" / WIDE_RELATIVE_PATH
        return Path(self.root) / f"{code}_24h" / str(day) / WIDE_RELATIVE_PATH


@dataclass
class ProcessScores:
    """一个模型在一个过程上的两口径评分。"""

    code: str
    #: 口径 A：起报日 → 宽表；缺的起报日不在字典里。
    days: Dict[str, pd.DataFrame] = field(default_factory=dict)
    #: 口径 B：过程累积宽表；产物缺失时 None。
    total: Optional[pd.DataFrame] = None

    @property
    def missing_days(self) -> List[str]:
        """应有但没有的口径 A 起报日。"""
        return [day for day in expected_days(self.code) if day not in self.days]

    @property
    def present(self) -> bool:
        return bool(self.days) or self.total is not None


# ============================================================== 加载


def load_archive(name: str, root: Path) -> RainstormArchive:
    """校验产物根可用，返回 :class:`RainstormArchive`。

    Raises:
        ValueError: 目录不存在，或一个过程段都没有——**看着不像产物根**。
    """
    root = Path(root)
    if not root.is_dir():
        raise ValueError(f"{root} 不存在或不是目录：给的是 rainstorm_ts_single_* 的输出根")
    if not any(root.glob("*_24h")) and not any(root.glob("*_total")):
        raise ValueError(
            f"{root} 下既没有 `{{编号}}_24h/` 也没有 `{{编号}}_total/`："
            f"这不是暴雨过程个例的产物根（该给 --config 的 OUTPUT_ROOT）"
        )
    return RainstormArchive(str(name), root)


def _read_wide(path: Path) -> Optional[pd.DataFrame]:
    """读一份列联宽表；文件不在或空表返回 None。

    Raises:
        ValueError: 表在，但没有必需的列——**别拿一张不认识的表往下算**。
    """
    path = Path(path)
    if not path.is_file():
        return None
    frame = pd.read_csv(path)
    if frame.empty:
        return None
    frame.columns = [str(column).strip() for column in frame.columns]
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(
            f"{path} 缺列 {missing}，不是 TS 列联宽表；现有列 {list(frame.columns)}"
        )
    frame["threshold_mm"] = pd.to_numeric(frame["threshold_mm"], errors="coerce")
    for column in ("TS", "POD", "FAR", "BIAS", "n_pairs", "漏报率", "hits",
                   "misses", "false_alarms"):
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def load_process(archive: RainstormArchive, code: str) -> ProcessScores:
    """读一个模型在一个过程上的两口径产物（缺段留空，不报错）。"""
    get(code)  # 编号不在编目里时在这里就报错，别等下标越界
    scores = ProcessScores(code=code)
    for day in expected_days(code):
        frame = _read_wide(archive.path_of(code, day=day))
        if frame is not None:
            scores.days[day] = frame
    # 产物里有、但不在预期起报日里的段（比如评测口径改过）也读进来，
    # 否则那些天会既不出现在表里、也不算「缺」，凭空消失。
    day_root = Path(archive.root) / f"{code}_24h"
    if day_root.is_dir():
        for child in sorted(path for path in day_root.iterdir() if path.is_dir()):
            if child.name in scores.days:
                continue
            frame = _read_wide(child / WIDE_RELATIVE_PATH)
            if frame is not None:
                scores.days[child.name] = frame
    scores.total = _read_wide(archive.path_of(code))
    return scores


def load_all(
    archives: Sequence[RainstormArchive], codes: Sequence[str]
) -> Dict[str, Dict[str, ProcessScores]]:
    """``{模型名: {编号: ProcessScores}}``。"""
    return {
        archive.name: {code: load_process(archive, code) for code in codes}
        for archive in archives
    }


# ============================================================== 取数


def thresholds_of(scores: ProcessScores) -> List[float]:
    """这个模型这个过程上出现过的阈值（升序）。"""
    frames = list(scores.days.values()) + ([scores.total] if scores.total is not None else [])
    values: set = set()
    for frame in frames:
        if frame is not None and "threshold_mm" in frame.columns:
            values.update(float(item) for item in frame["threshold_mm"].dropna().unique())
    return sorted(values)


def process_days(
    code: str,
    per_model: Dict[str, Dict[str, ProcessScores]],
    names: Sequence[str],
) -> List[str]:
    """这个过程要出现在表里的全部起报日：**实际有产物的 ∪ 应有产物的**。

    少了后半截，缺产物的起报日会从表里消失，读者看不出「缺」。
    """
    days = {
        day
        for name in names
        for day in (
            per_model[name][code].days if per_model[name].get(code) is not None else {}
        )
    }
    days.update(expected_days(code))
    return sorted(days)


def cell(frame: Optional[pd.DataFrame], threshold: float, column: str = "TS") -> Optional[float]:
    """取某一档的某个列值；表不在 / 没这档 / 值为空都返回 None。"""
    if frame is None or column not in frame.columns:
        return None
    part = frame[np.isclose(frame["threshold_mm"], float(threshold))]
    if part.empty:
        return None
    value = pd.to_numeric(part[column], errors="coerce").iloc[0]
    return None if pd.isna(value) else float(value)


def daily_values(scores: ProcessScores, threshold: float, column: str = "TS"):
    """口径 A：``{起报日: 值}``，含没有有效值的起报日（值为 None）。"""
    return {day: cell(frame, threshold, column) for day, frame in scores.days.items()}


_COUNT_COLUMNS = ("hits", "misses", "false_alarms", "n_pairs")


def daily_counts(
    scores: ProcessScores, threshold: float
) -> Optional[Dict[str, float]]:
    """口径 A 汇总：把该过程**全部起报日**的四格表计数逐格加总。

    这是口径 A 每过程数值的唯一来源。TS 是比率，逐日取算术平均会让事件很少的
    起报日（比如只报出 12 个站、TS 恰好为 0）和事件上千的起报日**等权**，
    均值被小样本日拖偏；汇总量再按定义算一次就没有这个问题
    （SKILL.md 硬性规则 4）。返回 None 表示一个起报日都没有可用的计数。
    """
    total = {key: 0.0 for key in _COUNT_COLUMNS}
    seen = False
    for frame in scores.days.values():
        hits = cell(frame, threshold, "hits")
        misses = cell(frame, threshold, "misses")
        false_alarms = cell(frame, threshold, "false_alarms")
        if hits is None or misses is None or false_alarms is None:
            continue
        seen = True
        total["hits"] += hits
        total["misses"] += misses
        total["false_alarms"] += false_alarms
        total["n_pairs"] += cell(frame, threshold, "n_pairs") or 0.0
    return total if seen else None


def _from_counts(counts: Dict[str, float], column: str) -> Optional[float]:
    """四格表计数 → 指标值；分母为 0 返回 None（无事件，不当 0）。"""
    hits, misses = counts["hits"], counts["misses"]
    false_alarms = counts["false_alarms"]
    if column in _COUNT_COLUMNS:
        return counts[column]
    if column == "TS":
        denominator = hits + misses + false_alarms
        return hits / denominator if denominator else None
    if column == "POD":
        return hits / (hits + misses) if hits + misses else None
    if column == "FAR":
        return false_alarms / (hits + false_alarms) if hits + false_alarms else None
    if column == "BIAS":
        return (hits + false_alarms) / (hits + misses) if hits + misses else None
    return None


def daily_pooled(
    scores: ProcessScores, threshold: float, column: str = "TS"
) -> Optional[float]:
    """口径 A 汇总值：全部起报日的计数加总后按 ``column`` 重算一次。"""
    counts = daily_counts(scores, threshold)
    return None if counts is None else _from_counts(counts, column)


def total_value(scores: ProcessScores, threshold: float, column: str = "TS") -> Optional[float]:
    """口径 B：过程累积值。"""
    return cell(scores.total, threshold, column)


def n_pairs_of(scores: ProcessScores, threshold: float, *, total: bool = False):
    """有效配对数：口径 B 取累积表，口径 A 取各起报日的最小 / 最大值。"""
    if total:
        return cell(scores.total, threshold, "n_pairs")
    values = [
        value
        for value in (cell(frame, threshold, "n_pairs") for frame in scores.days.values())
        if value is not None
    ]
    if not values:
        return None
    low, high = min(values), max(values)
    return low if low == high else (low, high)


# ============================================================== 文本工具


def _fmt_delta(value: Optional[float]) -> str:
    number = finite_or_none(value)
    return "—" if number is None else f"{number:+.3f}"


def _fmt_pct(value: Optional[float], digits: int = 1) -> str:
    """比例 → 百分数文本（``0.085`` → ``8.5%``）；None 写成 ``—``。"""
    number = finite_or_none(value)
    return "—" if number is None else f"{number * 100:.{digits}f}%"


def _fmt_count(value) -> str:
    """样本量：单值或 ``(最小, 最大)``。"""
    if value is None:
        return "—"
    if isinstance(value, tuple):
        return f"{int(value[0])}–{int(value[1])}"
    return f"{int(value)}"


def _grade_label(threshold: float) -> str:
    """``10.0`` → ``≥10``；小数量级保留一位（``0.1`` → ``≥0.1``）。"""
    return f"≥{threshold:g}"


def _delta_cell(values: Sequence[Optional[float]]) -> str:
    """Δ 列：「主模型 − 第一个对比模型」；任一为空就写 ``—``，不补 0。"""
    if len(values) < 2 or any(value is None for value in values[:2]):
        return "—"
    return _fmt_delta(values[0] - values[1])


class _Numbering:
    """图与表按在报告里出现的顺序编号（同 ``ts_report._Numbering`` 的约定）。"""

    def __init__(self) -> None:
        self._figures = 0
        self._tables = 0

    def figure(self) -> int:
        self._figures += 1
        return self._figures

    def table(self) -> int:
        self._tables += 1
        return self._tables


def _figure(lines: List[str], numbering: _Numbering, rel_path: Path, caption: str) -> None:
    """图注在**下方**（中文论文惯例），路径用正斜杠，Windows 上也能渲染。"""
    lines += [
        f"![{rel_path.stem}]({Path(rel_path).as_posix()})",
        "",
        f"图 {numbering.figure()}：{caption}",
        "",
    ]


def _table_caption(lines: List[str], numbering: _Numbering, text: str) -> None:
    """表题在表**上方**。"""
    lines += [f"表 {numbering.table()}：{text}", ""]


def _markdown_table(lines: List[str], header: Sequence[str], rows: Sequence[Sequence[str]]) -> None:
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "---|" * len(header))
    for row in rows:
        lines.append("| " + " | ".join(str(item) for item in row) + " |")
    lines.append("")


# ============================================================== 挑选依据表


def _screening_header(names: Sequence[str]) -> List[str]:
    header = ["编号", "起止", "等级"]
    for threshold in SCREENING_THRESHOLDS:
        header += [f"{_grade_label(threshold)} {name}" for name in names]
        header.append(f"Δ({names[0]}−{names[1]})")
    return header


def _screening_rows(
    codes: Sequence[str],
    per_model: Dict[str, Dict[str, ProcessScores]],
    names: Sequence[str],
    *,
    total: bool,
) -> List[List[str]]:
    """挑选依据宽表的表体：一行一个过程（口径 A 取起报日汇总，口径 B 取累积）。"""
    rows: List[List[str]] = []
    for code in codes:
        process = get(code)
        cells = [code, process.span, process.grade]
        for threshold in SCREENING_THRESHOLDS:
            values: List[Optional[float]] = []
            for name in names:
                scores = per_model[name].get(code)
                if scores is None or not scores.present:
                    cells.append("缺")
                    values.append(None)
                    continue
                value = (
                    total_value(scores, threshold) if total else daily_pooled(scores, threshold)
                )
                cells.append(fmt_number(value))
                values.append(value)
            cells.append(_delta_cell(values))
        rows.append(cells)
    return rows


# ============================================================== 统计口径


def _leader(values: Dict[str, Optional[float]]) -> Tuple[Optional[str], Optional[str], float]:
    """谁领先。返回 ``(领先者, 落后者, 差值)``。

    差值小于 :data:`TIE_THRESHOLD` 一律算并列，领先者返回 None
    （SKILL.md 硬性规则 14）；有效值少于两个时同样返回 None。
    """
    usable = {name: value for name, value in values.items() if value is not None}
    if len(usable) < 2:
        return None, None, 0.0
    ordered = sorted(usable.items(), key=lambda item: item[1], reverse=True)
    (best_name, best), (worst_name, worst) = ordered[0], ordered[-1]
    gap = float(best) - float(worst)
    if gap < TIE_THRESHOLD:
        return None, None, gap
    return best_name, worst_name, gap


def _mean_of(values: Sequence[Optional[float]]) -> Optional[float]:
    """忽略 None 的均值；一个有效值都没有时返回 None（不当 0）。"""
    usable = [float(value) for value in values if value is not None]
    return float(np.mean(usable)) if usable else None


def _base_rate(
    scores: ProcessScores,
    threshold: float,
    *,
    total: bool,
) -> Optional[float]:
    """实况基准率 = (hits + misses) / n_pairs，即实况达阈值的站点占比。

    两个口径都**先在各自口径内把计数加总再算比例**——口径 A 汇总全部起报日、
    口径 B 就是那一根累积窗——而不是跨天取比率的平均，理由同 :func:`daily_counts`。
    """
    if total:
        frame = scores.total
        hits = cell(frame, threshold, "hits")
        misses = cell(frame, threshold, "misses")
        pairs = cell(frame, threshold, "n_pairs")
        if hits is None or misses is None or not pairs:
            return None
        return (hits + misses) / pairs
    counts = daily_counts(scores, threshold)
    if counts is None or not counts["n_pairs"]:
        return None
    return (counts["hits"] + counts["misses"]) / counts["n_pairs"]


def _base_rate_mean(
    codes: Sequence[str],
    per_model: Dict[str, Dict[str, ProcessScores]],
    name: str,
    *,
    total: bool,
    threshold: float = RANK_THRESHOLD,
) -> Optional[float]:
    """一个模型在全部有产物的过程上的基准率均值。

    基准率是**实况侧**的性质（同一批站、同一批窗），两个模型算出来应当几乎一样，
    所以报告里只报主模型这一个数、并不拿它做模型间比较。
    """
    values = []
    for code in codes:
        scores = per_model[name].get(code)
        if scores is None or not scores.present:
            continue
        rate = _base_rate(scores, threshold, total=total)
        if rate is not None:
            values.append(rate)
    return float(np.mean(values)) if values else None


def _shared_codes(
    codes: Sequence[str],
    per_model: Dict[str, Dict[str, ProcessScores]],
    names: Sequence[str],
) -> List[str]:
    """双方都有产物的过程，保持 codes 的原有顺序。

    跨模型求差**必须**在这批过程上算：只要有一边缺段，各算各的均值就不是同分母，
    相减出来的 Δ 里混着覆盖率差，不是模型差异。本报告的 Δ 全部只用这一批。
    """
    return [
        code
        for code in codes
        if all(
            per_model[name].get(code) is not None and per_model[name][code].present
            for name in names
        )
    ]


def _aggregate(
    codes: Sequence[str],
    per_model: Dict[str, Dict[str, ProcessScores]],
    names: Sequence[str],
    *,
    total: bool,
    column: str = "TS",
    threshold: float = RANK_THRESHOLD,
    shared_only: bool = False,
) -> Dict[str, Optional[float]]:
    """各模型在全部过程上的等权平均（缺段的过程不进分母）。

    ``shared_only=True`` 时**逐行配对**：只保留各模型都有有效值的过程，各模型
    因此一定同分母。这是求 Δ 或横向比大小的唯一正确算法——只按「有没有产物」筛
    （:func:`_shared_codes`）还不够：一个模型的整段 ``_total`` 可能缺、而 ``_24h``
    在，那一行照样是两个分母。只描述单个模型自身水平的地方（比如「≥50mm 档最高
    只有多少」）不必走这条。
    """
    if shared_only:
        return {
            name: _mean_of(values)
            for name, values in _paired_values(
                codes, per_model, names, total=total,
                column=column, threshold=threshold,
            ).items()
        }
    return {
        name: _mean_of([
            (total_value(per_model[name][code], threshold, column) if total
             else daily_pooled(per_model[name][code], threshold, column))
            for code in codes
            if per_model[name].get(code) is not None
        ])
        for name in names
    }


def _paired_values(
    codes: Sequence[str],
    per_model: Dict[str, Dict[str, ProcessScores]],
    names: Sequence[str],
    *,
    total: bool,
    column: str = "TS",
    threshold: float = RANK_THRESHOLD,
) -> Dict[str, List[float]]:
    """逐过程取各模型的值，**只留各模型都有有效值的过程**，保证各模型等长。

    「有产物」和「有有效值」是两回事：一段过程可能整个 ``_total`` 缺、
    可能该档无事件（``—``）。两种都会让某个模型比别家少一个数，
    而少一个数就是不同的分母。
    """
    pools: Dict[str, List[float]] = {name: [] for name in names}
    for code in codes:
        row: Dict[str, Optional[float]] = {}
        for name in names:
            scores = per_model[name].get(code)
            if scores is None or not scores.present:
                row[name] = None
                continue
            row[name] = (
                total_value(scores, threshold, column)
                if total
                else daily_pooled(scores, threshold, column)
            )
        if any(value is None for value in row.values()):
            continue
        for name in names:
            pools[name].append(row[name])
    return pools


def _win_counts(
    codes: Sequence[str],
    per_model: Dict[str, Dict[str, ProcessScores]],
    names: Sequence[str],
    *,
    total: bool,
) -> Dict[str, int]:
    """逐过程胜负计数（只看主模型 vs 第一个对比模型；并列不算赢也不算输）。"""
    counts = {names[0]: 0, names[1]: 0, "并列": 0, "无法比较": 0}
    for code in codes:
        values: Dict[str, Optional[float]] = {}
        for name in names[:2]:
            scores = per_model[name].get(code)
            if scores is None or not scores.present:
                values[name] = None
                continue
            values[name] = (
                total_value(scores, RANK_THRESHOLD)
                if total
                else daily_pooled(scores, RANK_THRESHOLD)
            )
        if any(value is None for value in values.values()):
            counts["无法比较"] += 1
            continue
        winner, _, _ = _leader(values)
        counts["并列" if winner is None else winner] += 1
    return counts


def _counts_phrase(counts: Dict[str, int], names: Sequence[str]) -> str:
    text = (
        f"{names[0]} 领先 {counts[names[0]]} 个、落后 {counts[names[1]]} 个、"
        f"并列 {counts['并列']} 个"
    )
    if counts["无法比较"]:
        text += f"、缺段无法比较 {counts['无法比较']} 个"
    return text


def _leader_phrase(
    winner: Optional[str], gap: float, *, scope: str
) -> str:
    if winner is None:
        return f"{scope}两模型差值 {fmt_number(gap)}，小于并列阈值 {TIE_THRESHOLD}，本口径下不分胜负"
    return f"{scope}{winner} 领先 {gap:.3f}"


def _coverage_note(
    codes: Sequence[str],
    per_model: Dict[str, Dict[str, ProcessScores]],
    names: Sequence[str],
) -> str:
    """覆盖率一句话；全齐时给「无缺段」。"""
    fragments = []
    for name in names:
        missing_processes = [
            code for code in codes
            if per_model[name].get(code) is None or not per_model[name][code].present
        ]
        missing_days = sum(
            len(per_model[name][code].missing_days)
            for code in codes
            if per_model[name].get(code) is not None
        )
        if not missing_processes and not missing_days:
            fragments.append(f"{name} 无缺段")
            continue
        detail = []
        if missing_processes:
            detail.append(f"{len(missing_processes)} 个过程整段缺")
        if missing_days:
            detail.append(f"{missing_days} 个起报日缺")
        fragments.append(f"{name} 缺 " + "、".join(detail))
    return "；".join(fragments)


# ============================================================== 正文文字


def _caliber_block(
    codes: Sequence[str],
    per_model: Dict[str, Dict[str, ProcessScores]],
    names: Sequence[str],
) -> List[str]:
    """第一节的「两种口径怎么算」。

    计算方式一段是**配置层面的事实**（跟模型、跟产物都无关），所以写死；
    只有末尾两个基准率是跟着产物算的——换一批产物这两个数就变。

    基准率是实况侧的性质，两个模型算出来几乎一样，所以只报主模型这一个数，
    措辞上也点明是「实况侧」，免得被读成模型差异。
    """
    grade = _grade_label(RANK_THRESHOLD)
    rate_day = _base_rate_mean(codes, per_model, names[0], total=False)
    rate_total = _base_rate_mean(codes, per_model, names[0], total=True)
    return [
        "**两种口径怎么算**",
        "",
        "**口径 A（逐日 24h）——把过程拆成逐日 24 小时的窗，每天单独评一次。**"
        "过程期 `[S-1, E]` 逐日 00UTC 各起报一根，一根预报只评 lead 24h 那一个窗；"
        "窗是 **08–08 北京时**（起报日 08 时 → 次日 08 时），相邻起报日的窗首尾相接、"
        "**互不重叠**，过程有 N 个自然日就有 N+1 个起报日（`S-1` 到 `E` 各一个）、"
        "N+1 个独立的 24h 窗。每个窗内逐站二值化"
        "（实况达阈值记「有事件」、否则「无事件」，预报同理），数出 "
        "hits / misses / false_alarms，出一张四格表、一个 TS。"
        "报告里一个过程的**口径 A 数值**是把该过程**所有起报日的四格表逐格加总**"
        "（Σhits、Σmisses、Σfalse_alarms）后按定义**重新算一次** TS，"
        "**不是**把这些逐日 TS 加起来除以天数：TS 是比率，简单平均会让只报出"
        "十几个站、TS 恰好为 0 的起报日与事件上千的起报日**等权**，把均值拖偏；"
        "汇总重算等价于把该过程的全部「站点·日」样本当成一个整体。"
        "底层 tp 是 6h 步，24h 窗由 `[6, 12, 18, 24]` 四个 6h 步累加而成。",
        "",
        "**口径 B（过程累积）——整段加成一个总量，只评一次。**"
        f"只在 `S-1` 日 00UTC 起报**一根**，窗长 = 过程全长 `(E-S+2)×24h`，"
        "窗为 `[S-1 08:00, E+1 08:00]`；预报侧把窗内**所有 6h 步的 tp 累加**成总量，"
        "实况侧在同一窗内同样累加；然后拿**总量**比阈值，逐站判一次，"
        "出一张四格表、一个 TS。",
        "",
        "**「一天大雨一天小雨」在两种口径下不是一回事。**"
        f"口径 A 每天单独判——大雨那天算 {_grade_label(50.0)}mm 事件、"
        "小雨那天只是「有降水」；口径 B 加起来判——50 + 5 = 55mm，"
        f"按 {_grade_label(50.0)}mm 记**一次**事件。因此「8 天每天 8mm 的连绵小雨」"
        "与「1 天 60mm 的一场暴雨」在口径 B 眼里**完全相同**。"
        "口径 B 只回答「总量报得准不准」，回答不了「每天都报得准不准」。",
        "",
        f"两种口径下**实况侧**的基准率（实况 {grade}mm 的站点占比）因此差得很远："
        f"口径 A 逐过程汇总 **{_fmt_pct(rate_day)}**，"
        f"口径 B 逐过程 **{_fmt_pct(rate_total)}**。"
        "注意两者的统计单元不同（前者是「站点·日」的汇总比例，"
        "后者是「站点」的比例再对过程取平均），但是量级差这么多已经足够说明问题："
        "**基准率不在一个量级，TS 就不能跨口径比**——这是报告里两套数分节呈现的原因。",
        "",
    ]


def _summary_paragraph(
    codes: Sequence[str],
    per_model: Dict[str, Dict[str, ProcessScores]],
    names: Sequence[str],
) -> str:
    """结论摘要那一整段的规则化诊断。"""
    sentences: List[str] = []

    for scope, total in (("口径 B（过程累积）的 ", True), ("口径 A（逐日 24h 汇总）的 ", False)):
        pools = _paired_values(codes, per_model, names, total=total)
        values = {name: _mean_of(pools[name]) for name in names}
        paired = len(pools[names[0]])
        pool_note = (
            f"全部 {len(codes)} 个过程"
            if paired == len(codes)
            else f"各模型都有有效值的 {paired} 个过程（共 {len(codes)} 个）"
        )
        winner, _, gap = _leader(values)
        detail = "、".join(f"{name} {fmt_number(values[name])}" for name in names)
        sentences.append(
            f"{scope}{_grade_label(RANK_THRESHOLD)}mm 均值水平（{pool_note}）：{detail}，"
            + _leader_phrase(winner, gap, scope="")
        )

    counts = _win_counts(codes, per_model, names, total=True)
    sentences.append(
        f"逐过程看（口径 B 的 {_grade_label(RANK_THRESHOLD)}mm），"
        + _counts_phrase(counts, names)
    )

    hard = _aggregate(codes, per_model, names, total=True, threshold=50.0,
                      shared_only=True)
    sentences.append(
        f"极强降水更难：≥50mm 档的过程累积 TS 为 "
        + "、".join(f"{name} {fmt_number(hard[name])}" for name in names)
        + f"，相对 {_grade_label(RANK_THRESHOLD)}mm 档整体回落"
    )

    sentences.append(f"数据覆盖：{_coverage_note(codes, per_model, names)}")
    return "；".join(sentences) + "。"


def _business_conclusion(
    codes: Sequence[str],
    per_model: Dict[str, Dict[str, ProcessScores]],
    names: Sequence[str],
) -> str:
    """业务定性一句话：看强降水档能用到什么程度。

    只看全过程的**均值水平**（各模型各算各的，这里不做模型间相减，故不用共同子集），
    不拿单个过程说事——过程间差异太大，单个过程的结果没有代表性。
    """
    values = _aggregate(codes, per_model, names, total=True)
    best = max((value for value in values.values() if value is not None), default=None)
    if best is None:
        return "本批产物没有可用的过程累积评分，无法给出业务定性。"
    if best >= 0.5:
        level = "对过程总量级降水（≥25mm）具备可用技巧，可作为过程是否发生的定性参考"
    elif best >= 0.3:
        level = "对过程总量级降水（≥25mm）具备有限技巧，只宜作趋势性参考，不宜直接用于落区判定"
    else:
        level = "对过程总量级降水（≥25mm）技巧偏低，本批数据只能用于量级与相对强弱比较"
    return (
        f"就本批 {len(codes)} 个过程而言，{level}；单过程、单起报的结果受样本量与"
        f"过程本身可预报性影响很大，不作个例定论。"
    )


def _process_overview(
    code: str,
    per_model: Dict[str, Dict[str, ProcessScores]],
    names: Sequence[str],
) -> str:
    """过程概况一句话：起报日数、窗长、各模型样本量。"""
    process = get(code)
    days = process_days(code, per_model, names)
    window = window_days(code) * 24
    pairs: List[str] = []
    for name in names:
        scores = per_model[name].get(code)
        if scores is None or not scores.present:
            pairs.append(f"{name} 缺段")
            continue
        pairs.append(
            f"{name} 口径 A {_fmt_count(n_pairs_of(scores, RANK_THRESHOLD))} 对、"
            f"口径 B {_fmt_count(n_pairs_of(scores, RANK_THRESHOLD, total=True))} 对"
        )
    if days:
        coverage = f"口径 A 覆盖 {len(days)} 个起报日（{days[0]}–{days[-1]}，day-1 的 08–08 窗）"
    else:
        coverage = "口径 A 一个起报日的产物都没有"
    return (
        f"{code}（{process.span}，{process.grade}）：{coverage}；"
        f"口径 B 为 {process.start} 前一天起报、窗长 {window}h。"
        f"{_grade_label(RANK_THRESHOLD)}mm 档有效配对数：" + "；".join(pairs) + "。"
    )


def _daily_reading(
    code: str,
    per_model: Dict[str, Dict[str, ProcessScores]],
    names: Sequence[str],
    days: Sequence[str],
) -> str:
    """逐日一节的小结：汇总水平、日间起伏、领先者。"""
    lines: List[str] = []
    for name in names:
        scores = per_model[name].get(code)
        values = daily_values(scores, RANK_THRESHOLD) if scores is not None else {}
        usable = {day: value for day, value in values.items() if value is not None}
        if not usable:
            lines.append(f"{name} 在 {_grade_label(RANK_THRESHOLD)}mm 档没有有效 TS")
            continue
        ordered = sorted(usable.items(), key=lambda item: item[1], reverse=True)
        lines.append(
            f"{name} 汇总 {fmt_number(daily_pooled(scores, RANK_THRESHOLD))}，"
            f"逐日起伏 {ordered[-1][1]:.3f}（{ordered[-1][0]}）–"
            f"{ordered[0][1]:.3f}（{ordered[0][0]}），有效 {len(usable)}/{len(days)} 天"
        )
    values = {
        name: daily_pooled(per_model[name][code], RANK_THRESHOLD)
        if per_model[name].get(code) is not None else None
        for name in names
    }
    winner, _, gap = _leader(values)
    return "；".join(lines) + "；" + _leader_phrase(winner, gap, scope="汇总口径下") + "。"


def _highest_usable(scores: Optional[ProcessScores]) -> Optional[float]:
    """这个过程上还有命中（``hits > 0``）的最高等级；都没有则 None。

    TS 为 NaN 的档分两种：**无事件**和**无有效配对**。只有"无事件"说明量级到了
    天花板，所以判据取 ``hits > 0`` 而不是 TS 非空。
    """
    if scores is None or scores.total is None or "hits" not in scores.total.columns:
        return None
    hit = scores.total[pd.to_numeric(scores.total["hits"], errors="coerce").fillna(0) > 0]
    if hit.empty:
        return None
    return float(hit["threshold_mm"].max())


def _total_reading(
    code: str,
    per_model: Dict[str, Dict[str, ProcessScores]],
    names: Sequence[str],
) -> str:
    """口径 B 一节的小结：累积水平、误差形态、可用量级上限、领先者。"""
    fragments: List[str] = []
    shapes: List[str] = []
    for name in names:
        scores = per_model[name].get(code)
        if scores is None or scores.total is None:
            fragments.append(f"{name} 无口径 B 产物")
            continue
        cap = _highest_usable(scores)
        fragments.append(
            f"{name} {_grade_label(RANK_THRESHOLD)}mm TS "
            f"{fmt_number(total_value(scores, RANK_THRESHOLD))}"
            f"（POD {fmt_number(total_value(scores, RANK_THRESHOLD, 'POD'))}、"
            f"FAR {fmt_number(total_value(scores, RANK_THRESHOLD, 'FAR'))}、"
            f"BIAS {fmt_number(total_value(scores, RANK_THRESHOLD, 'BIAS'))}），"
            f"有命中的最高等级 {_grade_label(cap) if cap is not None else '无'}"
        )
        pod = total_value(scores, RANK_THRESHOLD, "POD")
        far = total_value(scores, RANK_THRESHOLD, "FAR")
        if pod is None or far is None:
            shapes.append(f"{name} 的 POD / FAR 列不全，误差形态判不了")
        elif far > 1.0 - pod:
            shapes.append(f"{name} 空报多于漏报（FAR {fmt_number(far)} > 漏报率 {fmt_number(1 - pod)}）")
        else:
            shapes.append(f"{name} 漏报多于空报（漏报率 {fmt_number(1 - pod)} > FAR {fmt_number(far)}）")

    values = {
        name: total_value(per_model[name][code], RANK_THRESHOLD)
        if per_model[name].get(code) is not None else None
        for name in names
    }
    winner, _, gap = _leader(values)
    return (
        "；".join(fragments) + "；" + "；".join(shapes) + "；"
        + _leader_phrase(winner, gap, scope="过程累积口径下") + "。"
    )


def _process_reading(
    code: str,
    per_model: Dict[str, Dict[str, ProcessScores]],
    names: Sequence[str],
) -> str:
    """过程小结：两个口径给的是不是同一个结论。"""
    rows: List[str] = []
    for name in names:
        scores = per_model[name].get(code)
        if scores is None or not scores.present:
            rows.append(f"{name} 缺段")
            continue
        rows.append(
            f"{name} 口径 A 汇总 {fmt_number(daily_pooled(scores, RANK_THRESHOLD))}"
            f" / 口径 B 累积 {fmt_number(total_value(scores, RANK_THRESHOLD))}"
        )
    day_values = {
        name: daily_pooled(per_model[name][code], RANK_THRESHOLD)
        if per_model[name].get(code) is not None else None
        for name in names
    }
    total_values = {
        name: total_value(per_model[name][code], RANK_THRESHOLD)
        if per_model[name].get(code) is not None else None
        for name in names
    }
    day_winner, _, _ = _leader(day_values)
    total_winner, _, _ = _leader(total_values)
    if day_winner is None or total_winner is None:
        verdict = "两个口径至少有一个分不出胜负，本过程的名次是否翻转判不了"
    elif day_winner != total_winner:
        verdict = (
            f"**两个口径名次相反**：逐日 24h 看 {day_winner} 好，过程累积看 "
            f"{total_winner} 好——该过程「总量报得准」与「每天都报得准」不是一回事，"
            f"引用本过程时必须带上口径"
        )
    else:
        verdict = f"两个口径一致，都是 {total_winner} 更好"
    return "；".join(rows) + "。" + verdict + "。"


# ============================================================== 章节


def _header_lines(
    archives: Sequence[RainstormArchive],
    processes: Sequence[str],
    title: str,
    archive: Optional[str],
    change: Optional[str],
) -> List[str]:
    names = [item.name for item in archives]
    lines = [f"# {title}", ""]
    if archive:
        lines += [f"**归档：{archive}**", ""]
    lines += [
        f"**主模型：{names[0]}**；对比模型：" + "、".join(names[1:]),
        "",
        f"**过程编目**：《2025年暴雨过程纪要表（修订）-1.docx》，编号 202501–202543，"
        f"共 {len(rainstorm_catalog.CATALOG)} 个过程，"
        f"本报告详析其中 {len(processes)} 个（{'、'.join(processes)}）。",
        "",
        "**归档文件**",
        "",
    ]
    lines += [f"- {item.name}：`{item.root}`" for item in archives]
    lines.append("")
    lines += [
        "> 两种口径**互不可比**，只在本口径内跨模型、跨过程比：口径 A（逐日 24h）"
        "把过程期拆成逐日起报的 day-1 窗，一天一个结果，基准率随每日实况浮动；"
        "口径 B（过程累积）用过程起点前一天的一根预报把整段降水累加后评总量，"
        "基准率显著更高，同一个过程的 TS 因此明显高于口径 A。",
        "",
    ]
    return lines


def _conclusion_section(
    codes: Sequence[str],
    per_model: Dict[str, Dict[str, ProcessScores]],
    names: Sequence[str],
    processes: Sequence[str],
    change: Optional[str],
) -> List[str]:
    lines = ["## 一、结论摘要", ""]
    lines += [
        "本报告评估 " + "、".join(names) + f" 共 {len(names)} 个模型在 2025 年 "
        f"{len(codes)} 个暴雨过程上的站点降水分类检验表现。评分对象是**站点 24h 与"
        f"过程累积降水量**的分类命中情况（zd_sta_10285 站表的全部站；评测不分纬度带，"
        f"只出全球行），指标是 TS 及其配套的 POD / FAR / 频率偏差；**不重算任何指标**，"
        f"每个数字都直接来自产物里的 `diagnostics/categorical_wide.csv`。"
        f"本报告详析 {'、'.join(processes)} 共 {len(processes)} 个过程，"
        f"其余过程的挑选依据见第二节。",
        "",
    ]
    if change:
        lines += [f"**本次改动**：{change}", ""]
    lines += _caliber_block(codes, per_model, names)
    lines += [_summary_paragraph(codes, per_model, names), ""]
    lines += [f"**业务定性**：{_business_conclusion(codes, per_model, names)}", ""]
    return lines


def _screening_section(
    codes: Sequence[str],
    per_model: Dict[str, Dict[str, ProcessScores]],
    names: Sequence[str],
    numbering: _Numbering,
    overview_daily_path: Path,
    overview_path: Path,
) -> List[str]:
    lines = ["## 二、过程清单与挑选依据", ""]
    lines += [
        f"下表覆盖全部 {len(codes)} 个过程，供挑选详析对象用。"
        f"口径 A 的列是**该过程全部起报日汇总**得到的 TS——把各起报日的 "
        f"hits / misses / false_alarms 逐格加总后按 TS = Σhits / "
        f"(Σhits + Σmisses + Σfalse_alarms) **重新算一次**，"
        f"**不是**把各起报日的 TS 加起来除以天数：TS 是比率，那样平均会让只报出"
        f"十几个站、TS 恰好为 0 的起报日与事件上千的起报日**等权**，"
        f"把均值拖偏（见第六节第 2 条）；口径 B 的列是过程累积 TS。"
        f"「缺」= 该段产物缺失；「—」= 有产物但该档无有效 TS（无事件或无有效配对）。"
        f"Δ 一律是「{names[0]} − {names[1]}」，**正值表示主模型领先**。",
        "",
    ]
    grades = "、".join(_grade_label(item) for item in SCREENING_THRESHOLDS)

    _table_caption(lines, numbering,
                   f"各过程口径 A（逐日 24h 汇总）TS —— {len(codes)} 个过程 × {grades}mm")
    _markdown_table(lines, _screening_header(names),
                    _screening_rows(codes, per_model, names, total=False))

    _table_caption(lines, numbering,
                   f"各过程口径 B（过程累积）TS —— {len(codes)} 个过程 × {grades}mm")
    _markdown_table(lines, _screening_header(names),
                    _screening_rows(codes, per_model, names, total=True))

    _figure(lines, numbering, overview_daily_path,
            f"各过程的逐日 24h 汇总 TS（{_grade_label(RANK_THRESHOLD)}mm；"
            f"按 {names[0]} 降序排列，柱高越高越好；与下图**不可比**，仅看本图内的"
            f"模型间高低）")
    _figure(lines, numbering, overview_path,
            f"各过程的过程累积 TS（{_grade_label(RANK_THRESHOLD)}mm；"
            f"按 {names[0]} 降序排列，柱高越高越好）")

    total_counts = _win_counts(codes, per_model, names, total=True)
    day_counts = _win_counts(codes, per_model, names, total=False)
    lines += [
        f"**小结**：全 {len(codes)} 个过程中，口径 B 的胜负是 "
        + _counts_phrase(total_counts, names)
        + "；口径 A 的起报日汇总口径是 "
        + _counts_phrase(day_counts, names)
        + "。两种口径的胜负过程并不完全重合，挑选详析对象时以口径 B 为主，"
        "再点名几个两口径名次相反的过程。",
        "",
    ]
    return lines


def _process_section(
    code: str,
    per_model: Dict[str, Dict[str, ProcessScores]],
    names: Sequence[str],
    numbering: _Numbering,
    daily_path: Path,
    threshold_path: Path,
    index: int,
) -> List[str]:
    process = get(code)
    lines = [f"### 3.{index} {code}（{process.span}，{process.grade}）", ""]
    lines += [f"#### 3.{index}.1 过程概况", "",
              _process_overview(code, per_model, names), ""]

    # ── 3.x.2 口径 A ─────────────────────────────────────────────
    days = process_days(code, per_model, names)
    lines += [f"#### 3.{index}.2 口径 A：逐日 24h TS", ""]
    lines += [
        "起报日之间是**独立的 24h 窗**，每天一个结果、互不合并；"
        "同一天里各模型评的是同一个 08–08 窗，可以直接横向比。",
        "",
    ]
    header = ["起报日"]
    for threshold in SCREENING_THRESHOLDS:
        header += [f"{_grade_label(threshold)} {name}" for name in names]
        header.append(f"Δ({names[0]}−{names[1]})")
    body: List[List[str]] = []
    for day in days:
        cells = [day]
        for threshold in SCREENING_THRESHOLDS:
            values: List[Optional[float]] = []
            for name in names:
                scores = per_model[name].get(code)
                if scores is None or not scores.present:
                    cells.append("缺")
                    values.append(None)
                    continue
                value = cell(scores.days.get(day), threshold)
                cells.append(fmt_number(value))
                values.append(value)
            cells.append(_delta_cell(values))
        body.append(cells)
    if body:
        _table_caption(lines, numbering,
                       f"{code} 逐日 24h TS —— 起报日 × "
                       + "、".join(_grade_label(item) for item in SCREENING_THRESHOLDS) + "mm")
        _markdown_table(lines, header, body)
        _figure(lines, numbering, daily_path,
                f"{code}（{process.span}）逐日 24h TS（08–08 北京时，day-1；"
                f"纵轴 0–1，没有有效 TS 的起报日不画点）")
    else:
        lines += [f"{code} 没有任何起报日的口径 A 产物。", ""]
    lines += [f"**小结**：{_daily_reading(code, per_model, names, days)}", ""]

    # ── 3.x.3 口径 B ─────────────────────────────────────────────
    thresholds = sorted({
        threshold
        for name in names
        for threshold in (thresholds_of(per_model[name][code])
                          if per_model[name].get(code) is not None else [])
    })
    lines += [f"#### 3.{index}.3 口径 B：过程累积 TS", ""]
    lines += [
        f"整段过程只用**一根预报**（{process.start} 前一天 00UTC 起报）累加到底，"
        f"窗长 {window_days(code) * 24}h；因此这一节与上一节**不可比**，"
        f"基准率比逐日窗高得多。",
        "",
    ]
    if thresholds:
        header = ["等级"] + [f"{name} TS" for name in names] + [f"Δ({names[0]}−{names[1]})"]
        body = []
        for threshold in thresholds:
            cells = [_grade_label(threshold)]
            values = []
            for name in names:
                scores = per_model[name].get(code)
                if scores is None or not scores.present:
                    cells.append("缺")
                    values.append(None)
                    continue
                value = total_value(scores, threshold)
                cells.append(fmt_number(value))
                values.append(value)
            cells.append(_delta_cell(values))
            body.append(cells)
        _table_caption(lines, numbering, f"{code} 过程累积 TS —— 降水等级 × 模型")
        _markdown_table(lines, header, body)
        _figure(lines, numbering, threshold_path,
                f"{code}（{process.span}）过程累积 TS 随降水等级的变化"
                f"（等级越高样本越少，高等级档的起伏不作结论）")
    else:
        lines += [f"{code} 没有口径 B 产物。", ""]
    lines += [f"**小结**：{_total_reading(code, per_model, names)}", ""]

    lines += [f"#### 3.{index}.4 过程小结", "",
              _process_reading(code, per_model, names), ""]
    return lines


def _cross_section(
    codes: Sequence[str],
    per_model: Dict[str, Dict[str, ProcessScores]],
    names: Sequence[str],
    numbering: _Numbering,
    scatter_path: Path,
) -> List[str]:
    lines = ["## 四、跨过程综合对比", ""]
    pool = _shared_codes(codes, per_model, names)
    if len(pool) == len(codes):
        scope_note = f"把 {len(codes)} 个过程汇起来看。"
    else:
        scope_note = (
            f"把 {len(codes)} 个过程汇起来看，其中 {len(codes) - len(pool)} 个过程"
            f"各模型产物没同时到齐。跨模型汇总**逐行只取各模型都有有效值的过程**"
            f"（最多 {len(pool)} 个）——各算各的均值会得到两个分母，"
            f"相减出来的 Δ 里混着覆盖率差，不是模型差异。"
        )
    lines += [
        scope_note
        + f"口径 A 是各过程**汇总值**的再平均（过程等权，不是按起报日数加权）；"
        f"口径 B 是各过程累积 TS 的均值。两个口径**各自汇总、不混算**。",
        "",
    ]
    header = ["指标"] + list(names) + [f"Δ({names[0]}−{names[1]})"]
    body: List[List[str]] = []
    counts: List[int] = []

    def _row(label: str, column: str, threshold: float, total: bool) -> None:
        pools = _paired_values(codes, per_model, names, total=total,
                               column=column, threshold=threshold)
        values = {name: _mean_of(pools[name]) for name in names}
        counts.append(len(pools[names[0]]))
        cells = [label] + [fmt_number(values[name]) for name in names]
        cells.append(_delta_cell([values[name] for name in names]))
        body.append(cells)

    _row("口径 A 汇总 TS（≥10mm）", "TS", 10.0, False)
    _row("口径 A 汇总 TS（≥25mm）", "TS", RANK_THRESHOLD, False)
    _row("口径 A 汇总 TS（≥50mm）", "TS", 50.0, False)
    _row("口径 B 累积 TS（≥10mm）", "TS", 10.0, True)
    _row("口径 B 累积 TS（≥25mm）", "TS", RANK_THRESHOLD, True)
    _row("口径 B 累积 TS（≥50mm）", "TS", 50.0, True)
    _row("口径 B 命中率 POD（≥25mm）", "POD", RANK_THRESHOLD, True)
    _row("口径 B 空报率 FAR（≥25mm）", "FAR", RANK_THRESHOLD, True)
    _row("口径 B 频率偏差 BIAS（≥25mm）", "BIAS", RANK_THRESHOLD, True)

    if min(counts) == max(counts):
        pair_note = (
            f"全部 {len(codes)} 个过程"
            if min(counts) == len(codes)
            else f"各模型都有有效值的 {min(counts)} 个过程"
        )
    else:
        pair_note = f"各模型都有有效值的 {min(counts)}–{max(counts)} 个过程"
    _table_caption(lines, numbering,
                   f"跨过程汇总 —— {'、'.join(names)}（各指标为{pair_note}的等权平均，"
                   f"**Δ 同分母**）")
    _markdown_table(lines, header, body)

    points = {
        name: [
            (daily_pooled(per_model[name][code], RANK_THRESHOLD),
             total_value(per_model[name][code], RANK_THRESHOLD))
            for code in pool
        ]
        for name in names
    }
    _figure(lines, numbering, scatter_path,
            f"两种口径的逐过程对照（{_grade_label(RANK_THRESHOLD)}mm；同上 "
            f"{len(pool)} 个过程，每点一个过程，虚线为 y = x，"
            f"点上方的过程累积口径更有利）")

    day_counts = _win_counts(codes, per_model, names, total=False)
    total_counts = _win_counts(codes, per_model, names, total=True)
    same_direction = (
        (day_counts[names[0]] > day_counts[names[1]])
        == (total_counts[names[0]] > total_counts[names[1]])
    )
    day_level = _aggregate(codes, per_model, names, total=False, shared_only=True)
    total_level = _aggregate(codes, per_model, names, total=True, shared_only=True)
    hard_level = _aggregate(codes, per_model, names, total=True, threshold=50.0,
                            shared_only=True)
    lines += [
        f"**小结**：全 {len(codes)} 个过程中，口径 B 的胜负是 "
        + _counts_phrase(total_counts, names)
        + "；口径 A 是 " + _counts_phrase(day_counts, names)
        + ("；两种口径的胜负方向一致" if same_direction else
           "；**两种口径给出的胜负方向不一致**，说明领先与否取决于口径而非模型本身，"
           "引用结论必须带口径")
        + f"。总体水平：口径 B 的 {_grade_label(RANK_THRESHOLD)}mm 档在 "
        + "、".join(f"{name} {fmt_number(total_level[name])}" for name in names)
        + f" 之间，口径 A 的同一档只有 "
        + "、".join(f"{name} {fmt_number(day_level[name])}" for name in names)
        + f"；≥50mm 档更低（"
        + "、".join(f"{name} {fmt_number(hard_level[name])}" for name in names)
        + "），且该档的有效配对数在各过程上远小于低等级档，**这一档的差异不宜当结论**。",
        "",
    ]
    return lines


def _advice_section(
    codes: Sequence[str],
    per_model: Dict[str, Dict[str, ProcessScores]],
    names: Sequence[str],
    change: Optional[str],
) -> List[str]:
    lines = ["## 五、改进建议", ""]
    items: List[str] = []

    day_counts = _win_counts(codes, per_model, names, total=False)
    total_counts = _win_counts(codes, per_model, names, total=True)
    main_total = total_counts[names[0]]
    rival_total = total_counts[names[1]]
    main_day = day_counts[names[0]]
    rival_day = day_counts[names[1]]

    if main_total > rival_total and main_day <= rival_day:
        items.append(
            f"主模型在过程累积口径上 {main_total} 胜 {rival_total} 负，但在逐日口径上"
            f"只有 {main_day} 胜 {rival_day} 负——**优势集中在「总量报得准」，"
            f"不集中在「每天都报得准」**。若业务关心逐日强降水落区，改进重点应放在"
            f"逐日窗的时序误差上，而不是继续优化总量。"
        )
    elif main_total > rival_total and main_day > rival_day:
        items.append(
            f"主模型在两个口径上都领先（过程累积 {main_total} 胜 {rival_total} 负，"
            f"逐日 {main_day} 胜 {rival_day} 负），优势不是靠单个口径撑起来的。"
            f"但两个口径的名次**逐过程并不重合**，引用具体过程时必须带口径。"
        )
    elif main_total < rival_total and main_day >= rival_day:
        items.append(
            f"主模型的过程累积口径以 {main_total} 胜 {rival_total} 负落后，"
            f"逐日口径却是 {main_day} 胜 {rival_day} 负不落后——"
            f"**落后集中在「总量报得准」，不是「每天都报得准」**，"
            f"改进重点应放在过程总量的量级偏差上。"
        )
    elif main_total < rival_total:
        items.append(
            f"主模型在两个口径上都落后（过程累积 {main_total} 胜 {rival_total} 负，"
            f"逐日 {main_day} 胜 {rival_day} 负）；先排查是**过程总量**还是"
            f"**逐日分布**的问题，再定改动方向。"
        )
    else:
        items.append(
            f"两个模型的过程累积口径胜负持平（{main_total}:{rival_total}），"
            f"逐日口径是 {main_day}:{rival_day}；在这批目标上做取舍的收益有限，"
            f"建议换更有区分度的检验对象（更极端的量级档、或另外几年）再看。"
        )

    # 失分最大的几个过程：点名到编号，才能回去单独调阅
    losses = []
    for code in codes:
        ours = per_model[names[0]].get(code)
        rival = per_model[names[1]].get(code)
        if ours is None or rival is None:
            continue
        mine, reference = total_value(ours, RANK_THRESHOLD), total_value(rival, RANK_THRESHOLD)
        if mine is None or reference is None:
            continue
        losses.append((mine - reference, code))
    losses.sort()
    if losses and losses[0][0] < -TIE_THRESHOLD:
        top = "、".join(f"{code}（{delta:+.3f}）" for delta, code in losses[:3])
        items.append(
            f"主模型落后最多的过程是 {top}（口径 B 的 {_grade_label(RANK_THRESHOLD)}mm，"
            f"括号内为主模型 − {names[1]}）。这几个过程建议单独调阅原始预报场，"
            f"确认是漏报还是空报主导。"
        )

    hard_level = _aggregate(codes, per_model, names, total=True, threshold=50.0)
    usable_hard = [value for value in hard_level.values() if value is not None]
    if usable_hard and max(usable_hard) < 0.3:
        items.append(
            f"≥50mm 档的过程累积 TS 最高只有 {fmt_number(max(usable_hard))}，极端量级整体偏弱；"
            f"该档的有效配对数在各过程上远小于低等级档，**先把样本补齐再谈改进**，"
            f"不要在这个量级上调参。"
        )

    coverage = []
    for name in names:
        missing = [code for code in codes
                   if per_model[name].get(code) is None or not per_model[name][code].present]
        if missing:
            coverage.append(
                f"{name} 缺 {len(missing)} 个过程（{'、'.join(missing[:5])}"
                + ("…" if len(missing) > 5 else "") + "）"
            )
    if coverage:
        items.append("补齐缺段后再定论：" + "；".join(coverage) + "。")

    if not change:
        items.append(
            "**本次没有给改动说明，本节的差异只能报、不能归因**。要判断某个模块有没有用，"
            "需要先给出「如果它有用，哪个数字该往哪动」的预期，再回来对照本报告的表。"
        )

    lines += [f"{index}. {text}" for index, text in enumerate(items, start=1)]
    lines.append("")
    return lines


def _caveats_section(archives: Sequence[RainstormArchive]) -> List[str]:
    lines = ["## 六、口径与注意事项", ""]
    lines += [
        "1. **两种口径互不可比**。口径 A 的窗只有 24h，实况达到阈值的站点占比低、"
        "基准率也低；口径 B 把整段累加成一个总量，基准率高得多。同一个过程的 TS "
        "在两者之间可以差好几倍；跨口径比大小没有意义，只在本口径内比。",
        "2. **口径 A 的过程值是把起报日的四格表汇总后重算**，不是逐日 TS 的算术"
        "平均。TS 是比率，简单平均会给「只报出十几个站、TS 恰好为 0」的起报日"
        "与事件上千的起报日**同样的权重**，把均值拖偏；汇总重算等价于把该过程"
        "全部「站点·日」样本当成一个整体，小样本日自然只占很小的权重。"
        "注意口径 A 汇总的仍是**逐日 24h 窗**的样本，与口径 B 那根累积窗不是"
        "同一个量，两者照样不可比。",
        "3. **跨模型汇总只比同一批过程**。某个模型缺段时，各算各的均值就是不同的"
        "分母，相减得到的 Δ 里混着覆盖率差、不是模型差异；第四节的汇总表因此**逐行"
        "只取各模型都有有效值的过程**，各模型的过程数一定相等（表题给出这个数，会小于 43）。"
        "注意「有产物」和「有有效值」是两回事：一段过程可能整个 `_total` 缺、"
        "可能该档无事件，两种都会让某个模型少一个数。",
        f"4. **差值小于 {TIE_THRESHOLD} 一律视为并列**，报告里的「领先 / 落后」都不含"
        f"这一区间；胜负数因此可能加起来小于过程总数。",
        "5. **「缺」与「—」不是一回事**：「缺」是该段产物整个不在（多半是那几天的预报"
        "或观测数据缺），「—」是有产物但该档无有效 TS（无事件或无有效配对）。"
        "两者都不补 0。",
        "6. **高等级档的样本量急剧收缩**。≥100mm 尤其是 ≥250mm 档在很多过程上根本"
        "没有事件，TS 为空是「无事件」而不是「技巧为 0」；这些档的差异不作结论。",
        "7. **指标不重估，只汇总**。逐日值、口径 B 的值都直接来自产物里的 "
        "`diagnostics/categorical_wide.csv`，一个字没动；口径 A 的过程级数值是把"
        "同一份产物里的 hits / misses / false_alarms 按起报日**加总后按定义再算"
        "一次**（第 2 条），没有用默认值或估算值补全任何缺格。",
        "8. **本报告不含降水空间分布**。评测产物里只有站点列联表的聚合统计"
        "（hits / misses / false_alarms / n_pairs / TS…），没有逐站或逐格的降水量，"
        "画不出实况场、也画不出预报−实况差值场。要看落区得回头读原始预报 nc 与"
        "站点报文（配置里的预报根目录 / 站点根目录），那是另一个脚本。",
        "9. **过程边界用 08–08 日界**，与纪要表的 0–0 日历口径头尾各差 8 小时；"
        "口径 B 的窗因此是 (结束日 − 开始日 + 2) 天，比纪要表的自然日数多一天。",
        "10. **202519 与 202520 起止完全相同**（纪要表把同一时段的两个侧面分列两条），"
        "各评各的、不合并——本报告里它们各占一行，数字相同属正常。",
        "",
        "> **数据来源**：" + "；".join(f"{item.name} `{item.root}`" for item in archives) + "。"
        "所有指标由 `configs/rainstorm_ts_single_*` 的评测流程产出；本报告只汇总、不重估，口径 A 的过程级数值按第六节第 2 条由产物里的四格表计数加总得出。",
        "",
    ]
    return lines


# ============================================================== 装配


def _validate_processes(processes: Sequence[str]) -> List[str]:
    """去重保序 + 校验编号存在。"""
    if not processes:
        raise ValueError("至少要给一个过程编号（--process 202501,202502）")
    seen: set = set()
    ordered: List[str] = []
    for code in (str(item).strip() for item in processes):
        if code in seen:
            continue
        if code not in rainstorm_catalog.BY_CODE:
            raise ValueError(
                f"过程编号 {code!r} 不在编目里；有效范围 "
                f"{rainstorm_catalog.codes()[0]}–{rainstorm_catalog.codes()[-1]}"
            )
        seen.add(code)
        ordered.append(code)
    return ordered


def build_rainstorm_report(
    archives: Sequence[RainstormArchive],
    out_dir: Path,
    *,
    processes: Sequence[str],
    title: str = "2025 年暴雨过程个例检验报告",
    archive: Optional[str] = None,
    change: Optional[str] = None,
) -> Dict[str, Path]:
    """写图 + ``REPORT.md``，返回 ``{相对路径: 绝对路径}`` 的产物清单。

    Args:
        archives: 产物根，第一个是主模型；**至少两个**（对比表没有意义）。
        out_dir: 报告输出目录，图和 md 都写这里。
        processes: 要详析的过程编号，**必给**。
        title: 报告 H1 标题。
        archive: 归档名；给了就在标题下加一行**归档**。
        change: 本次改动说明；不给就在「改进建议」里说明差异只能报、不能归因。

    Raises:
        ValueError: 模型不足两个 / 名字重复 / 过程编号非法。
        ImportError: 没装 matplotlib（出图必需）。
    """
    try:
        from reporting.reports.rainstorm_plots import RainstormPlotter
    except ImportError as error:  # pragma: no cover - 取决于运行环境
        raise ImportError(
            "出图需要 matplotlib（pip install matplotlib）；只要正文和 CSV 请改用 "
            "skills/evaluation-reporting/scripts/render_rainstorm_summary.py"
        ) from error

    archives = list(archives)
    if len(archives) < 2:
        raise ValueError(f"至少要两个模型（对比报告），收到 {len(archives)} 个")
    names = [item.name for item in archives]
    if len(set(names)) != len(names):
        raise ValueError(f"模型名重复：{names}；同名会让表格列对不上号")

    selected = _validate_processes(processes)
    codes = rainstorm_catalog.codes()
    out_dir = Path(out_dir)
    figure_dir = out_dir / FIGURES_DIR
    figure_dir.mkdir(parents=True, exist_ok=True)

    per_model = load_all(archives, codes)
    plotter = RainstormPlotter()
    numbering = _Numbering()
    artifacts: Dict[str, Path] = {}

    def _register(relative: Path) -> Path:
        artifacts[Path(relative).as_posix()] = out_dir / relative
        return relative

    # ── 图 1、图 2：全过程的两种口径分组条形（口径 A 在前）───────────
    overview_daily_rel = _register(Path(FIGURES_DIR) / "process_overview_daily.png")
    plotter.plot_process_overview(
        codes,
        {
            name: {
                code: daily_pooled(per_model[name][code], RANK_THRESHOLD)
                for code in codes
                if per_model[name].get(code) is not None
            }
            for name in names
        },
        threshold=RANK_THRESHOLD,
        save_path=out_dir / overview_daily_rel,
        order=names[0],
        metric_label="逐日 24h 汇总",
    )
    overview_rel = _register(Path(FIGURES_DIR) / "process_overview.png")
    plotter.plot_process_overview(
        codes,
        {
            name: {
                code: total_value(per_model[name][code], RANK_THRESHOLD)
                for code in codes
                if per_model[name].get(code) is not None
            }
            for name in names
        },
        threshold=RANK_THRESHOLD,
        save_path=out_dir / overview_rel,
        order=names[0],
    )

    # ── 第 3 节：每个详析过程两张图 ────────────────────────────────
    per_process_rels: Dict[str, Tuple[Path, Path]] = {}
    for code in selected:
        daily_rel = _register(Path(FIGURES_DIR) / f"{code}_daily_ts.png")
        threshold_rel = _register(Path(FIGURES_DIR) / f"{code}_threshold_ts.png")
        per_process_rels[code] = (daily_rel, threshold_rel)

        days = process_days(code, per_model, names)
        series = {
            name: {
                day: {threshold: cell(frame, threshold)
                      for threshold in SCREENING_THRESHOLDS}
                for day, frame in (
                    per_model[name][code].days if per_model[name].get(code) is not None else {}
                ).items()
            }
            for name in names
        }
        plotter.plot_daily_ts(days, series, SCREENING_THRESHOLDS,
                              code=code, span=get(code).span,
                              save_path=out_dir / daily_rel)

        thresholds = sorted({
            threshold
            for name in names
            for threshold in (thresholds_of(per_model[name][code])
                              if per_model[name].get(code) is not None else [])
        })
        plotter.plot_threshold_bars(
            thresholds,
            {
                name: {
                    threshold: (total_value(per_model[name][code], threshold)
                                if per_model[name].get(code) is not None else None)
                    for threshold in thresholds
                }
                for name in names
            },
            code=code, span=get(code).span, save_path=out_dir / threshold_rel,
        )

    # ── 图：两口径逐过程散点 ──────────────────────────────────────
    scatter_rel = _register(Path(FIGURES_DIR) / "caliber_scatter.png")
    plotter.plot_caliber_scatter(
        {
            name: [
                (daily_pooled(per_model[name][code], RANK_THRESHOLD),
                 total_value(per_model[name][code], RANK_THRESHOLD))
                for code in codes
                if per_model[name].get(code) is not None
            ]
            for name in names
        },
        threshold=RANK_THRESHOLD,
        save_path=out_dir / scatter_rel,
    )

    # ── 正文 ──────────────────────────────────────────────────────
    lines: List[str] = []
    lines += _header_lines(archives, selected, title, archive, change)
    lines += _conclusion_section(codes, per_model, names, selected, change)
    lines += _screening_section(
        codes, per_model, names, numbering, overview_daily_rel, overview_rel
    )
    lines += ["## 三、逐过程分析", ""]
    for index, code in enumerate(selected, start=1):
        daily_rel, threshold_rel = per_process_rels[code]
        lines += _process_section(code, per_model, names, numbering,
                                  daily_rel, threshold_rel, index)
    lines += _cross_section(codes, per_model, names, numbering, scatter_rel)
    lines += _advice_section(codes, per_model, names, change)
    lines += _caveats_section(archives)

    report_path = out_dir / "REPORT.md"
    report_path.write_text("\n".join(lines).rstrip("\n") + "\n", encoding="utf-8")
    artifacts["REPORT.md"] = report_path
    return artifacts
