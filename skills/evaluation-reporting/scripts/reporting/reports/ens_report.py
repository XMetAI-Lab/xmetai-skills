#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""集合预报多模型评估报告：读评估产物 → 渲染 Markdown。

输入是集合支线的评估产物目录（形状同 ``det_report``）；集合特有指标在长表里是
``crps`` / ``spread`` / ``spread_error_ratio``。

**三条硬差别**（踩了会静默算错）：``spread_error`` 展开出的 rmse 与确定性 rmse
同名，只能靠 ``product_kind`` 区分（``det_report.metric_rows``）；本模块多 CRPS
与离散度；谱的 log-RMS 用 **ln**（``det_report.spectrum_rms`` 是 log10，差
×2.3026），§4/§5 一律自算，不读 ``base.spectrum_rms``，``log_rms_ln`` 复用
:mod:`visualization.wave_report`。取数一律走 ``init_date``。

格式契约见 ``assets/templates/weather_rmse_ens.md``：表不编号、图带 ``图 N：``。
图 1 是总体热力图，图 2–6 对应 §4.1–§4.5，图 7–6+k 是 §4.6 空间分布
（k = 变量数 × 月份数，k=0 时无此节图），图 7+k 是 §4.7 纬向谱，图 8+k–9+k 是
§4.8 球谐带（不可得时保留节位、标「本批未出」且两图一起缺席）。图一律 SVG。
球谐带与单成员 §4.5 共用 ``wave_report`` 的 helper，但算在集合平均场上，
数字不可互相搬运。
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import (
    Any,
    Dict,
    List,
    Mapping,
    NamedTuple,
    Optional,
    Sequence,
    Tuple,
)

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from reporting.core.common import (
    bullet_lines,
    cn_number,
    date_scope_counts,
    date_scope_detail,
    date_scope_word,
    date_span,
    fmt_number,
    lead_span,
    own_dates_mode,
    sph_coverage_notes,
    table_lines,
)
from reporting.reports.det_plots import DetPlotter, plot_overall_heatmap
from reporting.reports.spatial_report import (
    SpatialPayload,
    build_spatial_section,
    load_spatial_csv,
    render_spatial,
)
from reporting.core.style import (
    BASE_WIDTH,
    LEGEND_FRAME,
    NEUTRAL,
    apply_style,
    font_sizes,
    model_colors,
    snap_width,
)
from reporting.reports.det_report import (
    Archive,
    Bundle,
    LEAD_BANDS,
    PairTest,
    _best_cell,
    _best_names,
    _relative,
    _signed,
    analyze,
    by_date_frames,
    load_spectrum,
    metric_rows,
    paired_tests,
    region_display,
    region_table,
    report_date,
)
from reporting.reports.wave_report import (
    SPHERICAL_ABSENT,
    SPHERICAL_BAND_COUNT,
    SPHERICAL_BANDS,
    _band_label,
    _band_tag,
    _bias_pct,
    _grouped_bars,
    _mean_ratio,
    _pooled_logrms,
    _save_figure,
    band_daily_logrms,
    band_dates,
    load_band_power,
    log_ratio_arrays,
    log_rms_ln,
    probe_bands,
)


#: 综合相对 RMSE 的基线（各模型几何均值 = 100）
BASELINE = 100.0

#: 共同日期少于这个数就在 §1 / §6 / §7 点名提示检验力不足
SMALL_SAMPLE_DATES = 30

#: 纬向谱偏差的口径标签，§3 / §4 / §5 共用一套措辞
SPECTRUM_DEVIATION_LABEL = "纬向谱偏差"

#: 球谐带两个汇总标签；与骨架 §3/§4 的表头一致
SPHERICAL_RMS_LABEL = "球谐谱RMS×100"
SPHERICAL_BIAS_LABEL = "球谐比偏差(%)"

#: §6 指标族名 → 是否越大越好（不在表里的族默认越小越好）
_FAMILY_DIRECTION = {"ACC": True}


# ====================================================================== 加载


def load_daily_frames(
    root: Path, stem: str, dates: Sequence[str], name: str = "", *, region: Optional[str] = None
) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """集合特有指标 ``stem`` → ``(按 lead 平均的表, {起报日: 原表})``。

    两份都返回：出图与总表要按 lead 平均的表，配对检验要**逐日样本**
    （先平均再检验等于把样本量从 N 天压成 1 个点）。

    长表里 ``crps`` 只有一份（``product_kind=ensemble``），不像 ``rmse`` 那样
    和确定性指标撞名，所以这里不按 ``product_kind`` 过滤。

    Raises:
        FileNotFoundError: 某个共同起报日缺这个指标的结果——不静默跳过。
        ValueError: 各起报日的列集合不一致——跨日平均会按「有几天算几天」
            静默缩样本，让同一行里两个模型的样本量不同。
    """
    rows = metric_rows(root, stem, region=region)
    frames = by_date_frames(rows, dates, f"{name or root}: {stem}")
    columns: Optional[List[str]] = None
    for date, frame in frames.items():
        own = sorted(str(c) for c in frame.columns)
        if columns is None:
            columns = own
        elif own != columns:
            raise ValueError(
                f"{name or root}: {stem} 各起报日的列不一致——"
                f"{date} 是 {own}，其余起报日是 {columns}；"
                f"跨日平均会静默缩样本，不做"
            )
    if not frames:
        raise ValueError(f"{name or root}: 没有可读的 {stem} 结果")
    stacked = pd.concat(frames.values(), axis=0)
    return stacked.groupby(level=0).mean().sort_index(), frames


def load_spread_rmse_ratio(
    root: Path, dates: Sequence[str], name: str = "", *, region: Optional[str] = None
) -> Tuple[Dict[str, pd.DataFrame], str]:
    """Spread/RMSE → ``({起报日: lead × 变量}, 来源标签)``。

    ``spread_error`` 指标在长表里展开成 ``spread`` / ``rmse`` /
    ``spread_error_ratio`` 三行，优先直接用现成的 ratio；**任一**起报日缺它，
    就整批退到 ``spread ÷ rmse`` 现算——不混口径（一半读现成、一半现算，
    两个来源的小数位与分母处理未必一致）。

    注意这里的 ``rmse`` 要的是**集合口径**那一行（``product_kind=ensemble``），
    不是同名的确定性 RMSE。
    """
    ratio_rows = metric_rows(root, "spread_error_ratio", region=region)
    have = set(ratio_rows["init_date"].astype(str))
    if all(str(date) in have for date in dates):
        return by_date_frames(ratio_rows, dates, f"{name or root}: spread_error_ratio"), \
            "产物自带的 spread_error_ratio"

    spread = by_date_frames(
        metric_rows(root, "spread", region=region), dates, f"{name or root}: spread"
    )
    rmse = by_date_frames(
        metric_rows(root, "rmse", product_kind="ensemble", region=region),
        dates, f"{name or root}: 集合 rmse",
    )
    frames: Dict[str, pd.DataFrame] = {}
    for date in spread:
        own_spread, own_rmse = spread[date], rmse[date]
        shared = [c for c in own_spread.columns if c in own_rmse.columns]
        if not shared:
            raise ValueError(
                f"{name or root}: {date} 的 spread 与 rmse 没有同名变量；"
                f"spread 有 {list(own_spread.columns)}，rmse 有 {list(own_rmse.columns)}"
            )
        frames[date] = own_spread[shared] / own_rmse[shared].replace(0.0, np.nan)
    return frames, "由 spread ÷ 集合口径 rmse 现算"


def _per_date_means(frames: Mapping[str, pd.DataFrame]) -> Dict[str, pd.Series]:
    """``{日期: lead × 变量}`` → ``{变量: 日期 → 该变量在全部 lead 上的均值}``。"""
    if not frames:
        return {}
    variables = sorted({str(c) for frame in frames.values() for c in frame.columns})
    values: Dict[str, Dict[str, float]] = {variable: {} for variable in variables}
    for date, frame in frames.items():
        for variable in variables:
            if variable not in frame.columns:
                continue
            numbers = pd.to_numeric(frame[variable], errors="coerce")
            values[variable][date] = float(numbers.mean())
    return {
        variable: pd.Series(series).sort_index()
        for variable, series in values.items()
        if series
    }


def _shared_columns(frames: Mapping[str, pd.DataFrame]) -> List[str]:
    """各模型都有的列名（字典序）。"""
    shared: Optional[set] = None
    for frame in frames.values():
        own = {str(c) for c in frame.columns}
        shared = own if shared is None else (shared & own)
    return sorted(shared or set())


def _unique_summary_value(archive: Archive, column: str) -> Optional[int]:
    """派生 summary 视图里某列的唯一数值；列缺失或有多值就返回 None。

    新产物只在长表里记「算出来的数」，不记集合成员数这类**运行参数**，所以
    ``n_members`` 一律取不到、渲染成「—」。``n_leads`` 有（由长表现算）。
    """
    if column not in archive.summary.columns:
        return None
    values = pd.to_numeric(archive.summary[column], errors="coerce").dropna().unique()
    return int(values[0]) if len(values) == 1 else None


def _acc_per_variable(
    root: Path, dates: Sequence[str], variables: Sequence[str], name: str = ""
) -> Dict[str, pd.Series]:
    """``acc`` 的确定性行 → ``{变量: 起报日 → 全 lead 平均 ACC(%)}``。

    ``det_report._acc_by_date`` 只算一个变量（报告只要 ``acc_variable``），
    §6 的 ACC 指标族要逐变量，所以这里重算一遍。``analyze`` 已经在入口乘过
    ``BASELINE``，本函数同样乘——**下游不要再乘一次**。
    """
    acc = metric_rows(root, "acc", product_kind="deterministic")
    per_date = acc.groupby([acc["init_date"].astype(str), acc["variable"].astype(str)])["value"].mean()
    wanted = [str(date) for date in dates]
    values: Dict[str, Dict[str, float]] = {}
    for variable in variables:
        variable = str(variable)
        try:
            series = per_date.xs(variable, level=1)
        except KeyError:
            available = sorted({str(v) for v in acc["variable"].dropna()})
            raise ValueError(
                f"{name or root}: 产物里没有变量 {variable} 的 acc 结果；现有变量: {available}"
            ) from None
        missing = [date for date in wanted if date not in series.index]
        if missing:
            raise FileNotFoundError(
                f"{name or root}: {len(missing)} 个起报日没有 {variable} 的 acc 结果"
                f"（如 {missing[0]}）"
            )
        values[variable] = {date: BASELINE * float(series[date]) for date in wanted}
    return {variable: pd.Series(series).sort_index() for variable, series in values.items()}


# ================================================================== 统计检验


def _dominant(frame: pd.DataFrame, names: Sequence[str], *, higher_is_better: bool = False) -> str:
    """``日期 × 模型`` 表里显著优于**其余全部**模型的那个模型；没有就「无显著差异」。

    判定用 Holm 校正后的 p（``paired_tests`` 已经在做）。只要有一个模型压过
    其余全部，这一项就归它；否则整项记无显著差异——于是「各模型优胜项数 +
    无显著差异数 = 该项总数」，不会重复计数。
    """
    tests = paired_tests(frame, higher_is_better=higher_is_better)
    if not tests:
        return "无显著差异"
    for name in names:
        involved = [t for t in tests if name in (t.a, t.b)]
        if involved and all(t.significant and t.better == name for t in involved):
            return str(name)
    return "无显著差异"


def _family_table(
    families: Mapping[str, Mapping[str, pd.DataFrame]],
    names: Sequence[str],
) -> pd.DataFrame:
    """``{指标族: {项名: 日期 × 模型}}`` → 每族每模型的显著优胜项计数。

    Returns:
        index = 指标族名（字典序），columns = 模型名 + ``无显著差异``。
    """
    rows = {}
    for family in sorted(families):
        higher = _FAMILY_DIRECTION.get(family, False)
        counts = {str(name): 0 for name in names}
        ties = 0
        for _, frame in sorted(families[family].items()):
            winner = _dominant(frame, names, higher_is_better=higher)
            if winner == "无显著差异":
                ties += 1
            else:
                counts[winner] += 1
        rows[family] = {**counts, "无显著差异": ties}
    return pd.DataFrame(rows).T


def _log_deviation(ratio: pd.DataFrame) -> pd.Series:
    """逐 lead 的 ``|ln(ratio)|`` 在变量上的平均（ratio ≤ 0 的样本丢弃）。"""
    return np.log(ratio.where(ratio > 0)).abs().mean(axis=1)


def _log_deviation_scalar(ratio: pd.DataFrame) -> float:
    """跨变量 × lead 的 ``mean|ln(Spread/RMSE)|``（§4 表头的「Spread偏差(绝对对数)」）。"""
    values = _log_deviation(ratio).dropna()
    return float(values.mean()) if not values.empty else float("nan")


# ================================================================== 汇总分析


class EnsBundle(NamedTuple):
    """``det_report.Bundle`` + 集合特有量。

    第一个字段是 ``base``，``names`` / ``dates`` / ``archives`` 等经 property
    透传，于是各 ``_section_*`` 里 ``bundle.xxx`` 的写法和 ``Bundle`` 一致。
    """

    base: Bundle
    members: Dict[str, Optional[int]]
    leads: Dict[str, Optional[int]]
    member_label: str
    crps_by_lead: Dict[str, pd.DataFrame]  # 模型 -> lead × 变量
    crps_per_date: Dict[str, Dict[str, pd.Series]]  # 模型 -> 变量 -> 日期
    crps_variables: List[str]
    crps_mean: pd.Series  # 模型 -> CRPS
    acc_variables: List[str]
    acc_per_date: Dict[str, Dict[str, pd.Series]]
    spread_ratio_by_lead: Dict[str, pd.DataFrame]  # 模型 -> lead × 变量
    spread_ratio_per_date: Dict[str, Dict[str, pd.Series]]
    spread_variables: List[str]
    spread_ratio_source: str
    spread_log_deviation: pd.Series  # 模型 -> mean|ln(Spread/RMSE)|
    #: 带名 × 模型，该带上的平均 Spread/RMSE（与正文同源同口径，只是换了统计范围）。
    #: 某个带取不到 spread 行时该行整体缺席，缺席的带名记在 ``region_spread_skipped``。
    region_spread_ratio: pd.DataFrame
    region_spread_skipped: List[str]
    spectrum_by_variable: Dict[str, Dict[str, pd.DataFrame]]  # 变量 -> 模型 -> 波数×{pred,obs}
    spectrum_deviation: pd.DataFrame  # 谱变量 × 模型（%）
    spectrum_deviation_scalar: pd.Series  # 模型 -> %（谱变量平均）
    # --- 球谐带功率（集合平均场口径；与 wave 报告的单成员口径不同源）---
    sph_available: bool
    sph_rms: pd.Series  # 模型 -> 全频带/变量/lead/日期总 log-RMS×100
    sph_bias: pd.Series  # 模型 -> mean|pred/obs-1|×100
    sph_band_logrms: pd.DataFrame  # 频带 × 模型
    sph_band_ratio: pd.DataFrame  # 频带 × 模型（均值比）
    sph_variable: pd.DataFrame  # 变量 × 模型
    sph_tests: Dict[str, List[PairTest]]  # 频带 -> 配对检验列表
    sph_coverage: Dict[str, Dict[str, Tuple[int, int]]]  # 模型 -> 变量 -> (用, 请求)
    family_wins: pd.DataFrame  # 指标族 × (模型..., 无显著差异)
    family_totals: Dict[str, int]  # 指标族 -> 项数
    small_sample: bool

    # --- 透传，让 _section_* 能把 EnsBundle 当 Bundle 用 ---
    @property
    def names(self) -> List[str]:
        return self.base.names

    @property
    def dates(self) -> List[str]:
        return self.base.dates

    @property
    def dates_by_model(self) -> Dict[str, List[str]]:
        return self.base.dates_by_model

    @property
    def archives(self) -> Dict[str, Archive]:
        return self.base.archives

    @property
    def common_variables(self) -> List[str]:
        return self.base.common_variables

    @property
    def fa_variables(self) -> List[str]:
        return self.base.fa_variables

    @property
    def spectrum_variables(self) -> List[str]:
        return self.base.spectrum_variables

    @property
    def acc_variable(self) -> str:
        return self.base.acc_variable


def _common_dates_or_raise(archives: Sequence[Archive]) -> List[str]:
    """共同日期 < 2 时点名报错：说清谁和谁没有交集，而不是只说「共同日期不够」。"""
    sets = {a.name: set(a.summary["init_date"]) for a in archives}
    shared = set.intersection(*sets.values()) if sets else set()
    if len(shared) >= 2:
        return sorted(shared)

    spans = "；".join(
        f"{name} {date_span(sorted(own))}（{len(own)} 天）" for name, own in sets.items()
    )
    order = list(sets)
    pairs = [
        f"{a}∩{b} {len(sets[a] & sets[b])} 天"
        for index, a in enumerate(order)
        for b in order[index + 1:]
    ]
    raise ValueError(
        f"共同日期只有 {len(shared)} 天，做不了配对检验。\n"
        f"  各模型自有的起报日期：{spans}\n"
        f"  两两交集：{'；'.join(pairs)}\n"
        f"  没有共同日期的模型之间无法比较——先补齐数据，"
        f"或把它们拆成有交集的两组、各出一份报告。"
    )


def analyze_ens(archives: Sequence[Archive], *,
                members: Optional[int] = None,
                own_dates: bool = False) -> EnsBundle:
    """算齐集合报告要的一切。

    Args:
        members: 集合成员数，命令行给的**字面量**。产物里根本没记这个字段
            （见 :func:`_unique_summary_value`），所以报告要写成员数只能从外面传；
            给了就对**所有模型**一律用它。不给就退回产物里若有若无的
            ``n_members`` 列，新产物上取不到、渲染成「—」。
        own_dates: 每个模型的 CRPS／离散度／ACC／谱按**它自己的全部起报日**算，
            而不是只在共同日期上算；只有 §6 的配对检验退回共同日期。
            见 :func:`det_report.analyze` 的同名参数。

    Raises:
        ValueError: 共同日期 < 2（点名说清谁和谁没交集）、缺逐日文件、
            各日期的列集合不一致。
    """
    _common_dates_or_raise(archives)
    base = analyze(archives, own_dates=own_dates)
    names = base.names
    dates = base.dates  # 共同日期：§6 与 §2「有效共同日期」那一行用
    dates_by_model = base.dates_by_model  # 各模型本次统计实际用的起报日

    def _pair(frame: pd.DataFrame) -> pd.DataFrame:
        """裁到共同日期；``own_dates`` 时再丢掉任何模型缺数的那些日子。

        §6 的指标族是**逐日配对**的（项 = 变量、样本 = 日期），同一次 Wilcoxon 里
        所有模型必须站在同一批日期上；各模型覆盖时段不同时，并集里那些只有一两个
        模型有数的日子谁也代表不了，直接不要。默认路径**不丢**，逐字保持既有行为。
        """
        frame = frame.reindex(index=[str(d) for d in dates])
        return frame.dropna(how="any") if own_dates else frame

    members = {
        a.name: members if members is not None else _unique_summary_value(a, "n_members")
        for a in archives
    }
    leads = {a.name: _unique_summary_value(a, "n_leads") for a in archives}

    # --- 集合特有指标：CRPS 与离散度（逐日读，不用 summary 里 precomputed 的列）---
    crps_by_lead: Dict[str, pd.DataFrame] = {}
    crps_per_date: Dict[str, Dict[str, pd.Series]] = {}
    spread_ratio_by_lead: Dict[str, pd.DataFrame] = {}
    spread_ratio_per_date: Dict[str, Dict[str, pd.Series]] = {}
    spread_sources = set()
    for archive in archives:
        by_lead, frames = load_daily_frames(
            archive.root, "crps", dates_by_model[archive.name], archive.name
        )
        crps_by_lead[archive.name] = by_lead
        crps_per_date[archive.name] = _per_date_means(frames)

        ratio_frames, source = load_spread_rmse_ratio(
            archive.root, dates_by_model[archive.name], archive.name
        )
        spread_sources.add(source)
        stacked = pd.concat(ratio_frames.values(), axis=0)
        spread_ratio_by_lead[archive.name] = stacked.groupby(level=0).mean().sort_index()
        spread_ratio_per_date[archive.name] = _per_date_means(ratio_frames)

    crps_variables = _shared_columns(crps_by_lead)
    spread_variables = _shared_columns(spread_ratio_by_lead)
    if not crps_variables:
        raise ValueError("这些模型的 crps 逐日文件没有共享变量，算不出 CRPS 指标")
    if not spread_variables:
        raise ValueError("这些模型的离散度比值没有共享变量，算不出 Spread/RMSE")

    # --- ACC 逐变量（§6 的 ACC 指标族要逐变量，base 只给 acc_variable）---
    acc_columns = _shared_columns(base.acc_by_lead)
    acc_per_date = {
        archive.name: _acc_per_variable(
            archive.root, dates_by_model[archive.name], acc_columns, archive.name
        )
        for archive in archives
    }

    # --- 纬向谱：逐变量、逐模型，口径与 §4.7 一致（ln）---
    spectrum_by_variable: Dict[str, Dict[str, pd.DataFrame]] = {}
    spectrum_rows: Dict[str, Dict[str, float]] = {name: {} for name in names}
    for variable in base.spectrum_variables:
        per_model: Dict[str, pd.DataFrame] = {}
        for archive in archives:
            pred, obs = load_spectrum(
                archive.root, dates_by_model[archive.name], variable, archive.name
            )
            per_model[archive.name] = pd.DataFrame(
                {"pred": pred.mean(axis=1), "obs": obs.mean(axis=1)}
            )
            spectrum_rows[archive.name][variable] = float(log_rms_ln(pred, obs).mean())
        spectrum_by_variable[variable] = per_model

    # pd.DataFrame({模型: {谱变量: 值}}) → index = 谱变量、columns = 模型
    spectrum_deviation = pd.DataFrame(spectrum_rows).reindex(base.spectrum_variables)
    spectrum_deviation_scalar = spectrum_deviation.mean(axis=0)

    # --- 球谐带功率：与纬向谱同一组变量，口径是集合平均场 ---
    # 读数与统计全部复用 wave_report 的球谐带工具；区别只在变量集和日期集来自
    # 集合 Bundle。完全取不到可用频带时整块标「本批未出」，不制造空表。
    band_names = [_band_label(lo, hi) for lo, hi in SPHERICAL_BANDS]
    sph_variables = list(base.spectrum_variables)
    sph_available = bool(sph_variables) and all(
        probe_bands(a.root, dates_by_model[a.name], sph_variables) for a in archives
    )
    sph_tests: Dict[str, List[PairTest]] = {}
    sph_rms_values: Dict[str, float] = {}
    sph_bias_values: Dict[str, float] = {}
    sph_band_logrms = pd.DataFrame(index=band_names)
    sph_band_ratio = pd.DataFrame(index=band_names)
    sph_variable = pd.DataFrame(index=sph_variables)
    sph_coverage: Dict[str, Dict[str, Tuple[int, int]]] = {}
    if sph_available:
        logrms_rows: Dict[str, Dict[str, float]] = {}
        ratio_rows: Dict[str, Dict[str, float]] = {}
        variable_rows: Dict[str, Dict[str, float]] = {}
        daily: Dict[str, Dict[str, pd.Series]] = {}
        for archive in archives:
            logs: Dict[str, List[np.ndarray]] = {}
            ratios: Dict[str, List[np.ndarray]] = {}
            per_variable: Dict[str, List[np.ndarray]] = {}
            daily_parts: Dict[str, List[pd.Series]] = {b: [] for b in band_names}
            wanted = list(dates_by_model[archive.name])
            coverage: Dict[str, Tuple[int, int]] = {}
            for variable in sph_variables:
                used = band_dates(archive.root, wanted, variable)
                coverage[variable] = (len(used), len(wanted))
                if not used:
                    continue
                table = load_band_power(archive.root, used, variable, archive.name)
                for low, high in SPHERICAL_BANDS:
                    label = _band_label(low, high)
                    pair = table.get(_band_tag(low, high))
                    if pair is None:
                        continue
                    log_ratio, ratio = log_ratio_arrays(*pair)
                    logs.setdefault(label, []).append(log_ratio)
                    ratios.setdefault(label, []).append(ratio)
                    per_variable.setdefault(variable, []).append(log_ratio)
                    daily_parts[label].append(band_daily_logrms(*pair))
            logrms_rows[archive.name] = {
                b: _pooled_logrms(logs.get(b, [])) for b in band_names
            }
            ratio_rows[archive.name] = {
                b: _mean_ratio(ratios.get(b, [])) for b in band_names
            }
            variable_rows[archive.name] = {
                v: _pooled_logrms(samples) for v, samples in per_variable.items()
            }
            sph_rms_values[archive.name] = _pooled_logrms(
                [sample for b in band_names for sample in logs.get(b, [])]
            )
            sph_bias_values[archive.name] = _bias_pct(
                [sample for b in band_names for sample in ratios.get(b, [])]
            )
            daily[archive.name] = {
                b: (
                    pd.concat(daily_parts[b], axis=1).mean(axis=1)
                    if daily_parts[b]
                    else pd.Series(dtype=float)
                )
                for b in band_names
            }
            sph_coverage[archive.name] = coverage
        for label in band_names:
            frame = pd.DataFrame({name: daily[name][label] for name in names})
            sph_tests[label] = paired_tests(_pair(frame))
        sph_band_logrms = pd.DataFrame(logrms_rows).reindex(index=band_names)
        sph_band_ratio = pd.DataFrame(ratio_rows).reindex(index=band_names)
        sph_variable = pd.DataFrame(variable_rows).reindex(index=sph_variables)
    sph_rms = pd.Series(
        {name: sph_rms_values.get(name, float("nan")) for name in names}
    )
    sph_bias = pd.Series(
        {name: sph_bias_values.get(name, float("nan")) for name in names}
    )

    crps_mean = pd.Series(
        {name: float(crps_by_lead[name].mean(axis=1).mean()) for name in names}
    )
    spread_log_deviation = pd.Series(
        {name: _log_deviation_scalar(spread_ratio_by_lead[name]) for name in names}
    )

    # --- §6 指标族：项 = 变量，样本 = **共同日期**（逐日配对）---
    # 上面那几个逐日字典（crps/spread/acc/谱）在 own_dates=True 时是按各模型自己的
    # 日期算的，长度不一；这里塞进 DataFrame 后统一裁到共同日期，检验才成立。
    # own_dates=False 时所有帧本来就只有共同日期，``_pair`` 是恒等变换。
    families: Dict[str, Dict[str, pd.DataFrame]] = {
        "RMSE": {v: _pair(f) for v, f in base.per_variable_rmse.items()},
        "CRPS": {
            variable: _pair(pd.DataFrame({name: crps_per_date[name][variable] for name in names}))
            for variable in crps_variables
        },
        "ACC": {
            variable: _pair(pd.DataFrame({name: acc_per_date[name][variable] for name in names}))
            for variable in sorted(acc_per_date[names[0]])
        },
        "FA ratio 偏差": {
            variable: _pair(pd.DataFrame(
                    {name: (base.fa_ratio_by_lead[name][variable] - 1.0).abs() for name in names}
                ))
            for variable in base.fa_variables
        },
        "Spread/RMSE 距1": {
            variable: _pair(pd.DataFrame(
                    {name: (spread_ratio_per_date[name][variable] - 1.0).abs() for name in names}
                ))
            for variable in spread_variables
        },
    }
    # 无该项数据的指标族不出现在表里（骨架 §6 的要求）
    families = {
        family: items
        for family, items in families.items()
        if items and all(len(frame.columns) == len(names) for frame in items.values())
    }
    family_wins = _family_table(families, names)
    family_totals = {family: len(items) for family, items in families.items()}

    distinct_members = sorted({m for m in members.values() if m is not None})
    if not distinct_members:
        member_label = "—"
    elif len(distinct_members) == 1:
        member_label = str(distinct_members[0])
    else:
        member_label = "/".join(str(m) for m in distinct_members)

    # 分纬度带的离散度：口径与正文的 Spread/RMSE 完全一致（同样优先用产物自带的
    # spread_error_ratio），只是把统计范围换成单个带。取不到的带**不塞 NaN 充数**，
    # 整行缺席并把带名记下来，渲染层据此说明，不让人以为那个带算出过 0 或 1。
    per_region_spread: Dict[str, pd.Series] = {}
    region_spread_skipped: List[str] = []
    for region in base.region_names:
        own: Dict[str, float] = {}
        try:
            for archive in archives:
                frames, _source = load_spread_rmse_ratio(
                    archive.root, dates_by_model[archive.name], archive.name, region=region
                )
                own[archive.name] = float(np.nanmean(pd.concat(frames.values(), axis=0).values))
        except (FileNotFoundError, ValueError):
            region_spread_skipped.append(region)
            continue
        per_region_spread[region] = pd.Series(own)
    region_spread_ratio = pd.DataFrame(per_region_spread).T

    return EnsBundle(
        base=base,
        members=members,
        leads=leads,
        member_label=member_label,
        crps_by_lead=crps_by_lead,
        crps_per_date=crps_per_date,
        crps_variables=crps_variables,
        crps_mean=crps_mean,
        acc_variables=sorted(acc_per_date[names[0]]),
        acc_per_date=acc_per_date,
        spread_ratio_by_lead=spread_ratio_by_lead,
        spread_ratio_per_date=spread_ratio_per_date,
        spread_variables=spread_variables,
        spread_ratio_source="；".join(sorted(spread_sources)),
        spread_log_deviation=spread_log_deviation,
        region_spread_ratio=region_spread_ratio,
        region_spread_skipped=region_spread_skipped,
        spectrum_by_variable=spectrum_by_variable,
        spectrum_deviation=spectrum_deviation,
        spectrum_deviation_scalar=spectrum_deviation_scalar,
        sph_available=sph_available,
        sph_rms=sph_rms,
        sph_bias=sph_bias,
        sph_band_logrms=sph_band_logrms,
        sph_band_ratio=sph_band_ratio,
        sph_variable=sph_variable,
        sph_tests=sph_tests,
        sph_coverage=sph_coverage,
        family_wins=family_wins,
        family_totals=family_totals,
        small_sample=len(dates) < SMALL_SAMPLE_DATES,
    )


# ====================================================================== 出图


def _spectrum_panels(
    out_dir: Path,
    filename: str,
    spectra: Mapping[str, Mapping[str, pd.DataFrame]],
    *,
    variables: Sequence[str],
    names: Sequence[str],
) -> Path:
    """一个谱变量一个双对数子图，子图内 N 条模型预测谱 + 一条共享观测谱。

    ``DetPlotter.plot_model_spectrum`` 只收**单个**变量、且自建一张图，
    产不出骨架「§4.7 / 图 7+k」要的「{谱变量数}个变量{模型数}模型曲线」单张 PNG，
    所以这里自绘。观测谱只画一次——所有模型跑的是同一批共同日期、同一个
    ERA5 观测，各模型的 ``obs`` 列本就逐点相同。

    双对数谱必须保留全波数动态范围，不套用折线图的纵轴放大；可读性靠加大
    画布、加粗少许线宽解决（``图 8+k``/``图 9+k`` 的球谐带柱图另走
    ``wave_report._grouped_bars``）。

    字号**不按 16 in 画布放大**：这张图比别的图宽一倍多，按宽度算出来的
    字号在报告里会明显比邻居大。统一取 183 mm 基准档（``BASE_WIDTH``），
    图内文字与图 2–6 绝对字号一致。
    """
    apply_style()
    variables = [str(v) for v in variables]
    if not variables:
        raise ValueError("没有可画的谱变量")
    colors = model_colors([str(n) for n in names])

    ncols = max(1, min(2, len(variables)))
    nrows = int(math.ceil(len(variables) / ncols))
    width = snap_width(6.4 * ncols)
    sizes = font_sizes(BASE_WIDTH)
    figure, axes = plt.subplots(
        nrows, ncols, figsize=(width, 4.6 * nrows), squeeze=False
    )
    axes = axes.ravel()

    for index, variable in enumerate(variables):
        axis = axes[index]
        observed_drawn = False
        for name in names:
            frame = spectra.get(variable, {}).get(name)
            if frame is None or frame.empty:
                continue
            data = frame.copy()
            data.index = pd.to_numeric(data.index, errors="coerce")
            data = data[data.index.notna() & (data.index > 0)].sort_index()
            pred = pd.to_numeric(data["pred"], errors="coerce")
            axis.plot(pred.index, pred.values, color=colors[str(name)],
                      linewidth=1.3, label=str(name))
            if not observed_drawn:
                obs = pd.to_numeric(data["obs"], errors="coerce")
                axis.plot(obs.index, obs.values, color=NEUTRAL["observation"],
                          linestyle="--", linewidth=1.3, label="Observation")
                observed_drawn = True
        axis.set_xscale("log")
        axis.set_yscale("log")
        # 与其他图一致：面板标题 = 「(a) 变量名」，整行加粗。
        axis.set_title(
            f"({chr(ord('a') + index)}) {variable}",
            fontsize=sizes["title"], fontweight="bold",
        )
        axis.set_xlabel("Zonal wavenumber k", fontsize=sizes["label"])
        axis.set_ylabel("Power", fontsize=sizes["label"])
        axis.grid(which="both", linewidth=0.6)
        axis.tick_params(labelsize=sizes["tick"])
    for axis in axes[len(variables):]:
        axis.axis("off")
    # 图例放整图底部，不放进子图内部：谱曲线铺满整个双对数面板，`loc="best"`
    # 不管落在哪个角都会压住曲线（还会盖住刻度数字）。取第一个有线的子图的
    # 句柄即可——所有子图画的是同一批模型。
    handles: list = []
    labels: list = []
    for axis in axes[: len(variables)]:
        handles, labels = axis.get_legend_handles_labels()
        if handles:
            break
    if handles:
        legend = figure.legend(
            handles, labels, loc="lower center", ncols=min(len(labels), 7),
            fontsize=sizes["legend"],
            handlelength=1.8, columnspacing=1.4, labelspacing=0.45,
            borderaxespad=0.2, **LEGEND_FRAME,
        )
        legend.get_frame().set_linewidth(0.8)
        figure.tight_layout(rect=(0.0, 0.06, 1.0, 1.0))
    else:
        figure.tight_layout()
    return _save_figure(figure, Path(out_dir) / filename)


def _overall_heatmap(
    out_dir: Path,
    filename: str,
    bundle: "EnsBundle",
) -> Path:
    """总体指标热力图：行是模型、列是指标，格内数字是原始值。

    颜色是**列内**批内归一化的相对表现（``1 = 本批最优``、``0 = 最差``），
    深蓝 0 → 中灰 0.5 → 深红 1。各指标量纲不同，颜色**只在同一列内**可比，
    不能跨列读；这条必须写进图注，否则红蓝会被当成同一把尺子。

    行按综合相对 RMSE 从好到差排；ACC 越高越好，其余指标越低越好。球谐
    指标不可得时那两列整列不画（不是画成空白列）。
    """
    base = bundle.base
    names = sorted(bundle.names, key=lambda name: base.relative_composite[name])
    columns = [
        ("Rel. RMSE", base.relative_composite, False, 2),
        ("CRPS", bundle.crps_mean, False, 2),
        ("ACC", base.acc_mean, True, 2),
        ("Spread dev.", bundle.spread_log_deviation, False, 2),
        ("FA bias", base.fa_bias, False, 2),
        ("Spectrum dev.", bundle.spectrum_deviation_scalar, False, 2),
    ]
    if bundle.sph_available:
        columns += [
            ("Sph. RMS", bundle.sph_rms, False, 2),
            ("Sph. bias", bundle.sph_bias, False, 2),
        ]

    figure = plot_overall_heatmap(names, columns)
    return _save_figure(figure, Path(out_dir) / filename)


# ====================================================================== 渲染


def _named(names: Sequence[str]) -> str:
    return "、".join(str(n) for n in names)


# ------------------------------------------------------------------ 日期口径
# 报告有两种日期口径，取决于 ``analyze`` 的 ``own_dates``：
#   * 默认（共同日期）：所有模型的每一项统计都只在各模型**都**有数的那些日子上算。
#   * 各自的全部起报日：每个模型按自己的全部起报日算，天数可以互不相同，
#     只有 §6 的配对检验退回共同日期（日期不对齐做不了 Wilcoxon）。
# 下面这组函数是两种口径下**唯一**该写日期的地方；不要在别处手拼「共同日期」四个字，
# 否则 own_dates 那一档会出现「正文说 26 天、表里是 228 天」这种自相矛盾。





def _paired_frame(bundle: EnsBundle, frame: pd.DataFrame) -> pd.DataFrame:
    """把一个逐日帧裁到共同日期，供「显著优胜」那类配对判断使用。

    §4.1 / §4.2 的表里同时印**均值**和**显著优胜**：均值是各模型自己全部起报日上的
    （各自的天数），显著优胜必须来自共同日期上的配对检验——两者口径不同是有意的，
    表下会写明。各用各的时那个帧是并集 + NaN，直接送进 ``_wilcoxon_p`` 会让每一对
    各自在自己的交集上检验，``Holm`` 校正的对象就不是同一批假设了。
    """
    frame = frame.reindex(index=[str(d) for d in bundle.dates])
    return frame.dropna(how="any") if own_dates_mode(bundle) else frame


def _thin_models(bundle: EnsBundle) -> List[str]:
    """自己的起报日数低于经验下限的模型（按天数升序）。

    各用各的时，天少的那几个模型自己的平均值就不稳，这份名单要出现在风险一节；
    默认口径下全体共享共同日期，没有「谁比谁少」这回事，返回空表。
    """
    if not own_dates_mode(bundle):
        return []
    thin = [
        (len(bundle.dates_by_model[name]), name)
        for name in bundle.names
        if len(bundle.dates_by_model[name]) < SMALL_SAMPLE_DATES
    ]
    return [name for _count, name in sorted(thin)]


def _extreme(series: pd.Series, mode: str, digits: int = 3, suffix: str = "") -> str:
    """极值句里的「名字（数值）」——并列时逐个列出，与附表用同一套判据。

    ``mode="min"`` 配「最低／最小／最好」，``"max"`` 配「最高／最大／最差」；
    ``suffix`` 是单位（如 ACC 的 ``"%"``）。
    两个模型数值相等时 ``idxmin`` 只挑一个，正文会写出「最好 A，最差 A」
    这种自己跟自己比的话；这里按显示精度判并列，杜绝这种读法。
    """
    names = _best_names(series, digits, mode=mode)
    if not names:
        return "—"
    finite = series.dropna()
    value = finite.max() if mode == "max" else finite.min()
    return f"{_named(names)}（{fmt_number(value, digits)}{suffix}）"


def _lead_bands(leads: Sequence[float]) -> List[Tuple[str, List[float]]]:
    """按 ``LEAD_BANDS`` 切实际存在的 lead，段名用段内首末值现写（空段不输出）。"""
    finite = [float(x) for x in leads if math.isfinite(float(x))]
    bands: List[Tuple[str, List[float]]] = []
    for low, high in LEAD_BANDS:
        subset = [lead for lead in finite if low < lead <= high]
        if subset:
            bands.append((f"{min(subset):g}–{max(subset):g}h", subset))
    return bands


def _row_relative(values: Mapping[str, float]) -> pd.Series:
    """把一行（模型 → 值）按各模型几何均值重标成 100 基线。"""
    frame = pd.DataFrame({str(k): [v] for k, v in values.items()}, index=["row"])
    return _relative(frame).iloc[0]


def _band_values(frame: pd.DataFrame, subset: Sequence[float]) -> pd.DataFrame:
    """取 ``frame`` 中 index 落在 ``subset``（一组 lead）里的行。"""
    return frame.loc[frame.index.isin(list(subset))]


def _band_relative(base: Bundle, subset: Sequence[float], names: Sequence[str]) -> pd.Series:
    """一个时效段的综合相对 RMSE（以段内各模型的几何均值为 100）。

    §5 的表和它下面那段结论**必须**都走这个函数。曾经两处各算各的——表里重标成
    100 基线、结论里直接印原始值，同一段同一模型出现 97.242 与 97.518 两个数，
    而骨架那句「综合RMSE以{模型数}模型的几何均值为100」只对前者成立。
    """
    return _row_relative({
        name: float(_band_values(base.lead_relative, subset)[name].mean())
        for name in names
    })


def _failures_phrase(bundle: EnsBundle) -> str:
    broken = {
        name: len(bundle.archives[name].meta.get("failures") or [])
        for name in bundle.names
    }
    broken = {name: count for name, count in broken.items() if count}
    if not broken:
        return "所有模型的产物里每一行 status 都是 success，n_ok = n_dates，没有失败起报点"
    return "；".join(f"{name} 有 {count} 个起报点失败" for name, count in broken.items())


def _section_conclusion(
    bundle: EnsBundle, *, first: float, last: float, leads: int
) -> List[str]:
    base = bundle.base
    names = bundle.names
    items = [
        f"{cn_number(len(names))}个集合产物完整性检查通过；{_failures_phrase(bundle)}。",
        f"综合相对 RMSE 最好 {_extreme(base.relative_composite, 'min')}，"
        f"最差 {_extreme(base.relative_composite, 'max')}；"
        # 与 §4 表头同源：acc_mean 只是 acc_variable 一个变量，摘要里必须点名
        f"{base.acc_variable} ACC 最高 {_extreme(base.acc_mean, 'max', 2, suffix='%')}。",
        f"CRPS 最低 {_extreme(bundle.crps_mean, 'min')}，"
        f"最高 {_extreme(bundle.crps_mean, 'max')}；"
        f"离散度校准（平均 |ln(Spread/RMSE)|）最好 "
        f"{_extreme(bundle.spread_log_deviation, 'min', 4)}。",
        f"幅度真实性（全程 FA ratio 绝对偏差）最好 "
        f"{_extreme(base.fa_bias, 'min', 4, suffix=' pp')}，"
        f"最差 {_extreme(base.fa_bias, 'max', 4, suffix=' pp')}。",
        f"{SPECTRUM_DEVIATION_LABEL}（100×RMS[ln(pred/obs)]，"
        f"{len(base.spectrum_variables)}个谱变量平均）最小 "
        f"{_extreme(bundle.spectrum_deviation_scalar, 'min', 2)}，最大 "
        f"{_extreme(bundle.spectrum_deviation_scalar, 'max', 2)}。",
        f"集合配置：成员数 {bundle.member_label}；"
        f"逐日指标覆盖 RMSE（{len(base.common_variables)}个变量）、"
        f"CRPS（{_named(bundle.crps_variables)}）、"
        f"ACC（{_named(bundle.acc_variables)}）、"
        f"FA（{len(base.fa_variables)}个变量）与离散度"
        f"（{len(bundle.spread_variables)}个变量）。",
        f"异常变量检查：{base.anomaly_model} 的 {base.anomaly_variable} RMSE 为其余模型中位数的 "
        f"{fmt_number(base.anomaly_ratio, 2)} 倍"
        + ("，超出 3× 阈值，详见 4.1。" if base.anomaly_ratio >= 3.0 else "，未超 3× 阈值。"),
        f"选型取向：整体精度看综合相对 RMSE（{_named(_best_names(base.relative_composite))}），"
        f"集合可靠性看 CRPS（{_named(_best_names(bundle.crps_mean))}）与离散度"
        f"（{_named(_best_names(bundle.spread_log_deviation))}），"
        f"幅度与谱真实性看 FA 偏差（{_named(_best_names(base.fa_bias))}）与"
        f"{SPECTRUM_DEVIATION_LABEL}"
        f"（{_named(_best_names(bundle.spectrum_deviation_scalar))}），"
        f"几者不一定指向同一模型。",
    ]
    if bundle.sph_available:
        items.append(
            f"球谐带谱偏差（{SPHERICAL_RMS_LABEL}，"
            f"{SPHERICAL_BAND_COUNT}个总阶数频带）最小 "
            f"{_extreme(bundle.sph_rms, 'min', 2)}，最大 "
            f"{_extreme(bundle.sph_rms, 'max', 2)}；"
            f"它只看球谐带功率，与 §4.7 的纬向 FFT 谱不是同一分解。"
        )
    if bundle.small_sample:
        items.append(
            f"第 6 节的配对检验只有 {len(bundle.dates)} 个共同日期可用"
            f"（少于 {SMALL_SAMPLE_DATES} 天），Wilcoxon 检验的检验力很低，"
            f"「无显著差异」多半是样本不够、而不是两者真的等价；"
            f"本报告的排名只作方向性参考。"
        )
    if own_dates_mode(bundle):
        # 这是本报告最容易被读错的地方：表里的数来自不同时段，「谁更好」里有
        # 多少是时段差异、多少是模型差异，本报告分不开。必须写在摘要里。
        items.append(
            f"日期口径：正文与附录中**每个模型都用各自全部起报日**"
            f"（{date_scope_counts(bundle)}），表里的均值来自**不同时段**，"
            f"跨模型的相对位次因此同时包含模型差异与时段差异，"
            f"**不是同一批日期上的公平对比**；只有第 6 节的配对检验退回 "
            f"{len(bundle.dates)} 个共同日期（{date_span(bundle.dates)}）。"
        )
    return [
        "# 1. 结论摘要",
        "",
        # 先交代评估基本信息（评的是谁、多少天、什么指标），再给结论
        f"本报告评估{'、'.join(sorted(names))}共{len(names)}个模型，"
        + (
            f"覆盖{len(bundle.dates)}天（{date_span(bundle.dates)}）；"
            if not own_dates_mode(bundle)
            else f"覆盖时段不等长（{date_scope_counts(bundle)}）；"
        )
        + f"口径为集合平均场的RMSE、CRPS、ACC、FA与Spread/RMSE。"
        + (
            "正文与附录统一使用同一批共同日期，逐日结果按 lead 平均后比较，"
            "非共同日期不参与任何统计。"
            if not own_dates_mode(bundle)
            else "正文与附录中每个模型都用自己的全部起报日——各自的平均、各自的天数，"
            "因此各表各图并不是在同一批日期上对比（详见第 2 节与第 7 节）。"
        )
        + "确定性的单成员版本见 `weather_rmse_single` 报告，"
        + "其第 4.5 节是本报告球谐带的单成员对照；"
        + "本报告第 4.8 节的球谐带是集合平均场口径，"
        + "两份报告口径不同、数字不可互相搬运。",
        "",
        # 结论并成一段而不是逐条列点：读起来是判断，不是清单
        "".join(items),
        "",
        f"本报告中的FA指Forecast Activity，统一使用全程{first}–{last}h、"
        f"{leads}个lead，不对FA做短中长时效分段。",
        "",
    ]


def _section_samples(bundle: EnsBundle) -> List[str]:
    base = bundle.base
    names = bundle.names
    lines = ["# 2. 样本口径与数据质量", "", "**样本口径**", ""]

    rows = []
    for name in sorted(names):
        archive = base.archives[name]
        meta = archive.meta
        own = set(archive.summary["init_date"])
        common = own & set(base.dates)
        n_dates = meta.get("n_dates")
        n_ok = meta.get("n_ok")
        failed = len(meta.get("failures") or [])
        member = bundle.members.get(name)
        lead = bundle.leads.get(name)
        rows.append([
            name,
            str(n_dates) if n_dates is not None else "—",
            str(n_ok) if n_ok is not None else "—",
            str(len(archive.dates)),
            # 「本次使用日期」= 这个模型真正进了统计的天数。各用各的时它就等于
            # 「日期目录」，只用共同日期时它等于「有效共同日期」——两种口径下
            # 读者都能一眼看出谁被砍了多少，不用去猜渲染器用了哪一套。
            str(len(base.dates_by_model.get(name, []))),
            str(len(common)),
            str(member) if member is not None else "—",
            str(lead) if lead is not None else "—",
            f"{failed} 个起报点失败" if failed else "无",
        ])
    any_failed = sum(1 for name in names if (base.archives[name].meta.get("failures") or []))
    rows.append([
        "共同有效样本", "—", str(len(base.dates)), "—", "—", str(len(base.dates)),
        bundle.member_label,
        str(len(base.rmse_by_lead[names[0]].index)),
        f"{any_failed} 个模型有失败起报点" if any_failed else "无",
    ])
    lines += table_lines(
        ["**模型**", "**请求日期**", "**成功日期**", "**日期目录**",
         "**本次使用日期**", "**有效共同日期**", "**成员数**", "**Lead数**", "**失败情况**"],
        rows,
    )
    lines += [""]
    lines += bullet_lines([
        # 上半段交代「本次统计到底用了哪几天」，下半段解释「有效共同日期」那一列
        # 是给谁用的。两种口径下这句话必须能各说各话，所以整个按模式分叉。
        (
            f"正文与附录统一使用 {date_scope_detail(base)}；"
            f"各模型的非共同日期不参与任何统计与曲线，因此各表样本量一致、可直接横向比较。"
            if not own_dates_mode(bundle)
            # 这里用 _scope_counts 而不是 _scope_detail：句子前半已经说了「用自己的
            # 全部起报日」，再嵌一整句「各模型各自的全部起报日（…）」是重复。
            else f"正文与附录中每个模型都用自己的全部起报日（{date_scope_counts(base)}）。"
            f"表中「本次使用日期」即该模型真正进了统计的天数（= 它的「日期目录」），"
            f"**天数不同、时段也不同，各表之间的均值不可当作同一批日期的公平对比**；"
            f"「有效共同日期」那一列是第 6 节配对检验能用的天数，只有它才是各模型共有的。"
        ),
        f"「请求日期」「成功日期」取自各模型 ``scores.csv`` 里出现过的 ``init_date`` "
        f"其中全部行 ``status = success`` 的个数（新产物没有「请求日期」这个概念，"
        f"``n_dates`` 就是实际出了数的天数）。",
        f"离散度比值来源：{bundle.spread_ratio_source}。",
    ] + _quality_notes(bundle))
    return lines


def _quality_notes(bundle: EnsBundle) -> List[str]:
    """§2 的数据质量逐条说明。"""
    base = bundle.base
    notes: List[str] = []

    for name in sorted(bundle.names):
        meta = base.archives[name].meta
        n_dates = meta.get("n_dates")
        n_ok = meta.get("n_ok")
        if n_dates is None or n_ok is None:
            continue
        ratio = meta.get("missing_ratio")
        percent = fmt_number(BASELINE * float(ratio), 2) if ratio is not None else "—"
        tail = "，**已超过** batch 配置的缺失阈值" if meta.get("missing_threshold_exceeded") \
            else "，未超过缺失阈值"
        notes.append(
            f"{name}：请求 {n_dates} 个起报日期、成功 {n_ok} 个"
            f"（缺失率 {percent}%）{tail}；未成功的起报点不进入任何统计。"
        )

    declared = {
        name: (base.archives[name].meta.get("options") or {}).get("var_metrics") or {}
        for name in sorted(bundle.names)
    }
    if any(declared.values()):
        counts = "、".join(f"{name} {len(v)} 个" for name, v in declared.items())
        notes.append(
            f"变量覆盖按各模型配置里 ``var_metrics`` 的声明口径："
            f"{counts}。报告只比较各模型**都真的算出来**的变量"
            f"（RMSE {len(base.common_variables)} 个：{_named(base.common_variables)}；"
            f"ACC {len(bundle.acc_variables)} 个：{_named(bundle.acc_variables)}；"
            f"FA {len(base.fa_variables)} 个：{_named(base.fa_variables)}）。"
        )
        gaps = sorted({
            f"{name}.{variable}"
            for name, var_metrics in declared.items()
            for variable in var_metrics
            if variable not in base.common_variables
        })
        if gaps:
            notes.append(
                f"声明了 RMSE 但没进入本次共同比较的变量有 {len(gaps)} 个"
                f"（如 {'、'.join(gaps[:4])}）——这是各批 ``variables`` / ``var_metrics`` "
                f"配置不同导致的**口径差异**，不是数据缺陷；"
                f"要评这些变量得先把各批配置对齐后重跑。"
            )

    fa_gap = sorted(set(base.common_variables) - set(base.fa_variables))
    notes.append(
        f"指标列数在各模型间不完全一致也是同一原因：CRPS 只有 "
        f"{_named(bundle.crps_variables)}；ACC 只有 {_named(bundle.acc_variables)}"
        + (f"；FA 没有 {_named(fa_gap)}" if fa_gap else "")
        + f"——集合支线的 ``crps`` / ``fa`` 是逐变量配置决定算不算的。"
    )

    if len({m for m in bundle.members.values() if m is not None}) > 1:
        detail = "、".join(f"{name} {bundle.members[name]} 成员" for name in sorted(bundle.names))
        notes.append(
            f"各模型成员数不一致（{detail}）——CRPS 与离散度对成员数敏感，"
            f"成员数不同的模型之间比较这两个指标要格外小心。"
        )
    else:
        notes.append(
            f"各模型成员数一致（{bundle.member_label} 成员），"
            f"CRPS 与离散度在成员数口径上可直接横向比较。"
        )

    if bundle.small_sample:
        notes.append(
            f"配对检验可用的共同日期仅 {len(base.dates)} 天，低于 {SMALL_SAMPLE_DATES} "
            f"天的经验下限，第 6 节的配对检验不稳定，**不要**据此写结论。"
        )
    thin = _thin_models(bundle)
    if thin:
        # 各用各的时，「天数少的那几个模型自己的平均值不稳」和「配对检验样本不够」
        # 是两件独立的事，摘要里那句只说后者，这里补前者。
        detail = "、".join(
            f"{name} {len(bundle.dates_by_model[name])} 天" for name in thin
        )
        notes.append(
            f"**这些模型自己的起报日数低于 {SMALL_SAMPLE_DATES} 天的经验下限：{detail}**。"
            f"它们的每一个均值都只由这些天平均出来，与长时段模型的数值波动不在一个量级；"
            f"跨模型比大小前先看这一条。"
        )
    return notes


def _section_metrics(bundle: EnsBundle) -> List[str]:
    base = bundle.base
    first, last, count = lead_span(base.rmse_by_lead[bundle.names[0]].index)
    lines = ["# 3. 评估指标与解释", "", "**指标口径**", ""]

    rows = [
        ["ACC", "集合平均场与观测的距平相关系数，逐日先按 lead 平均再跨日平均（百分数）",
         "越大越好"],
        ["CRPS", "连续分级概率评分，集合分布与观测的差距，越低越好", "越小越好"],
        ["FA ratio 偏差",
         f"Forecast Activity 的 pred/obs 比值，全程 {first}–{last}h 平均后取 |ratio−1|×100（pp）",
         "越小越好"],
        ["RMSE", "集合平均场与观测的均方根误差", "越小越好"],
        ["Spread/RMSE", "集合成员标准差 ÷ 集合平均场 RMSE，理想值为 1", "越接近 1 越好"],
        ["Spread偏差(绝对对数)",
         "跨变量与 lead 的 mean|ln(Spread/RMSE)|，与「距1」是同一直觉的对数口径",
         "越小越好"],
        ["综合相对RMSE",
         f"各变量 RMSE 相对{len(bundle.names)}模型几何均值的 100 倍，再跨变量等权平均",
         "越小越好"],
        [SPECTRUM_DEVIATION_LABEL,
         "100 × 各谱变量 RMS_k[ln(pred/obs)] 的平均，k 为纬向波数 1–720", "越小越好"],
        [SPHERICAL_RMS_LABEL,
         f"{SPHERICAL_BAND_COUNT}个球谐总阶数频带上计算 RMS[ln(pred/obs)]×100，"
         "是所有日期、变量、lead 的总均方根",
         "越小越好" if bundle.sph_available else "越小越好；本批未出"],
        [SPHERICAL_BIAS_LABEL,
         "所有频带、变量、lead 与日期上的 mean |pred/obs−1|，单位 %",
         "越小越好" if bundle.sph_available else "越小越好；本批未出"],
    ]
    rows.sort(key=lambda row: row[0])
    lines += table_lines(["**指标**", "**定义/口径**", "**方向**"], rows)

    lines += [
        "",
        f"报告类型为集合预报（ensemble，{bundle.member_label} 成员）。"
        f"RMSE、ACC、FA 一律是**集合平均场**口径（逐日 ``*_ensmean.csv``），不是单个成员；"
        f"CRPS 与离散度用的是整个成员集合；两者不可混读。",
        "",
        f"所有对数一律取**自然对数 ln**。本报告的「{SPECTRUM_DEVIATION_LABEL}」与 "
        f"``weather_rmse_single`` 报告里的「频谱对数 RMS」**不同源**"
        f"（那边是 log10，数值相差 2.3026 倍），跨报告比大小前先统一底数。",
        "",
        f"逐日指标覆盖 {count} 个 lead（{first}–{last}h）；FA 只做全程统计、不分时效段"
        f"（避免与降水二分类的 False Alarm 混淆），"
        f"RMSE、CRPS、ACC 与 Spread/RMSE 参与分时效分析。",
        "",
        f"显著性一律用共同日期（{len(base.dates)} 天）上的逐日配对 Wilcoxon 符号秩检验，"
        f"多重比较用 Holm 逐步校正，Holm p < 0.05 才记「显著」。"
        f"「显著优胜」指该模型在同一项上显著优于其余**全部**模型。",
    ]
    return lines


def _rmse_notes(bundle: EnsBundle) -> List[str]:
    """§4.1 的逐变量读数：极值、极差、跨变量优胜是否一致。"""
    base = bundle.base
    items: List[str] = []
    best_sets: Dict[str, set] = {}
    for variable in base.common_variables:
        values = base.per_variable_rmse[variable].mean(axis=0).dropna()
        if values.empty:
            continue
        items.append(
            f"{variable}：RMSE 最低 {_extreme(values, 'min', 4)}，"
            f"最高 {_extreme(values, 'max', 4)}，"
            f"极差 {fmt_number(values.max() - values.min(), 4)}。"
        )
        best_sets[variable] = set(_best_names(values))
    if len(best_sets) > 1:
        shared = set.intersection(*best_sets.values())
        if shared:
            items.append(
                f"{_named(sorted(shared))} 在全部 {len(best_sets)} 个 RMSE 变量上都是最低，"
                f"RMSE 侧的排序与看哪个变量无关。"
            )
        else:
            detail = "；".join(
                f"{variable} 是 {_named(sorted(winners))}"
                for variable, winners in sorted(best_sets.items())
            )
            items.append(
                f"RMSE 侧的排序随变量变：{detail}——某个变量上误差最低不代表整批变量领先。"
            )
    if own_dates_mode(bundle):
        sample_note = (
            "上表均值基于各模型自己的全部起报日，「显著优胜」列用共同日期上的逐日配对检验，"
            "两者不是同一批样本。"
        )
    else:
        sample_note = (
            "上表均值与「显著优胜」列都用同一批共同日期，样本口径一致。"
        )
    items.append(
        "RMSE 为**集合平均场**口径，不是逐成员先算再平均；" + sample_note
    )
    return items


def _crps_notes(bundle: EnsBundle) -> List[str]:
    """§4.2 的逐变量读数。CRPS 是集合分布口径，不能与 RMSE 互读。"""
    items: List[str] = []
    best_sets: Dict[str, set] = {}
    for variable in bundle.crps_variables:
        frame = pd.DataFrame(
            {name: bundle.crps_per_date[name][variable] for name in bundle.names}
        )
        values = frame.mean(axis=0).dropna()
        if values.empty:
            continue
        items.append(
            f"{variable}：CRPS 最低 {_extreme(values, 'min', 4)}，"
            f"最高 {_extreme(values, 'max', 4)}，"
            f"极差 {fmt_number(values.max() - values.min(), 4)}。"
        )
        best_sets[variable] = set(_best_names(values))
    if len(best_sets) > 1:
        shared = set.intersection(*best_sets.values())
        if shared:
            items.append(
                f"{_named(sorted(shared))} 在全部 {len(best_sets)} 个 CRPS 变量上都是最低，"
                f"CRPS 侧的排序与看哪个变量无关。"
            )
        else:
            detail = "；".join(
                f"{variable} 是 {_named(sorted(winners))}"
                for variable, winners in sorted(best_sets.items())
            )
            items.append(
                f"CRPS 侧的排序随变量变：{detail}。"
            )
    items.append(
        "CRPS 同时惩罚平均场偏差与离散度不足，是集合分布质量的指标；"
        "它与 RMSE 的排序不必一致，选型时要说明以哪一个为准。"
    )
    return items


def _fa_notes(bundle: EnsBundle) -> List[str]:
    """§4.5 的逐变量读数。FA 只衡量活动幅度，不衡量位置或形态。"""
    base = bundle.base
    items: List[str] = []
    by_variable: Dict[str, pd.Series] = {}
    for variable in base.fa_variables:
        values = pd.Series({
            name: BASELINE * float(
                (base.fa_ratio_by_lead[name][variable] - 1.0).abs().mean()
            )
            for name in bundle.names
        }).dropna()
        if values.empty:
            continue
        by_variable[variable] = values
        items.append(
            f"{variable}：FA ratio 绝对偏差最小 {_extreme(values, 'min', 4, suffix=' pp')}，"
            f"最大 {_extreme(values, 'max', 4, suffix=' pp')}，"
            f"极差 {fmt_number(values.max() - values.min(), 4)} pp。"
        )
    if len(by_variable) > 1:
        items.append(
            "各变量上的优胜者见上表最后一列；FA 的排序通常比 RMSE/ACC 更依赖变量本身的活动幅度，"
            "不能拿一个变量的名次概括全部变量。"
        )
    items.append(
        "FA（Forecast Activity）只反映预测场活动幅度相对观测的高低，"
        "ratio 越接近 1 越好；它不衡量位置、形态或相位误差，"
        "与 RMSE、ACC 的好坏可以完全不同向。"
    )
    return items


def _spectrum_notes(bundle: EnsBundle) -> List[str]:
    """§4.7 的逐变量读数。纬向谱偏差是 ln 口径，别和 single 的 log10 比。"""
    items: List[str] = []
    best_sets: Dict[str, set] = {}
    for variable in bundle.spectrum_variables:
        values = bundle.spectrum_deviation.loc[variable].dropna()
        if values.empty:
            continue
        items.append(
            f"{variable}：纬向谱偏差最小 {_extreme(values, 'min', 2)}，"
            f"最大 {_extreme(values, 'max', 2)}，"
            f"极差 {fmt_number(values.max() - values.min(), 2)}。"
        )
        best_sets[variable] = set(_best_names(values))
    if len(best_sets) > 1:
        shared = set.intersection(*best_sets.values())
        if shared:
            items.append(
                f"{_named(sorted(shared))} 在全部 {len(best_sets)} 个谱变量上都是最小。"
            )
        else:
            detail = "；".join(
                f"{variable} 是 {_named(sorted(winners))}"
                for variable, winners in sorted(best_sets.items())
            )
            items.append(f"谱偏差的排序随变量变：{detail}。")
    items.append(
        "纬向谱偏差定义为 100×RMS[ln(pred/obs)]，按纬向 FFT 波数 1–720 汇总；"
        "它衡量各尺度的幅度结构，不看相位。所有对数都是自然对数 ln，"
        "与 `weather_rmse_single` 报告里的 log10 频谱不同源，数值不可直接比。"
    )
    return items


def _spherical_band_notes(bundle: EnsBundle) -> List[str]:
    """§4.8.1 的分频带读数。"""
    band_list = list(bundle.sph_band_logrms.index)
    if not band_list:
        return []
    # 整条频带可能一格数都没有（probe_bands 只要求「至少一个变量有数」），
    # 直接 idxmin 会在全 NaN 的行上退化；先 dropna，再退回「—」。
    best = {}
    for band in band_list:
        finite = bundle.sph_band_logrms.loc[band].dropna()
        best[band] = finite.idxmin() if not finite.empty else "—"
    items = [
        "球谐带 log-RMS×100 最低的模型在"
        + "、".join(f"{band} 段是 {best[band]}" for band in band_list)
        + "；均值比是该频带内 pred/obs 的总平均，越接近 1 表示该尺度上的能量越不偏。",
        f"最高频带（{band_list[-1]}）的总阶数最高、观测功率最小，"
        "比值的对数对微小差异最敏感，该频带的 log-RMS 绝对值不宜与低频带直接比较——"
        "这是频谱类指标的固有特征，不是模型在该尺度上一定更差；"
        "均值比在同一频带仍有意义，因为它不依赖逐点的分母。",
        "按频带看，log-RMS 最低的模型分别是 "
        + "、".join(
            f"{name} 拿下 "
            f"{sum(1 for band in band_list if best[band] == name)}"
            f" 个频带"
            for name in bundle.names
        )
        + "。",
    ]
    return items


def _spherical_variable_notes(bundle: EnsBundle) -> List[str]:
    """§4.8.2 的分变量读数与排序一致性。"""
    variables = sorted(bundle.sph_variable.index)
    if not variables:
        return []
    by_variable = bundle.sph_variable.mean(axis=0)
    by_band = bundle.sph_band_logrms.mean(axis=0)
    order_variable = by_variable.sort_values().index.tolist()
    order_band = by_band.sort_values().index.tolist()
    return [
        "逐变量的球谐带 log-RMS×100 排序为 "
        + " < ".join(f"{n}（{fmt_number(by_variable[n], 2)}）" for n in order_variable)
        + "，按 §4.8.1 的频带口径汇总为 "
        + " < ".join(f"{n}（{fmt_number(by_band[n], 2)}）" for n in order_band)
        + ("，两者一致。" if order_variable == order_band else "，**两者不一致**。")
        + "两者用的是同一批带功率样本，只是聚合维度不同（这里按变量汇总全部频带，"
        "§4.8.1 按频带汇总全部变量），排序不一致时说明模型间的优劣集中在个别频带上，"
        "选型时要指明关注哪一段尺度。"
    ]


def _spherical_test_notes(bundle: EnsBundle) -> List[str]:
    """§4.8.3 的配对检验读数。"""
    band_list = list(bundle.sph_band_logrms.index)
    rows = sum(len(bundle.sph_tests.get(band, [])) for band in band_list)
    significant = sum(
        1
        for band in band_list
        for test in bundle.sph_tests.get(band, [])
        if test.significant
    )
    pair_count = len(bundle.names) * (len(bundle.names) - 1) // 2
    if significant:
        tail = (
            f"全部 {rows} 个「频带 × 模型对」组合里有 {significant} 个达到显著，"
            "其余组合的差异被样本量或频带内的离散度淹没；"
        )
    else:
        tail = (
            f"全部 {rows} 个「频带 × 模型对」组合均未达显著，"
            "说明各模型在这些总阶数频带上的差异小于日度波动；"
        )
    return [
        f"逐频带在 {len(bundle.dates)} 个日度样本上做 Wilcoxon 符号秩检验，"
        f"每个频带各自对 {pair_count} 个模型对做 Holm 校正"
        "（指标是频带本身，同一频带内的模型对构成一个检验族）。"
        + tail
        + "统计显著不代表业务重要性。"
    ]


def _section_overall(
    bundle: EnsBundle,
    figures: Mapping[str, str],
    *,
    spatial: Optional[SpatialPayload] = None,
    figure_offset: int = 0,
) -> List[str]:
    """§4 全部小节。``figure_offset`` = 空间分布图的张数 k，其后图号整体后移。"""
    base = bundle.base
    names = bundle.names
    order = base.relative_composite.sort_values().index.tolist()
    rank = base.relative_composite.rank(method="min")
    first, last, leads = lead_span(base.rmse_by_lead[names[0]].index)
    shown = _named(names)

    lines = ["# 4. 总体评估结果", "", f"**{cn_number(len(names))}模型总体指标**", ""]
    rows = []
    for name in order:
        rows.append([
            f"{name}（第{int(rank[name])}）",
            fmt_number(base.relative_composite[name]),
            fmt_number(bundle.crps_mean[name]),
            fmt_number(base.acc_mean[name]),
            fmt_number(bundle.spread_log_deviation[name], 4),
            fmt_number(base.fa_bias[name], 4),
            fmt_number(bundle.spectrum_deviation_scalar[name], 2),
            fmt_number(bundle.sph_rms[name], 2) if bundle.sph_available else "—",
            fmt_number(bundle.sph_bias[name], 2) if bundle.sph_available else "—",
        ])
    lines += table_lines(
        # ACC 列**只代表 acc_variable 一个变量**（base.acc_mean 就是它），
        # 所以列名必须带上变量名：分变量的 ACC 在 §4.3，不能拿这一列概括全部变量。
        ["**模型（相对排序）**", "**综合相对RMSE**", "**CRPS**",
         f"**ACC（{base.acc_variable}）**",
         "**Spread偏差(绝对对数)**", "**FA偏差(全程)**", "**纬向谱偏差**",
         f"**{SPHERICAL_RMS_LABEL}**", f"**{SPHERICAL_BIAS_LABEL}**"],
        rows,
    )

    lines += [
        "",
        f"![overall_heatmap]({figures['overall_heatmap']})",
        "",
        f"图 1：{cn_number(len(names))}模型总体指标热力图。列为指标（ACC 越高越好，"
        "其余越低越好），行按综合相对 RMSE 从好到差排列，格内数字为原始值"
        "（统一保留两位小数）。**颜色按列单独做 Min-Max 归一化**：颜色深浅"
        "代表该指标在横向对比中的相对优劣（1 = 本批最优，0 = 最差），具体"
        "数值见格内；各列量纲不同，颜色只在同一列内可比，不能跨列比较。",
        "",
    ]

    lines += [""]
    lines += bullet_lines(_overall_notes(bundle))
    lines += [
        "",
        f"以下各节按指标拆开：每节先给分变量表，再给随 lead 的曲线，最后是判读。"
        f"表中均值基于{date_scope_word(bundle)}；「显著优胜」列来自共同日期"
        f"（{len(base.dates)} 天）上的逐日配对 Wilcoxon 检验。",
        "",
    ]

    # ---------------------------------------------------------------- 4.1 RMSE
    lines += ["## 4.1 集合均值RMSE", "", "**分变量集合均值RMSE**", ""]
    rmse_rows = []
    for variable in base.common_variables:
        frame = base.per_variable_rmse[variable]
        rmse_rows.append(
            [variable]
            + [fmt_number(frame[name].mean(), 4) for name in names]
            # 均值取上表那个帧（各自全部起报日），显著优胜取共同日期上的配对检验
            + [_dominant(_paired_frame(bundle, frame), names)]
        )
    lines += table_lines(["**变量**"] + list(names) + ["**显著优胜**"], rmse_rows)
    lines += [
        "",
        f"![ensemble_rmse_vs_lead]({figures['ensemble_rmse_vs_lead']})",
        "",
        f"图 2：{len(base.common_variables)}个变量上{shown}的集合均值 RMSE 随 lead 变化"
        f"（{date_scope_word(bundle)}平均）。纵轴按各变量的数据范围缩放，便于分辨模型间差异；"
        "绝对数值见上表。",
        "",
    ]
    lines += bullet_lines(_rmse_notes(bundle))

    # ---------------------------------------------------------------- 4.2 CRPS
    lines += ["", "## 4.2 CRPS", "", "**分变量平均CRPS**", ""]
    crps_rows = []
    for variable in bundle.crps_variables:
        frame = pd.DataFrame(
            {name: bundle.crps_per_date[name][variable] for name in names}
        )
        crps_rows.append(
            [variable]
            + [fmt_number(frame[name].mean(), 4) for name in names]
            + [_dominant(_paired_frame(bundle, frame), names)]
        )
    lines += table_lines(["**变量**"] + list(names) + ["**显著优胜**"], crps_rows)
    lines += [
        "",
        f"![ensemble_crps_vs_lead]({figures['ensemble_crps_vs_lead']})",
        "",
        f"图 3：{_named(bundle.crps_variables)} 的 CRPS 随 lead 变化"
        f"（{date_scope_word(bundle)}平均）。纵轴按数据范围缩放，越低越好。",
        "",
    ]
    lines += bullet_lines(_crps_notes(bundle))

    # ---------------------------------------------------------------- 4.3 ACC
    if bundle.acc_variables:
        lines += ["", "## 4.3 ACC", "", "**分变量平均ACC（%）**", ""]
        acc_rows = []
        for variable in bundle.acc_variables:
            frame = pd.DataFrame(
                {name: bundle.acc_per_date[name][variable] for name in names}
            )
            acc_rows.append(
                [variable]
                + [fmt_number(frame[name].mean(), 2) for name in names]
                # 同 §4.1：均值取各自全部起报日，显著优胜取共同日期
                + [_dominant(_paired_frame(bundle, frame), names, higher_is_better=True)]
            )
        lines += table_lines(["**变量**"] + list(names) + ["**显著优胜**"], acc_rows)
        lines += [
            "",
            f"![ensemble_acc_vs_lead]({figures['ensemble_acc_vs_lead']})",
            "",
            f"图 4：{_named(bundle.acc_variables)} 的 ACC 随 lead 变化"
            f"（{date_scope_word(bundle)}平均）。纵轴按数据范围缩放，越高越好。",
            "",
        ]
        lines += bullet_lines(_acc_notes(bundle))

    # ---------------------------------------------------------------- 4.4 Spread
    lines += ["", "## 4.4 离散度校准", "", "**分变量平均 Spread/RMSE 与距 1 偏差**", ""]
    spread_rows = []
    for variable in bundle.spread_variables:
        ratio = pd.DataFrame(
            {name: bundle.spread_ratio_by_lead[name][variable] for name in names}
        )
        mean = ratio.mean(axis=0)
        linear = (ratio - 1.0).abs().mean(axis=0)
        spread_rows.append(
            [variable]
            + [f"{name} {fmt_number(mean[name], 4)}" for name in names]
            + [f"{name} {fmt_number(linear[name], 4)}" for name in names]
            + [_best_cell(linear)]
        )
    lines += table_lines(
        ["**变量**"]
        + [f"**{name} 平均Spread/RMSE**" for name in names]
        + [f"**{name} 距1**" for name in names]
        + ["**更接近1**"],
        spread_rows,
    )
    lines += [
        "",
        f"![ensemble_spread_rmse_ratio_vs_lead]"
        f"({figures['ensemble_spread_rmse_ratio_vs_lead']})",
        "",
        f"图 5：{len(bundle.spread_variables)}个变量的 Spread / Ensemble-mean RMSE "
        f"随 lead 变化（{date_scope_word(bundle)}平均），虚线为理想值 1；"
        "纵轴按数据范围缩放。",
        "",
    ]
    lines += bullet_lines(_spread_notes(bundle))

    # ---------------------------------------------------------------- 4.5 FA
    lines += ["", "## 4.5 全程FA ratio", ""]
    lines.append(
        f"FA仅指Forecast Activity，全程{first}–{last}h统计，不进行时效分段，"
        f"以避免与降水False Alarm混淆。"
    )
    lines += ["", "**全程FA ratio绝对偏差**", ""]
    fa_rows = []
    for variable in base.fa_variables:
        values = {
            name: BASELINE * float((base.fa_ratio_by_lead[name][variable] - 1.0).abs().mean())
            for name in names
        }
        fa_rows.append(
            [variable]
            + [f"{name} {fmt_number(values[name], 4)}" for name in names]
            + [_best_cell(pd.Series(values))]
        )
    lines += table_lines(
        ["**变量**"]
        + [f"**{name} 平均绝对ratio偏差**" for name in names]
        + ["**更接近1**"],
        fa_rows,
    )
    lines += [
        "",
        f"![ensemble_fa_ratio_vs_lead]({figures['ensemble_fa_ratio_vs_lead']})",
        "",
        f"图 6：{_named(base.fa_variables)} 的 FA（Forecast Activity）ratio 随 lead 变化，"
        f"覆盖全程 {first}–{last}h 的 {leads} 个 lead，虚线为理想值 1；"
        "纵轴按数据范围缩放。",
        "",
    ]
    lines += bullet_lines(_fa_notes(bundle))

    # ---------------------------------------------------------------- 4.6 空间分布
    lines += [""]
    lines += build_spatial_section(
        spatial, heading="## 4.6 空间分布图", figure_start=7
    )

    # ---------------------------------------------------------------- 4.7 纬向谱
    lines += ["", "## 4.7 纬向功率谱", "", "**分变量纬向谱绝对对数偏差**", ""]
    spectrum_rows = []
    for variable in base.spectrum_variables:
        row = bundle.spectrum_deviation.loc[variable]
        spectrum_rows.append(
            [variable]
            + [f"{name} {fmt_number(row[name], 2)}" for name in names]
            + [_best_cell(row)]
        )
    lines += table_lines(
        ["**谱变量**"]
        + [f"**{name} 纬向谱绝对对数偏差**" for name in names]
        + ["**更优**"],
        spectrum_rows,
    )
    lines += [
        "",
        f"![ensemble_spectrum]({figures['ensemble_spectrum']})",
        "",
        f"图 {7 + figure_offset}：{_named(base.spectrum_variables)} 的纬向功率谱"
        f"（双对数，{date_scope_word(bundle)}平均），每条颜色曲线为一个模型的预测谱，"
        "黑色虚线为观测谱。",
        "",
    ]
    lines += bullet_lines(_spectrum_notes(bundle))

    # ---------------------------------------------------------------- 4.8 球谐带
    lines += ["", "## 4.8 球谐带功率", ""]
    if not bundle.sph_available:
        lines += [
            f"{SPHERICAL_ABSENT}因此本节无表、无图，保留节位以便将来补数后直接填入。",
            "",
        ]
        return lines

    band_names = list(bundle.sph_band_logrms.index)
    lines += [
        f"球谐带功率把集合平均场分解到 {SPHERICAL_BAND_COUNT} 个球谐总阶数频带；"
        "log-RMS×100 越小表示该频带能量越接近观测，均值比越接近 1 越好。"
        "它与 §4.7 的纬向 FFT 谱不是同一分解，不能互相替代。",
        "",
        "### 4.8.1 分频带结果",
        "",
        "**分频带球谐带 log-RMS×100**",
        "",
    ]
    band_rows = []
    for band in band_names:
        values = bundle.sph_band_logrms.loc[band]
        finite = values.dropna()
        band_rows.append(
            [band]
            + [fmt_number(values[name], 2) for name in names]
            + [finite.idxmin() if not finite.empty else "—"]
        )
    lines += table_lines(
        ["**频带**"] + [f"**{name}**" for name in names] + ["**最佳**"],
        band_rows,
    )

    lines += ["", "**分频带平均 pred/obs 均值比**", ""]
    ratio_rows = []
    for band in band_names:
        values = bundle.sph_band_ratio.loc[band]
        distance = (values - 1.0).abs().dropna()
        ratio_rows.append(
            [band]
            + [fmt_number(values[name], 4) for name in names]
            + [distance.idxmin() if not distance.empty else "—"]
        )
    lines += table_lines(
        ["**频带**"] + [f"**{name}**" for name in names] + ["**更接近1**"],
        ratio_rows,
    )
    lines += [
        "",
        f"![spherical_bands_logrms]({figures['spherical_bands_logrms']})",
        "",
        f"图 {8 + figure_offset}：{cn_number(len(names))}模型分球谐总阶数频带"
        f"（{' / '.join(band_names)}）的球谐带 log-RMS×100 对比。",
        "",
        f"![spherical_bands_ratio]({figures['spherical_bands_ratio']})",
        "",
        f"图 {9 + figure_offset}：{cn_number(len(names))}模型分球谐总阶数频带的 pred/obs 均值比对比"
        "（虚线为理想值 1）。",
        "",
    ]
    lines += bullet_lines(_spherical_band_notes(bundle))

    lines += ["", "### 4.8.2 分变量结果", "", "**分变量球谐带 log-RMS×100**", ""]
    variable_rows = []
    for variable in sorted(bundle.sph_variable.index):
        values = bundle.sph_variable.loc[variable]
        finite = values.dropna()
        variable_rows.append(
            [variable]
            + [fmt_number(values[name], 2) for name in names]
            + [finite.idxmin() if not finite.empty else "—"]
        )
    lines += table_lines(
        ["**变量**"] + [f"**{name}**" for name in names] + ["**最佳**"],
        variable_rows,
    )
    lines += [""]
    lines += bullet_lines(_spherical_variable_notes(bundle))
    coverage_notes = sph_coverage_notes(bundle)
    if coverage_notes:
        lines += [""]
        lines += bullet_lines(coverage_notes)

    lines += ["", "### 4.8.3 配对检验", ""]
    test_rows = []
    for band in band_names:
        for test in sorted(bundle.sph_tests.get(band, []), key=lambda t: (t.a, t.b)):
            test_rows.append([
                band,
                f"{test.a} vs {test.b}",
                _signed(test.delta_pct),
                fmt_number(test.p_holm, 4),
                "是" if test.significant else "否",
                test.better,
            ])
    lines += table_lines(
        ["**指标**", "**模型对**", "**A相对B**", "**Holm p**", "**显著**", "**胜者**"],
        test_rows,
    )
    lines += [""]
    lines += bullet_lines(_spherical_test_notes(bundle))
    return lines


def _overall_notes(bundle: EnsBundle) -> List[str]:
    base = bundle.base
    items = [
        f"综合相对 RMSE 以{cn_number(len(bundle.names))}模型的几何均值为 100："
        f"最低 {_extreme(base.relative_composite, 'min')}，"
        f"最高 {_extreme(base.relative_composite, 'max')}，"
        f"两者相差 {fmt_number(base.relative_composite.max() - base.relative_composite.min())}。",
        f"CRPS 最低 {_extreme(bundle.crps_mean, 'min')}，"
        f"最高 {_extreme(bundle.crps_mean, 'max')}；"
        f"CRPS 同时惩罚偏差与离散度不足，是与 RMSE 相互印证的一条独立证据。",
        # 这里用的是单变量 acc_mean（= acc_variable），必须点名是哪个变量；
        # 分变量的 ACC 在 §4.3，别把这一条读成全变量结论。
        f"{base.acc_variable} ACC 最高 {_extreme(base.acc_mean, 'max', 2, suffix='%')}，"
        f"最低 {_extreme(base.acc_mean, 'min', 2, suffix='%')}。",
        f"离散度平均绝对对数偏差最小 {_extreme(bundle.spread_log_deviation, 'min', 4)}，"
        f"最大 {_extreme(bundle.spread_log_deviation, 'max', 4)}；"
        f"越接近 0 表示集合离散度与自身误差越匹配。",
        f"FA ratio 绝对偏差最小 {_extreme(base.fa_bias, 'min', 4, suffix=' pp')}，"
        f"最大 {_extreme(base.fa_bias, 'max', 4, suffix=' pp')}；"
        f"FA 只反映活动幅度是否被系统性高估或低估，与 RMSE 好坏无关。",
        f"{SPECTRUM_DEVIATION_LABEL}最小 {_extreme(bundle.spectrum_deviation_scalar, 'min', 2)}，"
        f"最大 {_extreme(bundle.spectrum_deviation_scalar, 'max', 2)}；"
        f"数值越大表示预测谱的幅度结构偏离观测谱越多。",
    ]
    if bundle.sph_available:
        items.append(
            f"{SPHERICAL_RMS_LABEL}最小 "
            f"{_extreme(bundle.sph_rms, 'min', 2)}，最大 "
            f"{_extreme(bundle.sph_rms, 'max', 2)}；"
            f"{SPHERICAL_BIAS_LABEL}最小 "
            f"{_extreme(bundle.sph_bias, 'min', 2, suffix='%')}，最大 "
            f"{_extreme(bundle.sph_bias, 'max', 2, suffix='%')}。"
        )
    # 并列时按**集合**比，不按 idxmin 挑中的那一个：RMSE 并列冠军里只要有一个
    # 不是 CRPS 冠军，就仍然构成「两个口径指向不同模型」，值得写出来。
    rmse_best = sorted(_best_names(base.relative_composite))
    crps_best = sorted(_best_names(bundle.crps_mean))
    if not set(rmse_best) <= set(crps_best):
        items.append(
            f"口径分歧：综合相对 RMSE 最好的是 {_named(rmse_best)}，而 CRPS 最低的是 "
            f"{_named(crps_best)}——集合平均场精度与集合分布质量不是同一件事，"
            f"选型时要说明以哪一个为准。"
        )
    if own_dates_mode(bundle):
        items.append(
            f"口径提醒：本节各列是**各模型自己全部起报日**上的平均"
            f"（{date_scope_counts(bundle)}），不是同一批日期；"
            f"下表的相对位次里混着时段差异，第 6 节才是在共同日期上的配对检验。"
        )
    if bundle.small_sample:
        items.append(
            f"配对检验可用的共同日期只有 {len(bundle.dates)} 天，"
            f"上述排序的方向性高于显著性，不要当成定论。"
        )
    return items


def _acc_notes(bundle: EnsBundle) -> List[str]:
    """§4.3 分变量 ACC 的判读句。

    ACC 是**逐变量**的：``base.acc_mean`` 只代表 ``acc_variable`` 一个变量的
    逐日平均，§4 表头那一列就是它。只写表头会把「某个变量上 ACC 高」读成
    「所有变量都好」，所以这里逐变量摊开最高/最低/极差，并交代各 ACC 变量的
    优胜者是否一致，以及本批 ACC 到底覆盖了哪些变量。
    """
    items: List[str] = []
    best_sets: Dict[str, set] = {}
    for variable in bundle.acc_variables:
        mean = pd.DataFrame(
            {name: bundle.acc_per_date[name][variable] for name in bundle.names}
        ).mean(axis=0)
        finite = mean.dropna()
        if finite.empty:
            continue
        items.append(
            f"{variable}：ACC 最高 {_extreme(finite, 'max', 2, suffix='%')}，"
            f"最低 {_extreme(finite, 'min', 2, suffix='%')}，"
            f"极差 {fmt_number(finite.max() - finite.min(), 2)} pp。"
        )
        best_sets[variable] = set(_best_names(finite, 2, mode="max"))

    if len(best_sets) > 1:
        shared = set.intersection(*best_sets.values())
        if shared:
            items.append(
                f"{_named(sorted(shared))} 在全部 {len(best_sets)} 个 ACC 变量上都是最高，"
                f"ACC 侧的排序与看哪个变量无关。"
            )
        else:
            detail = "；".join(
                f"{variable} 是 {_named(sorted(winners))}"
                for variable, winners in sorted(best_sets.items())
            )
            items.append(
                f"ACC 侧的排序随变量变：{detail}——某个变量上 ACC 领先不代表整批变量领先。"
            )

    items.append(
        f"本批次 ACC 只覆盖 {_named(bundle.acc_variables)} 共 "
        f"{len(bundle.acc_variables)} 个变量，§4 表头的 ACC 列是其中 "
        f"{bundle.base.acc_variable} 一个变量；其余变量的准确度以 §4.1 的分变量 RMSE 为准。"
    )
    return items


def _spread_notes(bundle: EnsBundle) -> List[str]:
    items = []
    for name in bundle.names:
        ratio = bundle.spread_ratio_by_lead[name]
        values = pd.to_numeric(ratio.stack(), errors="coerce").dropna()
        if values.empty:
            continue
        under = BASELINE * float((values < 1.0).mean())
        median = float(values.median())
        # 这句判语必须跟着数字走：离散度偏小是「过于自信」，偏大是「过于保守」。
        # 写死成前者时，中位数 1.32、0% 低于 1 的模型也会被说成「过于自信」。
        verdict = "离散度偏小、集合过于自信" if median < 1.0 else "离散度偏大、集合偏保守"
        if 1.0 < under < 99.0:
            # 中位数只说多数在哪一侧；两侧都有样本时不能读成单向系统性偏差。
            verdict += "，但两侧都有样本，不是单向系统性偏差"
        items.append(
            f"{name}：Spread/RMSE 跨变量 × lead 的中位数为 {fmt_number(median, 4)}，"
            f"{fmt_number(under, 2)}% 的样本低于 1（{verdict}）。"
        )
    items.append(
        "「平均Spread/RMSE」是逐 lead 比值先跨日期平均、再在 lead 上平均；"
        "「距1」是这条逐 lead 平均曲线在每个 lead 上离 1 的距离的平均，"
        "衡量整体偏移幅度，**不区分正负**——上下波动会被抵消掉一部分。"
    )
    items.append(
        f"要看不会抵消的波动幅度，用第 4 节表头与第 5 节的「Spread偏差(绝对对数)」"
        f"（mean|ln(Spread/RMSE)|）；它对小离散度的欠校准更敏感，"
        f"与本节的线性距 1 互补而不重复。"
    )
    return items


def _section_lead(bundle: EnsBundle) -> List[str]:
    base = bundle.base
    names = bundle.names
    bands = _lead_bands(list(base.rmse_by_lead[names[0]].index))

    lines = [
        "# 5. 分时效表现", "",
        f"分时效分析只覆盖RMSE、CRPS、ACC和Spread/RMSE；FA不参与分段。"
        f"综合RMSE以{cn_number(len(names))}模型的几何均值为100。",
        "",
        f"**分时效段{cn_number(len(names))}模型指标**",
        "",
    ]
    rows = []
    for label, subset in bands:
        relative = _band_relative(base, subset, names)
        for name in sorted(names):
            crps = _band_values(bundle.crps_by_lead[name], subset)
            acc = _band_values(base.acc_by_lead[name], subset)
            ratio = _band_values(bundle.spread_ratio_by_lead[name], subset)
            rows.append([
                label, name, fmt_number(relative[name]),
                fmt_number(crps.mean(axis=1).mean()),
                fmt_number(acc.mean(axis=1).mean()),
                fmt_number(_log_deviation(ratio).mean(), 4),
            ])
    lines += table_lines(
        ["**时效**", "**模型**", "**相对综合RMSE**", "**CRPS**",
         "**ACC**", "**Spread绝对对数偏差**"],
        rows,
    )
    # 此处原本是「## 5.1 时效结论」小标题，已去掉（只留结论内容）。
    lines += [""]
    lines += bullet_lines(_lead_notes(bundle, bands))
    return lines


def _lead_notes(bundle: EnsBundle, bands: Sequence[Tuple[str, List[float]]]) -> List[str]:
    base = bundle.base
    names = bundle.names
    items = []
    for label, subset in bands:
        values = _band_relative(base, subset, names)
        items.append(
            f"{label}：综合相对 RMSE 最低 {_extreme(values, 'min')}，"
            f"最高 {_extreme(values, 'max')}。"
        )

    finite = [float(x) for x in base.rmse_by_lead[names[0]].index if math.isfinite(float(x))]
    if finite:
        middle = (min(finite) + max(finite)) / 2.0
        drop = {}
        for name in names:
            acc = base.acc_by_lead[name]
            early = float(acc[acc.index <= middle].mean(axis=1).mean())
            late = float(acc[acc.index > middle].mean(axis=1).mean())
            drop[name] = early - late
        items.append(
            f"以 {min(finite):g}–{max(finite):g}h 的中点为界，ACC 从短时效到长时效"
            f"掉得最多的是 {_extreme(pd.Series(drop), 'max', suffix=' 个百分点')}，"
            f"掉得最少的是 {_extreme(pd.Series(drop), 'min', suffix=' 个百分点')}。"
        )

    saturated = [
        name for name in names
        if float(base.acc_by_lead[name].iloc[0].mean()) >= 99.0
    ]
    if saturated:
        items.append(
            f"{_named(saturated)} 在首时效的 ACC 已 ≥ 99%，短时效上 ACC 接近饱和、"
            f"区分度低，短时效的模型差异要看 RMSE 与 CRPS。"
        )
    items.append(
        "各时效段的相对综合 RMSE 都以**该段内**各模型的几何均值为 100，"
        "因此只能在同一行里横向比模型，不能跨行比大小。"
    )
    return items


def _section_tests(bundle: EnsBundle) -> List[str]:
    base = bundle.base
    names = bundle.names
    lines = [
        "# 6. 统计检验", "",
        f"主检验为共同日期（{len(base.dates)} 天）上的**配对 Wilcoxon 符号秩检验**，"
        f"每个指标项上把所有模型两两配对；同一项内的多重比较用 **Holm 逐步校正**，"
        f"Holm p < 0.05 才记「显著」。检验用的是逐日样本而不是平均后的单点，"
        f"所以样本量等于共同日期数。RMSE / CRPS / FA 偏差 / Spread-RMSE 距 1 都是"
        f"越小越好，ACC 越大越好。「指标族」= 同一类指标、项 = 变量。",
    ]
    if own_dates_mode(bundle):
        lines += [
            "",
            f"**本节是全报告唯一在共同日期上做的比较**：正文第 4、5 节各表的均值来自"
            f"各模型自己的全部起报日（{date_scope_counts(base)}），与本节的样本不是同一批，"
            f"表里的均值也**不会**等于本节检验时的均值——配对检验要求两边在同一天有数，"
            f"日期对不齐的样本对 Wilcoxon 没有意义。",
        ]
    lines += [
        "",
        "**分指标族显著优胜项汇总**",
        "",
    ]
    rows = []
    for family in bundle.family_wins.index:
        record = bundle.family_wins.loc[family]
        rows.append(
            [family]
            + [f"{name} {int(record[name])}" for name in names]
            + [str(int(record["无显著差异"]))]
        )
    lines += table_lines(
        ["**指标族**"]
        + [f"**{name} 显著优胜项**" for name in names]
        + ["**无显著差异**"],
        rows,
    )
    lines += [""]
    lines += bullet_lines(_test_notes(bundle))
    return lines


def _test_notes(bundle: EnsBundle) -> List[str]:
    base = bundle.base
    names = bundle.names
    total_dates = len(base.dates)
    items = []
    for family in bundle.family_wins.index:
        record = bundle.family_wins.loc[family]
        wins = {name: int(record[name]) for name in names}
        total = bundle.family_totals.get(family, sum(wins.values()) + int(record["无显著差异"]))
        dominant = max(wins, key=lambda key: wins[key])
        if wins[dominant] == 0:
            items.append(
                f"{family}：{total} 个项上没有任何模型显著优于其余全部模型，"
                f"全部记入「无显著差异」——{cn_number(len(names))}者在这些项上的差异看不出来。"
            )
        else:
            items.append(
                f"{family}：共 {total} 个项，{dominant} 拿下 {wins[dominant]} 个显著优胜项；"
                f"其余 {total - wins[dominant]} 个项上没有模型能压过其余全部。"
            )

    order = base.relative_composite.sort_values().index.tolist()
    items.append(
        f"综合相对 RMSE 的逐日配对检验（{' vs '.join(order[:2])}）基于 {total_dates} "
        f"个共同日期，是本报告最接近「整体精度」的单条检验。"
    )
    if bundle.small_sample:
        items.append(
            f"配对检验可用的共同日期仅 {total_dates} 天：Wilcoxon 符号秩检验在 "
            f"n < {SMALL_SAMPLE_DATES} 时几乎检验不出差异（n = {total_dates} 时理论最小两侧 p 值约 "
            f"{2.0 / (2 ** total_dates):.2e}，但需要全部样本同号才可能达到），"
            f"「无显著差异」不能读成「两者等价」。"
        )
    items.append(
        "本表的「显著优胜」要求该模型在**同一项上显著优于其余全部模型**，"
        "因此「各模型优胜项数 + 无显著差异 = 该项总数」，不会重复计数；"
        "只有两两检验、没有全局检验，结论是两两之间的方向性判断。"
    )
    return items


def _section_risks(bundle: EnsBundle) -> List[str]:
    base = bundle.base
    names = bundle.names
    items = [
        # 第一句说「正文各表的数是从哪些天来的」，第二句说「配对检验用的是哪些天」。
        # 各用各的时这两拨天不是同一批，分开写清楚，别让读者以为是同一套。
        (
            f"样本量：正文与附录各表的统计基于 {date_scope_detail(base)}，"
            f"分时效段的每个点又只有其中一个子集，越靠后的时效样本越少，"
            f"末端时效的曲线波动要谨慎解读。"
            if not own_dates_mode(bundle)
            else f"样本量：正文与附录各表**每个模型用各自全部起报日**"
            f"（{date_scope_counts(base)}），天数最少的那几个模型的曲线与均值波动"
            f"天然比其余模型大；分时效段的每个点又只有各自日期里的一个子集，"
            f"越靠后的时效样本越少，末端时效的曲线波动要谨慎解读。"
        ),
        (
            f"口径边界：正文与附录只使用共同日期。各模型产物里还有大量非共同日期"
            f"（起报日数远多于共同日期的那些模型尤其明显），本报告**没有**使用它们，"
            f"因此不要把本报告的数字与各模型单独跑出来的汇总数直接对比。"
            if not own_dates_mode(bundle)
            else f"口径边界：正文与附录**每个模型用各自全部起报日**，"
            f"因此同表里各列来自不同时段，位次里混着时段差异——"
            f"这是明知不公平仍直接对比的做法，**不构成同口径结论**；"
            f"唯一同口径的部分是第 6 节在 {len(base.dates)} 个共同日期上的配对检验。"
            f"正式验收前应把各批 ``periods`` 对齐后重跑。"
        ),
        f"集合平均场口径：RMSE / ACC / FA 都取自集合平均场（``product_kind`` 为 "
        f"``deterministic`` 的那批结果），衡量的是集合平均场而不是最优成员；"
        f"择优成员带来的收益在本报告中不可见。",
        f"对数底数：本报告的纬向谱偏差用 ln，``weather_rmse_single`` 报告的"
        f"「频谱对数 RMS」用 log10，数值相差 2.3026 倍，跨报告比大小前先统一底数。",
    ]
    if own_dates_mode(bundle):
        thin = _thin_models(bundle)
        if thin:
            detail = "、".join(
                f"{name} 只有 {len(bundle.dates_by_model[name])} 天" for name in thin
            )
            items.append(
                f"**样本量最少的模型：{detail}**（低于 {SMALL_SAMPLE_DATES} 天的经验下限）。"
                f"它们的均值由这些天平均而来，与长时段模型比大小前先看这一条；"
                f"要公平结论就把各批时段对齐后重跑。"
            )
    if bundle.small_sample:
        items.append(
            f"**配对检验可用的共同日期只有 {len(base.dates)} 天**，"
            f"第 6 节的显著性检验不构成任何模型优劣的结论，只能用于看方向。"
        )
    gaps = sorted(set(base.fa_variables) ^ set(base.common_variables))
    if gaps:
        items.append(
            f"指标覆盖不齐：FA 变量为 {_named(base.fa_variables)}，"
            f"RMSE 变量为 {_named(base.common_variables)}，差集为 {_named(gaps)}——"
            f"某模型在缺项指标上没有证据，不要把「没算」当成「算得差」。"
        )
    ratios = {
        name: meta.get("missing_ratio")
        for name in names
        for meta in [base.archives[name].meta]
    }
    if any(isinstance(r, (int, float)) and float(r) > 0.5 for r in ratios.values()):
        worst = max(
            names,
            key=lambda name: float(base.archives[name].meta.get("missing_ratio") or 0.0),
        )
        items.append(
            f"数据完整性：{worst} 的起报点缺失率最高"
            f"（{fmt_number(BASELINE * float(base.archives[worst].meta.get('missing_ratio') or 0.0), 2)}%），"
            f"它覆盖的日期段明显短于其余模型，共同日期被最短的那一批限定。"
        )
    items.append(
        f"验收建议：本报告作为**方向性**依据；正式验收前把各批 ``periods`` 对齐到"
        f"同一段起报日期、补齐缺失变量后重跑，再以重跑结果为准。"
    )
    return ["# 7. 风险与验收建议", ""] + [f"• {item}" for item in items]


def _appendix(bundle: EnsBundle, figures: Mapping[str, str]) -> List[str]:
    base = bundle.base
    count = cn_number(len(bundle.names))
    intro = (
        f"本附录使用{count}个模型共同的{len(base.dates)}个日期"
        f"（{base.dates[0]} 至 {base.dates[-1]}）。"
        f"正文第 4 节的指标曲线使用同一批共同日期；本附录只保留"
        f"分纬度带结果（附 L）与分季节结果（附 S）两个可选块。"
        if not own_dates_mode(bundle)
        else f"本附录中**每个模型画的是自己全部起报日上的平均**"
        f"（{date_scope_counts(base)}），"
        f"因此图上的日期范围并不相同，曲线高低里同时含模型差异与时段差异。"
        f"正文第 4 节的指标曲线与这里同口径；第 6 节的配对检验另用共同日期，"
        f"与本附录不同源。本附录只保留分纬度带结果（附 L）与分季节结果"
        f"（附 S）两个可选块。"
    )
    return [
        "# 附录：补充诊断图",
        "",
        intro,
        "",
    ] + _region_block(bundle, figures) + [""] + _season_block()


def _season_block() -> List[str]:
    """「附 S 分季节结果（可选）」：固定输出「本批未出」的静态说明。

    这一块**不预生成**：长表里没有季节列，季节要从 ``init_time`` 现推，而分季节
    对比是按需的（用户点名要哪一季、哪个纬度带才切）。口径 2026-09-22 已定，
    工序写在 ``references/rmse-batch-evaluation.md`` §6；这里只留节位与指路，
    与 ``assets/templates/weather_rmse_ens.md`` 的「附 S」逐字一致。
    """
    return [
        "## 附 S 分季节结果（可选）",
        "",
        "**本块本批未出。** 产物长表里没有季节维度（只有 `init_time`），季节要现推；",
        "分季节对比是**按需补**的——用户点名要哪一季、哪个纬度带，就临时切一份出来，",
        "**不需要重跑评测**。渲染器不预生成这一块，也不静默省略、不留空表。",
        "",
        "两条口径 2026-09-22 已定，照做即可（完整工序见",
        "`references/rmse-batch-evaluation.md` §6「按需子集对比」）：",
        "",
        "- 季节按**起报时刻**（`init_time`）切，**不按** `valid_time`——报告整套契约建立在",
        "  「同一天就是同一个 `init_date`」上，一组 `init_date` 必须干净地属于一个季节；",
        "  （默认口径下这些日期还必须是各模型**共有**的，`--own-dates` 下则各模型各切各的，",
        "  同一季里各模型的样本天数可以不同——别在这种情况下印「共 N 天」）；",
        "- DJF 按**气象冬季**归组（当年 12 月 + 次年 1、2 月）。不满整年的批次两个冬季",
        "  **各只有一截**（实测 `20250102–20251215`：DJF(2024/25) 只有 1/2–2/28，",
        "  DJF(2025/26) 只有 12/1–12/15），所以报 DJF 要说清是**哪一个冬季**，或者干脆",
        "  只出 MAM/JJA/SON；**不要**把同一自然年的 12 月拼进去充数。",
        "",
        "补数之后这一块的形态与「附 L」一致：一张分季节表（一行一个季节，按 DJF、MAM、JJA、",
        "SON 升序，各模型各占一列，末列 `Best`；基线在**子集内**重算并写在表下），加一条",
        "带天数的判读句。`season_summary.svg`（`图 S1`）与 `season_rmse_vs_lead.svg`（`图 S2`）",
        "是可选图——**没出图就别留号**，不指不存在的文件。",
    ]


def _region_block(bundle: EnsBundle, figures: Mapping[str, str]) -> List[str]:
    """「附 L 分纬度带结果（可选）」：分带准确度 + 分带离散度，三张图。

    没有共同的 region 行时保留节位、写明原因并标「本批未出」，不静默省略、
    也不留一张空表。
    """
    base = bundle.base
    count = cn_number(len(base.names))
    has_spread = not bundle.region_spread_ratio.empty and "lat_band_spread_ratio" in figures
    lines = ["## 附 L 分纬度带结果（可选）", ""]
    if not base.region_names:
        lines += [f"**本块本批未出。** {base.region_note}", ""]
        return lines

    table = region_table(base)
    rows, best_of, best_global = table.rows, table.best_of, table.best_global
    spread_rows: List[List[str]] = []
    for region in base.region_names:
        if not has_spread:
            continue
        if region in bundle.region_spread_ratio.index:
            spread = bundle.region_spread_ratio.loc[region]
            spread_rows.append(
                [region_display(region, base.region_labels)]
                + [fmt_number(spread[name]) for name in base.names]
                # 「更接近1」= 距 1 的绝对偏差最小，同一套并列规则复用。
                + [_best_cell((spread - 1.0).abs())]
            )
        else:
            spread_rows.append(
                [region_display(region, base.region_labels)]
                + ["—"] * len(base.names)
                + ["—"]
            )

    lines += [
        "本块把长表里 `region` 非空的行**单独汇总**，全球行**不参与**——全球平均与纬度带平均",
        "是两个量，混在一起算会把带间差异整个抹平。带名与边界照搬评测配置",
        "`options[\"regions\"]`：**报告不翻译、也不写死任何带名**，配置里叫什么就显示什么。",
        f"表里的相对RMSE是**对本带基线**取的（100 = 该带内的{count}模型几何均值），",
        "不是全球基线——热带和极区的绝对 RMSE 差一个量级，不除本带基线没法横向比。",
        "",
        "**分纬度带综合相对RMSE**",
        "",
    ]
    lines += table_lines(["纬度带"] + list(base.names) + ["Best"], rows)
    if has_spread:
        lines += ["", "**分纬度带 Spread/RMSE（距 1 越近越好）**", ""]
        lines += table_lines(
            ["纬度带"] + [f"{name} 平均Spread/RMSE" for name in base.names] + ["更接近1"],
            spread_rows,
        )
    lines += ["", _region_reading_ens(bundle, best_global, best_of), ""]
    lines += [
        f"![lat_band_rmse_vs_lead]({figures['lat_band_rmse_vs_lead']})",
        "",
        f"图 L1：各纬度带集合均值 RMSE 的相对值随 lead 变化"
        f"（各自除以本带基线，100 = 该带内{count}模型几何均值；"
        f"{date_scope_word(bundle)}平均）。",
        "",
        f"![lat_band_summary]({figures['lat_band_summary']})",
        "",
        f"图 L2：分纬度带综合相对 RMSE"
        f"（100 = 该带内{count}模型几何均值；{date_scope_word(bundle)}平均）。",
    ]
    if has_spread:
        lines += [
            "",
            f"![lat_band_spread_ratio]({figures['lat_band_spread_ratio']})",
            "",
            f"图 L3：分纬度带平均 Spread / Ensemble-mean RMSE"
            f"（虚线为理想值 1；{date_scope_word(bundle)}平均）。",
        ]
    return lines


def _region_reading_ens(
    bundle: EnsBundle, best_global: Mapping[str, Any], best_of: Mapping[str, Mapping[str, Any]],
) -> str:
    """集合分带判读：名次反转 + 离散度是否随带变化（两个方向业务含义相反）。"""
    base = bundle.base
    global_names = list(best_global)
    tied = len(global_names) > 1
    head_global = "、".join(global_names) if global_names else "—"
    tie_note = "（并列，分不出高下）" if tied else ""
    global_value = fmt_number(base.relative_composite[global_names[0]]) if global_names else "—"

    flips = [
        (region, list(best_of[region]))
        for region in base.region_names
        if set(best_of[region]) != set(global_names)
    ]
    if flips:
        detail = "；".join(
            f"{region_display(region, base.region_labels)} 的第一名是 {'、'.join(names)}"
            f"（{fmt_number(base.region_relative_composite.loc[region, names[0]])}）"
            for region, names in flips
        )
        head = (
            f"全球口径下综合相对RMSE最低的是 {head_global}{tie_note}（全球 {global_value}），"
            f"但分带看这个结论并不是处处成立：{detail}——优势不是全纬度一致的，"
            f"选型时要写清用在哪一纬带。"
        )
    elif tied:
        head = (
            f"各带的第一名与全球口径一致，都是 {head_global}{tie_note}"
            f"（全球 {global_value}）——全球与各带都分不出高下，"
            f"谈不上哪个模型在某一纬带上占优。"
        )
    else:
        head = (
            f"各带的第一名与全球口径一致，都是 {head_global}"
            f"（全球 {global_value}）——它的优势在全纬度上都成立。"
        )

    if not bundle.region_spread_ratio.empty:
        # 「低纬误差小但过度离散、中高纬误差大却离散不足」是常见形态，两件事的
        # 业务含义完全相反（前者该收窄集合、后者该放宽），所以必须分开点名。
        per_region = bundle.region_spread_ratio
        means = per_region.mean(axis=1)
        if float(means.max() - means.min()) < 1e-9:
            # 各带数值一模一样时，报「最高/最低」等于把同一条数念两遍再安上两个
            # 带名，读起来像是找到了纬度依赖，其实是没切。宁可说清没切成。
            head += (
                f" 离散度一栏在本批各纬度带上数值完全相同（都是 {fmt_number(float(means.iloc[0]))}），"
                f"看不出纬度依赖——这一批的 spread 行看上去没有按带切开，"
                f"要回评测侧核对，本批不能据此判断「离散度不随纬度变化」。"
            )
        else:
            over = means.idxmax()
            under = means.idxmin()
            head += (
                f" 离散度随纬度带变化：{region_display(over, base.region_labels)} 的 Spread/RMSE "
                f"最高（{fmt_number(means.loc[over])}），"
                f"{region_display(under, base.region_labels)} 最低"
                f"（{fmt_number(means.loc[under])}）——偏离 1 的方向决定该收窄还是放宽集合，"
                f"要按带分别处理，不能拿全球平均的一个数代替。"
            )
    if bundle.region_spread_skipped:
        skipped = "、".join(
            region_display(r, base.region_labels) for r in bundle.region_spread_skipped
        )
        head += f" {skipped} 上没有 spread 行，离散度一栏记 —。"
    return head


def _check_figure_contract(
    text: str, count: int = 9, required: int = 7, paired: Sequence[int] = (8, 9)
) -> None:
    """骨架的编号契约：``图 N：`` 的分布必须合法。

    - ``图 N：``（``N`` 在 ``1..count``）**最多出现一次**，且首现位置递增；
    - ``1..required`` 是**必出图**——图 1 总体指标热力图 + §4.1–§4.5 的五张
      曲线图 + §4.6 空间分布（有 CSV 时）+ §4.7 纬向谱，缺一个就报错；
    - ``required+1..count`` 允许缺席：那几节的产物可能没出数（骨架对这种情况的
      约定是保留节位、写明「本批未出」及原因，而不是画一张空图，更不是不指图
      却把号占掉）。因此**号可以是不连续的**——比如图 8/9 整块缺（球谐带不可得）；
    - ``paired`` 是球谐带的两张图，**成对出现或成对缺席**。

    有空间分布图时 ``count``/``required``/``paired`` 一起加 k（k = 空间图张数），
    k=0 时与改版前逐字等价。

    ``图 L1：``…``图 L3：`` 是**独立命名空间**（见骨架的编号契约）：可选块整块
    删掉时正文编号不受影响，正是把它单开一套号的目的，所以单独校验、且不参与
    ``1..count`` 的递增关系。
    """
    positions = []
    for index in range(1, count + 1):
        marker = f"图 {index}："
        found = text.count(marker)
        if found > 1:
            raise ValueError(f"图注契约被破坏：{marker} 出现 {found} 次，最多 1 次")
        if found == 0:
            if index <= required:
                raise ValueError(
                    f"图注契约被破坏：{marker} 缺失（图 1..{required} 是必出图）"
                )
            continue
        positions.append((index, text.index(marker)))
    if [where for _, where in positions] != sorted(where for _, where in positions):
        raise ValueError(f"图注契约被破坏：图 1..{count} 的首现位置不是递增的")

    present = [
        f"图 {index}：" for index in paired if f"图 {index}：" in text
    ]
    if present and len(present) != len(paired):
        raise ValueError(
            f"图注契约被破坏：球谐带两张图必须成对出现，现在是 {present}"
        )

    band = [f"图 L{index}：" for index in range(1, 4)]
    present = [marker for marker in band if marker in text]
    # 允许缺席，但必须是**从 L1 起的连续段**：整块不出（L1 就没有，标「本批未出」），
    # 或者 L1..Lk 连续。中间跳号（有 L1、L3 没 L2）说明有节的产物数据没出，
    # 编号却留了洞——那正是骨架要避免的。
    if present != band[: len(present)]:
        raise ValueError(
            f"图注契约被破坏：分纬度带的图号必须从 图 L1 起连续，现在是 {present}"
        )
    for marker in present:
        if text.count(marker) != 1:
            raise ValueError(f"图注契约被破坏：{marker} 出现 {text.count(marker)} 次")


def render_ens(bundle: EnsBundle, figures: Mapping[str, str], *,
               archive: Optional[str] = None,
               spatial: Optional[SpatialPayload] = None) -> str:
    """把 :class:`EnsBundle` 渲染成骨架形状的完整 Markdown。

    ``spatial`` 是 §4.6 的月平均空间分布（含离散度行）；None 时该节写
    「本批未出」，图号与本节不存在时逐字相同。
    """
    base = bundle.base
    names = bundle.names
    shown = _named(sorted(names))
    count = cn_number(len(names))
    first, last, leads = lead_span(base.rmse_by_lead[names[0]].index)
    if spatial is not None:
        figures = {**dict(figures), **spatial.figures}
    k = spatial.data.figure_count if spatial is not None else 0

    lines: List[str] = [
        f"# 集合预报{count}模型综合评估报告",
        "",
        f"**产物：{archive or _named(names)}**",
        "",
        f"报告日期：{report_date()}    报告类型：集合预报（{bundle.member_label}成员）",
        "",
        f"**评估对象：{shown}**",
        "",
    ]
    for section in (
        _section_conclusion(bundle, first=first, last=last, leads=leads),
        _section_samples(bundle),
        _section_metrics(bundle),
        _section_overall(bundle, figures, spatial=spatial, figure_offset=k),
        _section_lead(bundle),
        _section_tests(bundle),
        _section_risks(bundle),
        _appendix(bundle, figures),
    ):
        lines += section
        lines.append("")

    text = "\n".join(lines)
    _check_figure_contract(
        text, count=9 + k, required=7 + k, paired=(8 + k, 9 + k)
    )
    return text


# ====================================================================== 入口


def build_ens_report(archives: Sequence[Archive], out_dir: Path, *,
                     archive: Optional[str] = None,
                     members: Optional[int] = None,
                     own_dates: bool = False,
                     spatial_csv: Optional[Path] = None,
                     spatial_model: Optional[str] = None) -> Dict[str, Path]:
    """出图 + 写 ``REPORT.md``，返回产物名 → 路径。

    ``members`` 是命令行给的集合成员数（见 :func:`analyze_ens`）：产物里没有这个
    字段，不给就渲染成「—」。

    ``own_dates`` 让每个模型按自己的全部起报日统计（见 :func:`analyze_ens`）。
    各模型覆盖时段差很多、又不想为了对齐而丢数据时开它——报告里会点名各用了多少天、
    以及基线跨了不同时段这件事。

    ``spatial_csv`` 是 §4.6 的集合版月平均宽表（必须含 ``spread`` 列，来自
    ``extract_spatial_distribution_ens.py``）；None 时该节写「本批未出」。

    Raises:
        ValueError: CSV 没有 ``spread`` 列（喂了确定性取数的 CSV）。
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 先校验空间 CSV 再算全量统计：喂错了 CSV 不该让人等完 analyze_ens 才报错。
    spatial_data = None
    if spatial_csv is not None:
        spatial_data = load_spatial_csv(Path(spatial_csv))
        if not spatial_data.has_spread:
            raise ValueError(
                f"{spatial_csv} 没有 spread 列；集合报告的空间分布要用 "
                "extract_spatial_distribution_ens.py 取数"
            )

    bundle = analyze_ens(archives, members=members, own_dates=own_dates)
    names = bundle.names
    base = bundle.base
    count = cn_number(len(names))
    # 图标题里那截日期口径。各用各的时**不能**印一个区间——每个模型的区间都不同，
    # 随便挑一个印出来就是假的。默认口径逐字保持原样。
    date_note = (
        f"{len(base.dates)} 个共同日期平均（{date_span(base.dates)}）"
        if not own_dates_mode(bundle)
        else "各模型各自的全部起报日平均"
    )

    # 插入顺序 = 图号顺序：图 1 是 §4 开头的总体指标热力图，图 2–6 是
    # §4.1–§4.5 的指标曲线，图 7…7+k-1 是 §4.6 空间分布（可选），图 7+k 是
    # §4.7 纬向谱，图 8+k/9+k 是 §4.8 球谐带。所有图一律落 SVG
    # （矢量，论文排版用）。
    figures = {
        "overall_heatmap": "overall_heatmap.svg",
        "ensemble_rmse_vs_lead": "ensemble_rmse_vs_lead.svg",
        "ensemble_crps_vs_lead": "ensemble_crps_vs_lead.svg",
        "ensemble_acc_vs_lead": "ensemble_acc_vs_lead.svg",
        "ensemble_spread_rmse_ratio_vs_lead": "ensemble_spread_rmse_ratio_vs_lead.svg",
        "ensemble_fa_ratio_vs_lead": "ensemble_fa_ratio_vs_lead.svg",
        "ensemble_spectrum": "ensemble_spectrum.svg",
    }

    # 变量单位（集合产物统一用 SI 量纲）：纵轴写成「RMSE (Pa)」这类形式。
    variable_units = {
        "msl": "Pa",
        "u200": r"m s$^{-1}$",
        "v200": r"m s$^{-1}$",
        "ws200": r"m s$^{-1}$",
        "z500": r"m$^{2}$ s$^{-2}$",
    }

    # 正文指标曲线的论文版式（图 2–6 共用）：两列 7.2 in、统一模型色实线、去网格、
    # 底部灰框图例、2400 dpi；图 2 加前后期两个 inset，图 4 只加后期。字号由
    # plot_model_panels 按图宽查表，不在这里写死。
    panel_style = dict(
        ncols=2,
        panel_size=(3.6, 2.7),
        linewidth=2.0,
        marker=None,
        zoom_y=True,
        line_styles=None,
        grid=False,
        minimal_spines=True,
        legend_box=True,
        center_last=False,
        xlabel="Forecast lead time (h)",
        panel_letters=True,
        dpi=2400,
    )

    # 图 2：前期 6–30h、后期 330–360h 各一个放大 inset，只标纵轴刻度；
    # msl/z500 跨度大，固定 10 一格，其余变量自动取 4 档。
    rmse_style = dict(panel_style)
    rmse_style.update(
        zoom_windows=[(6.0, 30.0), (330.0, 360.0)],
        zoom_linewidth=1.2,
        zoom_tick_steps={"msl": 10.0, "z500": 10.0},
        zoom_positions=(
            (0.02, 0.62, 0.36, 0.35),
            (0.54, 0.02, 0.44, 0.36),
        ),
    )

    # 图 4：ACC 随时效单调下降，右上方是空白区，只放大后期 330–360h，
    # inset 放右上角；往右收一点，给左侧的纵轴刻度让开 200h 附近的曲线。
    # ACC 是百分比，纵轴刻度固定 2 一格。
    acc_style = dict(panel_style)
    acc_style.update(
        zoom_windows=[(330.0, 360.0)],
        zoom_linewidth=1.2,
        zoom_tick_steps={"msl": 2.0, "z500": 2.0},
        zoom_positions=((0.60, 0.62, 0.38, 0.35),),
    )

    plotter = DetPlotter()

    # 图 1：总体指标热力图（§4 开头，和上面那张总体指标表同一批数）
    _overall_heatmap(out_dir, figures["overall_heatmap"], bundle)

    # 图 2：集合均值 RMSE（§4.1）
    path = out_dir / figures["ensemble_rmse_vs_lead"]
    plotter.plot_model_panels(
        base.rmse_by_lead, base.common_variables, ylabel="RMSE",
        units=variable_units,
        save_path=path, **rmse_style,
    )
    plt.close("all")

    # 图 3：CRPS（§4.2）
    path = out_dir / figures["ensemble_crps_vs_lead"]
    plotter.plot_model_panels(
        bundle.crps_by_lead, bundle.crps_variables, ylabel="CRPS",
        units={var: variable_units.get(var, "-") for var in bundle.crps_variables},
        save_path=path, **panel_style,
    )
    plt.close("all")

    # 图 4：ACC（§4.3）
    path = out_dir / figures["ensemble_acc_vs_lead"]
    plotter.plot_model_panels(
        {name: base.acc_by_lead[name][bundle.acc_variables] for name in names},
        bundle.acc_variables, ylabel="ACC",
        units={var: "%" for var in bundle.acc_variables},
        save_path=path, **acc_style,
    )
    plt.close("all")

    # 图 5：Spread / RMSE（§4.4，理想值 1）
    path = out_dir / figures["ensemble_spread_rmse_ratio_vs_lead"]
    plotter.plot_model_panels(
        bundle.spread_ratio_by_lead, bundle.spread_variables,
        ylabel="Spread / RMSE", ref_line=1.0,
        units={var: "-" for var in bundle.spread_variables},
        # 虚线保留，但不进图例：图例只列模型，避免多出一行 "Reference = 1"。
        ref_line_label=None,
        save_path=path, **panel_style,
    )
    plt.close("all")

    # 图 6：FA ratio（§4.5，理想值 1）
    path = out_dir / figures["ensemble_fa_ratio_vs_lead"]
    plotter.plot_model_panels(
        base.fa_ratio_by_lead, base.fa_variables,
        ylabel="FA ratio", ref_line=1.0,
        units={var: "-" for var in base.fa_variables},
        # 同上：虚线保留，图例里不要 "Reference = 1"。
        ref_line_label=None,
        save_path=path, **panel_style,
    )
    plt.close("all")

    # 图 7+k：纬向功率谱（§4.7；一变量一子图，模型曲线 + 共享观测谱）
    _spectrum_panels(
        out_dir, figures["ensemble_spectrum"], bundle.spectrum_by_variable,
        variables=base.spectrum_variables, names=names,
    )

    # 图 8+k/9+k：球谐带 log-RMS 与均值比（§4.8）。与 wave_report 同源，数据在
    # analyze_ens 里算好；sph_available 为假时两张图一起不出、图号一起空着。
    if bundle.sph_available:
        figures["spherical_bands_logrms"] = "spherical_bands_logrms.svg"
        figures["spherical_bands_ratio"] = "spherical_bands_ratio.svg"
        _grouped_bars(
            out_dir, figures["spherical_bands_logrms"], bundle.sph_band_logrms.T,
            suptitle=None, ylabel="log-RMS×100", y_grid=False,
        )
        plt.close("all")
        _grouped_bars(
            out_dir, figures["spherical_bands_ratio"], bundle.sph_band_ratio.T,
            suptitle=None, ylabel="pred / obs", ref_line=1.0, y_grid=False,
        )
        plt.close("all")

    # 图 L1–L3：分纬度带（可选块）。产物里没有共同的 region 行就**不出图**，也不往
    # artifacts 里塞不存在的路径——报告那边会写「本批未出」，指一张没生成的图更糟。
    if base.region_names:
        band_labels = [
            region_display(region, base.region_labels) for region in base.region_names
        ]
        figures["lat_band_rmse_vs_lead"] = "lat_band_rmse_vs_lead.svg"
        figures["lat_band_summary"] = "lat_band_summary.svg"
        figures["lat_band_spread_ratio"] = "lat_band_spread_ratio.svg"

        plotter.plot_model_panels(
            {
                name: pd.DataFrame({
                    label: base.region_lead_relative[region][name]
                    for region, label in zip(base.region_names, band_labels)
                })
                for name in names
            },
            band_labels,
            ylabel="综合相对 RMSE（100 = 该带内几何均值）", ref_line=100.0,
            ncols=min(len(band_labels), 3),
            suptitle=f"{count}模型分纬度带综合相对 RMSE 随 lead 变化（{date_note}）",
            save_path=out_dir / figures["lat_band_rmse_vs_lead"],
        )
        plt.close("all")

        summary = base.region_relative_composite.copy()
        summary.index = band_labels
        plotter.plot_group_bars(
            summary, ylabel="综合相对 RMSE（100 = 该带内几何均值）",
            ref_line=100.0,
            suptitle=f"分纬度带综合相对 RMSE（{count}模型，{date_note}）",
            save_path=out_dir / figures["lat_band_summary"],
        )
        plt.close("all")

        if not bundle.region_spread_ratio.empty:
            spread = bundle.region_spread_ratio.copy()
            spread.index = [
                region_display(region, base.region_labels)
                for region in bundle.region_spread_ratio.index
            ]
            plotter.plot_group_bars(
                spread, ylabel="Spread / Ensemble-mean RMSE", ref_line=1.0,
                suptitle=f"分纬度带平均 Spread/RMSE（理想值 1，{date_note}）",
                save_path=out_dir / figures["lat_band_spread_ratio"],
            )
            plt.close("all")
        else:
            # 一个带都取不到 spread 行：整段离散度（表 + 图 L3）都不出，不留空图名。
            # 「图 L1…Lk 必须是从 L1 起的连续段」由 _check_figure_contract 守。
            figures.pop("lat_band_spread_ratio")

    # §4.6 空间分布（集合版固定 4 行：均值 / ERA5 / 偏差 / 离散度）。
    spatial = None
    if spatial_data is not None:
        model = spatial_model or bundle.names[0]
        paths = render_spatial(spatial_data, out_dir, model=model, mode="ens")
        figures.update({key: path.name for key, path in paths.items()})
        spatial = SpatialPayload(
            data=spatial_data, model=model, mode="ens",
            figures={key: path.name for key, path in paths.items()},
        )

    text = render_ens(bundle, figures, archive=archive, spatial=spatial)
    report_path = out_dir / "REPORT.md"
    report_path.write_text(text, encoding="utf-8", newline="\n")
    print(f"已保存: {report_path}")

    artifacts: Dict[str, Path] = {"report": report_path}
    for key, filename in figures.items():
        artifacts[key] = out_dir / filename
    return artifacts
