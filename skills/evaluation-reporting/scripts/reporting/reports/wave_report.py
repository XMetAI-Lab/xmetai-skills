#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""纬向 FFT 与球谐谱补充分析报告：读评估产物 → 谱统计 → 渲染 Markdown。

输入同 ``det_report``（``scores.csv`` + ``manifest.json`` + ``diagnostics/``）。
``weather_rmse_wave.md`` 骨架已删，谱分析并入 ``weather_rmse_single.md`` §4.5 /
``weather_rmse_ens.md`` §4.8；本模块单独出报告时没有独立格式契约。

**两套谱别混**：纬向 FFT 谱来自 ``diagnostics/spectrum_by_init.csv``（逐起报一条
曲线，波数 1–720），log-RMS 用 **ln**——``det_report.spectrum_rms`` 是 log10，
差 ×2.3026，本模块不复用它。球谐带功率来自长表 ``spherical_bands`` 展开的三行，
频带靠 ``group`` 列区分，按球谐总阶数 ``l`` 分段（``DEFAULT_SPHERICAL_BANDS``），
与按纬向波数分的 ``WAVE_BANDS`` 不可换算。

球谐节在配置无 ``spherical_bands`` 或缺 ``group`` 列时整节写「本批未出」占位，
不静默省略、不留空表，**也绝不跨频带平均**。
"""

from __future__ import annotations

from pathlib import Path
from typing import (
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
from reporting.reports.det_plots import DetPlotter
from reporting.core.style import (
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
    PairTest,
    analyze,
    _signed,
    load_spectrum,
    metric_rows,
    paired_tests,
    probe_dates,
    report_date,
)

from reporting.spectra.specialized import DEFAULT_SPHERICAL_BANDS

#: 纬向谱的完整波数范围（``diagnostics/spectrum_by_init.csv`` 的 wavenumber 1–720）
FULL_K_MAX = 720

#: 与球谐损失的最高阶数同范围的「匹配尺度」上限
MATCHED_K_MAX = 128

#: 五个波数带：(下界, 上界, 名称)，按行星尺度 → 小尺度
WAVE_BANDS: Tuple[Tuple[int, int, str], ...] = (
    (1, 3, "行星波"),
    (4, 9, "长波"),
    (10, 20, "天气尺度"),
    (21, 60, "中尺度"),
    (61, FULL_K_MAX, "小尺度"),
)

#: 球谐带频带 ``(下界, 上界)``。直接取**算分那一侧**的常量，报告端与
#: ``metrics/specialized.py`` 的 ``spherical_bands`` 同源，不另立口径——
#: 这里写死一份，配置改了带边界报告就会跟着错。
SPHERICAL_BANDS: Tuple[Tuple[int, int], ...] = tuple(
    (int(lo), int(hi)) for lo, hi in DEFAULT_SPHERICAL_BANDS
)


def _band_tag(low: int, high: int) -> str:
    """频带标签，也是长表 ``group`` 列的取值（``1_4`` / ``5_20`` …）。"""
    return f"{low}_{high}"


def _band_label(low: int, high: int) -> str:
    """报告表格里显示的频带名。"""
    return f"{low}–{high}"

#: 球谐带功率谱的频带数（骨架里写死「五个球谐总阶数频带」）
SPHERICAL_BAND_COUNT = len(SPHERICAL_BANDS)

#: 产物里没有**可用**的球谐带功率结果时写进正文的说明。三种情况都走这里：
#: 一是配置的 ``metrics`` 里压根没有 ``spherical_bands``；二是算了，但结果行缺
#: ``group`` 列、认不出哪一行属于哪个频带（修复前的产物就是这样，重跑评测即可）；
#: 三是老中间层归档转过来的产物——转换器 2026-10-10 之前没翻译球谐带这一块。
SPHERICAL_ABSENT = (
    "产物里没有可用的球谐带功率结果（要么配置的 `metrics` 里没有 `spherical_bands`，"
    "要么结果行缺 `group` 列、认不出频带，要么是老中间层归档转过来时这一块没被翻译），"
    "**本批未出**。"
)

def _season_block() -> List[str]:
    """「附 S 分季节结果（可选）」——与另两份骨架同一段静态说明。

    这一块**不预生成**：长表里没有季节列，季节要从 ``init_time`` 现推，而分季节
    对比是按需的（用户点名要哪一季、哪个纬度带才切）。口径 2026-09-22 已定，
    工序写在 ``references/rmse-batch-evaluation.md`` §6；这里只留节位与指路，
    而不是留一张空表。
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
        "  「同一天就是同一个 `init_date`」上，一组 `init_date` 必须干净地属于一个季节",
        "  （默认口径下这些日期还必须是各模型共有的；各模型各用全部起报日时，",
        "  同一季里各模型的样本天数可以不同，别在这种情况下印「共 N 天」）；",
        "- DJF 按**气象冬季**归组（当年 12 月 + 次年 1、2 月）。不满整年的批次两个冬季",
        "  **各只有一截**（实测 `20250102–20251215`：DJF(2024/25) 只有 1/2–2/28，",
        "  DJF(2025/26) 只有 12/1–12/15），所以报 DJF 要说清是**哪一个冬季**，或者干脆",
        "  只出 MAM/JJA/SON；**不要**把同一自然年的 12 月拼进去充数。",
        "",
        "季节对这份补充报告尤其要紧：谱比随季节变化明显（冬夏的行星波活动差一个量级），",
        "把四个季节平均成一条谱曲线，会把「某个模式只在一个季节小尺度偏低」这类问题",
        "整个抹掉。补数之后这一块的形态与主报告的「附 L」一致：一张分季节表（一行一个季节，",
        "按 DJF、MAM、JJA、SON 升序，各模型各占一列，末列 `最佳`；基线在**子集内**重算",
        "并写在表下），加一条带天数的判读句。`season_summary.png`（`图 S1`）与",
        "`season_rmse_vs_lead.png`（各季节综合相对RMSE 随预报时效的变化，`图 S2`）",
        "是可选图——**没出图就别留号**，不指不存在的文件。",
    ]


# ====================================================================== 谱指标


def _numeric(frame: pd.DataFrame) -> pd.DataFrame:
    """把谱表洗成纯 float 的 波数 × 日期 表。"""
    data = frame.copy()
    data.index = pd.to_numeric(data.index, errors="coerce")
    data = data[data.index.notna()]
    data = data.apply(pd.to_numeric, errors="coerce").sort_index()
    return data


def log_rms_ln(pred: pd.DataFrame, obs: pd.DataFrame,
               kmin: int = 1, kmax: int = FULL_K_MAX) -> pd.Series:
    """逐日 ``100 × RMS[ln(pred/obs)]``（波数 ``kmin..kmax``）。

    Returns:
        ``pd.Series``，index = 日期（``YYYYMMDD``），value = log-RMS×100。
    """
    pred = _numeric(pred)
    obs = _numeric(obs)
    band = (pred.index >= kmin) & (pred.index <= kmax)
    pred, obs = pred[band], obs[band]
    values: Dict[str, float] = {}
    for date in pred.columns:
        if date not in obs.columns:
            continue
        p, o = pred[date], obs[date]
        mask = (p > 0) & (o > 0)
        if not mask.any():
            continue
        ratio = (p[mask] / o[mask]).values
        values[str(date)] = 100.0 * float(np.sqrt(np.mean(np.log(ratio) ** 2)))
    return pd.Series(values, dtype=float)


def band_power_ratio(pred: pd.DataFrame, obs: pd.DataFrame,
                     kmin: int = 1, kmax: int = FULL_K_MAX) -> float:
    """波数 ``kmin..kmax``、全部日期上的带内积分能量比 ``mean(pred) / mean(obs)``。

    这里**不能**用 ``mean(pred/obs)``：高波数段观测功率接近零，逐波数比值的分母
    最先下溢，均值被极少数噪声点主导——小尺度段会算出 10^5 量级的数字，看着像
    「能量偏了二十万倍」，实际只是除零噪声，没有任何物理含义。先沿波数求和再相除
    是有界的，且正是「该尺度带内的能量偏了多少」这个问题的答案。
    """
    pred = _numeric(pred)
    obs = _numeric(obs)
    band = (pred.index >= kmin) & (pred.index <= kmax)
    pred, obs = pred[band], obs[band]
    powers = []
    for date in pred.columns:
        if date not in obs.columns:
            continue
        p, o = pred[date], obs[date]
        mask = (p > 0) & (o > 0)
        if mask.any():
            powers.append((p[mask].mean(), o[mask].mean()))
    if not powers:
        return float("nan")
    pred_mean = float(np.mean([p for p, _ in powers]))
    obs_mean = float(np.mean([o for _, o in powers]))
    if obs_mean == 0:
        return float("nan")
    return pred_mean / obs_mean


def _band_columns(rows: pd.DataFrame, tag: str, dates: Sequence[str]) -> Dict[str, pd.Series]:
    """某个频带 → ``{起报日: index = lead_h 的列}``。

    一次 ``groupby("init_date")`` 切完；原先对每个（频带, 日期）都整表扫一遍，
    348 天 × 5 频带就是 1740 次全表过滤，球谐带那节因此要跑十几分钟。
    """
    own = rows[rows["group"].astype(str) == tag]
    by_date = {
        str(date): part for date, part in own.groupby("init_date", sort=False)
    }
    empty = pd.Series(dtype=float)
    return {
        date: (
            by_date[date].set_index("lead_h")["value"].sort_index()
            if date in by_date
            else empty
        )
        for date in dates
    }


def load_band_power(root, dates: Sequence[str], variable: str,
                    name: str = "") -> Dict[str, Tuple[pd.DataFrame, pd.DataFrame]]:
    """``spherical_bands`` 的结果 → ``{频带标签: (pred, obs)}``。

    每个频带两张表，``index = lead_h``、``columns = 起报日``——两张都要，因为
    :func:`log_ratio_arrays` 按**逐样本**池化（骨架 §2.1 要的是总均方根，
    不是「先按日均值再对日均值取均方根」）。

    三个量在长表里各占一行（``spherical_band_power_forecast`` / ``_observation``
    / ``_ratio``），是哪个频带靠 ``group`` 列（``1_4`` / ``5_20`` …）。

    Raises:
        FileNotFoundError: 某个起报日没有该变量的球谐带结果。
        ValueError: 结果在，但 ``group`` 列缺失或全空，分不出频带。
    """
    forecast = metric_rows(root, "spherical_band_power_forecast")
    observed = metric_rows(root, "spherical_band_power_observation")
    forecast = forecast[forecast["variable"].astype(str) == str(variable)]
    observed = observed[observed["variable"].astype(str) == str(variable)]
    if forecast.empty:
        raise FileNotFoundError(f"{name or root}: 没有 {variable} 的球谐带功率结果")
    if "group" not in forecast.columns or forecast["group"].isna().all():
        raise ValueError(
            f"{name or root}: {variable} 的球谐带结果没有 group 列，认不出频带；"
            f"把几个频带混起来平均会得到一个不属于任何尺度的数，所以不做"
        )
    wanted = [str(date) for date in dates]
    have = set(forecast["init_date"].astype(str))
    missing = [date for date in wanted if date not in have]
    if missing:
        raise FileNotFoundError(
            f"{name or root}: {len(missing)} 个起报日没有 {variable} 的球谐带结果"
            f"（如 {missing[0]}）"
        )
    groups = set(forecast["group"].astype(str))
    table: Dict[str, Tuple[pd.DataFrame, pd.DataFrame]] = {}
    for low, high in SPHERICAL_BANDS:
        tag = _band_tag(low, high)
        if tag not in groups:
            continue
        pred_columns = _band_columns(forecast, tag, wanted)
        obs_columns = _band_columns(observed, tag, wanted)
        table[tag] = (pd.DataFrame(pred_columns), pd.DataFrame(obs_columns))
    return table


def band_dates(root, dates: Sequence[str], variable: str) -> List[str]:
    """该变量**实际有球谐带结果**的起报日（请求日 ∩ 产物日）。

    球谐带不是每一份产物的每个变量都满覆盖（实测有归档的 u200/v200/ws200 只算到
    某个日期为止）。口径是**有多少算多少**：缺的起报日不进统计，但也不让它把整节
    拖没——缺了多少由 :func:`_common.sph_coverage_notes` 写在正文里。
    """
    rows = metric_rows(root, "spherical_band_power_forecast")
    rows = rows[rows["variable"].astype(str) == str(variable)]
    have = set(rows["init_date"].astype(str))
    return [str(date) for date in dates if str(date) in have]


def probe_bands(root, dates: Sequence[str], variables: Sequence[str]) -> bool:
    """产物里有没有**可用**的球谐带功率结果。

    「可用」= 分得出频带（``group`` 列非空）**且**至少一个变量在至少一个请求日上
    有结果。逐变量逐日期的满覆盖**不是**出这一节的前提：缺的日子由
    :func:`band_dates` 取交集跳过、正文写明。

    这里曾经判「日期集合 ⊆ 有数的日期 且 变量集合 ⊆ 有数的变量」——两个集合各判
    一次、不管交叉，比 :func:`load_band_power` 的逐变量逐日期宽松，会出现
    「probe 说有、load 才炸」的落差；现在两边的口径对齐了。
    """
    if not dates or not variables:
        return False
    rows = metric_rows(root, "spherical_band_power_forecast")
    if rows.empty or "group" not in rows.columns or rows["group"].isna().all():
        return False
    wanted = {str(date) for date in dates}
    return any(
        not rows[
            (rows["variable"].astype(str) == str(variable))
            & (rows["init_date"].astype(str).isin(wanted))
        ].empty
        for variable in variables
    )


def log_ratio_arrays(pred: pd.DataFrame, obs: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """lead × 日期 的带功率表 → 全样本 ``(ln(pred/obs), pred/obs)``。

    骨架 §2.1 把球谐带 log-RMS 定义为「**所有**日期、变量、lead 的总均方根」，
    所以这里交回**逐样本**的数组而不是逐日的均值——先算 log 再平方平均，
    与「先按日均值、再对日均值取均方根」不是一回事。
    """
    pred = _numeric(pred)
    obs = _numeric(obs)
    shared = [d for d in pred.columns if d in obs.columns]
    if not shared:
        return np.empty(0), np.empty(0)
    p = pred[shared].to_numpy(dtype=float)
    o = obs[shared].to_numpy(dtype=float)
    mask = np.isfinite(p) & np.isfinite(o) & (p > 0) & (o > 0)
    if not mask.any():
        return np.empty(0), np.empty(0)
    ratio = p[mask] / o[mask]
    return np.log(ratio), ratio


def band_daily_logrms(pred: pd.DataFrame, obs: pd.DataFrame) -> pd.Series:
    """逐日 ``100 × RMS_lead[ln(pred/obs)]``（配对检验要的日度样本，§5.3 用）。"""
    pred = _numeric(pred)
    obs = _numeric(obs)
    values: Dict[str, float] = {}
    for date in pred.columns:
        if date not in obs.columns:
            continue
        p, o = pred[date].to_numpy(dtype=float), obs[date].to_numpy(dtype=float)
        mask = np.isfinite(p) & np.isfinite(o) & (p > 0) & (o > 0)
        if not mask.any():
            continue
        values[str(date)] = 100.0 * float(
            np.sqrt(np.mean(np.log(p[mask] / o[mask]) ** 2)))
    return pd.Series(values, dtype=float)


def _pooled_logrms(samples: Sequence[np.ndarray]) -> float:
    """若干 ``ln(pred/obs)`` 样本池 → ``100 ×`` 总均方根。"""
    stacked = [s for s in samples if s.size]
    if not stacked:
        return float("nan")
    values = np.concatenate(stacked)
    return 100.0 * float(np.sqrt(np.mean(values ** 2)))


def _mean_ratio(samples: Sequence[np.ndarray]) -> float:
    """若干 ``pred/obs`` 样本池 → 均值比（骨架 §5.1 那一列的「均值比」）。"""
    stacked = [s for s in samples if s.size]
    if not stacked:
        return float("nan")
    return float(np.mean(np.concatenate(stacked)))


def _bias_pct(samples: Sequence[np.ndarray]) -> float:
    """若干 ``pred/obs`` 样本池 → ``100 × mean|pred/obs − 1|``（骨架 §2.1 的比偏差）。"""
    stacked = [s for s in samples if s.size]
    if not stacked:
        return float("nan")
    return 100.0 * float(np.mean(np.abs(np.concatenate(stacked) - 1.0)))


# ====================================================================== 分析


class WaveBundle(NamedTuple):
    """``_single`` 的 Bundle + 谱专项的一切。"""

    base: Bundle
    spectra: Dict[str, Dict[str, pd.DataFrame]]  # 模型 -> 变量 -> 波数 × {pred, obs}
    zonal_full: pd.Series       # 模型 -> 全波数 1–720 的 log-RMS×100
    zonal_matched: pd.Series    # 模型 -> 匹配尺度 1–128 的 log-RMS×100
    zonal_per_date: pd.DataFrame  # 日期 × 模型（全波数，多变量平均）
    band_logrms: pd.DataFrame   # 波数带 × 模型
    band_ratio: pd.DataFrame    # 波数带 × 模型
    variable_logrms: pd.DataFrame  # 变量 × 模型（全波数）
    variable_matched: pd.DataFrame  # 变量 × 模型（匹配尺度 1–128）
    zonal_tests: List[PairTest]   # 全波数口径的配对检验
    primary: str                # 主模型（FGVP_pinpu 之类），没有就是空串
    reference: str              # 主模型的对照模型（FGVP）
    sph_available: bool         # 产物里有没有可用的球谐带功率结果（含 group 列）
    sph_rms: pd.Series          # 模型 -> 球谐带 log-RMS×100（全频带/变量/lead/日期 总均方根）
    sph_bias: pd.Series         # 模型 -> 球谐带比偏差 (%)，mean|pred/obs−1|
    sph_band_logrms: pd.DataFrame   # 频带 × 模型
    sph_band_ratio: pd.DataFrame    # 频带 × 模型（均值比）
    sph_variable: pd.DataFrame      # 变量 × 模型（球谐带 log-RMS×100）
    sph_tests: Dict[str, List[PairTest]]  # 频带名 -> 该频带上的配对检验
    # 模型 -> 变量 -> (实际进统计的起报日数, 请求的起报日数)。满覆盖时两者相等；
    # 有落差就是这一格被跳过了一部分，正文按它写覆盖说明。
    sph_coverage: Dict[str, Dict[str, Tuple[int, int]]]

    @property
    def names(self) -> List[str]:
        return self.base.names

    @property
    def dates(self) -> List[str]:
        return self.base.dates

    @property
    def dates_by_model(self) -> Dict[str, List[str]]:
        """模型 -> 本次统计实际用的起报日（口径见 :func:`det_report.analyze`）。

        默认等于 :attr:`dates`（共同日期）；``own_dates=True`` 时是各模型自己的
        全部起报日。**给单个模型算平均的地方读这里，配对检验读** :attr:`dates`。
        """
        return self.base.dates_by_model

    @property
    def spectrum_variables(self) -> List[str]:
        return self.base.spectrum_variables


def _pick_primary(names: Sequence[str]) -> Tuple[str, str]:
    """挑主模型与对照模型：``X_pinpu`` 是主，同前缀的 ``X`` 是参照。"""
    lowered = {str(name): str(name).lower() for name in names}
    for name, low in lowered.items():
        if low.endswith("_pinpu") or low.endswith("pinpu"):
            stem = low[: -len("_pinpu")] if low.endswith("_pinpu") else low[: -len("pinpu")]
            stem = stem.rstrip("_-")
            for other, other_low in lowered.items():
                if other != name and other_low == stem:
                    return name, other
            return name, ""
    return "", ""


def analyze_wave(archives: Sequence[Archive], *,
                 own_dates: bool = False) -> WaveBundle:
    """跑 ``_single`` 的全套统计，再叠上纬向谱专项。

    ``own_dates`` 与 :func:`det_report.analyze` 同义：每个模型的谱统计用**它自己的
    全部起报日**，而不是只用共同日期。**配对检验（``zonal_tests`` / ``sph_tests``）
    不受影响**——日期不对齐的样本做不了 Wilcoxon，那两处永远只在共同日期上做。

    Raises:
        ValueError / FileNotFoundError: 见 :func:`det_report.analyze` 与
            :func:`det_report.load_spectrum`。
    """
    base = analyze(archives, own_dates=own_dates)
    names, dates = base.names, base.dates
    dates_by_model = base.dates_by_model
    variables = base.spectrum_variables

    def _pair(frame: pd.DataFrame) -> pd.DataFrame:
        """配对检验的样本：**永远退回共同日期**，且各模型都得有数。

        各模型用自己的日期时，并集帧上每个模型各有一片 NaN；Wilcoxon 要的是同一批
        日子上的成对样本，所以这里 reindex 到共同日期再丢掉有缺的行。默认口径下
        各模型索引本来就是共同日期，``dropna`` 不会删掉任何一行（除了一开始就
        缺数的那些），与旧行为一致。
        """
        frame = frame.reindex(index=[str(d) for d in dates])
        return frame.dropna(how="any") if own_dates else frame

    # ``spectra`` 存「波数 × 日期 平均后」的 pred/obs 均值谱（只够画图）；
    # ``raw`` 存逐日的 pred/obs 原表，log-RMS 必须逐日算，不能拿均值谱算。
    spectra: Dict[str, Dict[str, pd.DataFrame]] = {}
    raw: Dict[str, Dict[str, Tuple[pd.DataFrame, pd.DataFrame]]] = {}
    for archive in archives:
        # 每个模型读自己的那批日子；默认口径下 dates_by_model 就是共同日期，
        # 与旧行为逐位一致。
        own = list(dates_by_model[archive.name])
        means, originals = {}, {}
        for variable in variables:
            pred, obs = load_spectrum(archive.root, own, variable, archive.name)
            originals[variable] = (pred, obs)
            means[variable] = pd.DataFrame({"pred": pred.mean(axis=1),
                                            "obs": obs.mean(axis=1)})
        spectra[archive.name] = means
        raw[archive.name] = originals

    # --- 逐变量 → 逐日 log-RMS，再对变量平均 ---
    zonal_per_date_frames = {}
    variable_logrms, variable_matched = {}, {}
    for name in names:
        per_date = []
        full_values, matched_values = {}, {}
        for variable in variables:
            pred, obs = raw[name][variable]
            full = log_rms_ln(pred, obs, 1, FULL_K_MAX)
            matched = log_rms_ln(pred, obs, 1, MATCHED_K_MAX)
            per_date.append(full)
            full_values[variable] = float(full.mean()) if not full.empty else float("nan")
            matched_values[variable] = float(matched.mean()) if not matched.empty else float("nan")
        variable_logrms[name] = pd.Series(full_values)
        variable_matched[name] = pd.Series(matched_values)
        zonal_per_date_frames[name] = pd.concat(per_date, axis=1).mean(axis=1) if per_date else pd.Series(dtype=float)

    zonal_per_date = pd.DataFrame(zonal_per_date_frames)
    if not own_dates:
        # 共同日期口径下各模型索引本来相同，这次 reindex 是保单（顺序 + 缺行）。
        zonal_per_date = zonal_per_date.reindex(index=[str(d) for d in dates])
    variable_logrms = pd.DataFrame(variable_logrms).reindex(index=variables)
    variable_matched = pd.DataFrame(variable_matched).reindex(index=variables)
    zonal_full = variable_logrms.mean(axis=0)
    zonal_matched = variable_matched.mean(axis=0)

    # --- 五个波数带 ---
    band_rows_logrms, band_rows_ratio = {}, {}
    for name in names:
        logrms_row, ratio_row = {}, {}
        for low, high, label in WAVE_BANDS:
            band_name = f"{label} {low}–{high}"
            logs, ratios = [], []
            for variable in variables:
                pred, obs = raw[name][variable]
                series = log_rms_ln(pred, obs, low, high)
                if not series.empty:
                    logs.append(float(series.mean()))
                ratio = band_power_ratio(pred, obs, low, high)
                if np.isfinite(ratio):
                    ratios.append(ratio)
            logrms_row[band_name] = float(np.mean(logs)) if logs else float("nan")
            ratio_row[band_name] = float(np.mean(ratios)) if ratios else float("nan")
        band_rows_logrms[name] = pd.Series(logrms_row)
        band_rows_ratio[name] = pd.Series(ratio_row)

    band_order = [f"{label} {low}–{high}" for low, high, label in WAVE_BANDS]
    band_logrms = pd.DataFrame(band_rows_logrms).reindex(index=band_order)
    band_ratio = pd.DataFrame(band_rows_ratio).reindex(index=band_order)

    zonal_tests = paired_tests(_pair(zonal_per_date))
    primary, reference = _pick_primary(names)

    # --- 球谐带功率（没有结果、或结果缺 group 列认不出频带时，整块留空、正文标注缺席）---
    band_names = [_band_label(lo, hi) for lo, hi in SPHERICAL_BANDS]
    sph_available = all(
        probe_bands(a.root, dates_by_model[a.name], variables) for a in archives
    )
    sph_tests: Dict[str, List[PairTest]] = {}
    sph_rms: Dict[str, float] = {}
    sph_bias: Dict[str, float] = {}
    sph_band_logrms = pd.DataFrame(index=band_names)
    sph_band_ratio = pd.DataFrame(index=band_names)
    sph_variable = pd.DataFrame(index=variables)
    sph_coverage: Dict[str, Dict[str, Tuple[int, int]]] = {}
    if sph_available:
        logrms_rows, ratio_rows, variable_rows = {}, {}, {}
        daily: Dict[str, Dict[str, pd.Series]] = {}
        for archive in archives:
            # logs/ratios 是**逐样本**池（骨架 §2.1 要总均方根，不是日均值的均方根）；
            # per_variable 按变量分池，daily 按频带留日度样本供 §5.3 配对检验。
            logs: Dict[str, List[np.ndarray]] = {}
            ratios: Dict[str, List[np.ndarray]] = {}
            per_variable: Dict[str, List[np.ndarray]] = {}
            daily_parts: Dict[str, List[pd.Series]] = {b: [] for b in band_names}
            wanted = list(dates_by_model[archive.name])
            coverage: Dict[str, Tuple[int, int]] = {}
            for variable in variables:
                # 有多少算多少：只拿这个变量**真的有球谐带**的起报日。
                used = band_dates(archive.root, wanted, variable)
                coverage[variable] = (len(used), len(wanted))
                if not used:
                    # 这个变量在这份产物里一天都没有：不参与统计，表里该格留空，
                    # 缺多少由正文的覆盖说明交代。
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
                b: _pooled_logrms(logs.get(b, [])) for b in band_names}
            ratio_rows[archive.name] = {
                b: _mean_ratio(ratios.get(b, [])) for b in band_names}
            variable_rows[archive.name] = {
                v: _pooled_logrms(s) for v, s in per_variable.items()}
            sph_rms[archive.name] = _pooled_logrms(
                [a for b in band_names for a in logs.get(b, [])])
            sph_bias[archive.name] = _bias_pct(
                [a for b in band_names for a in ratios.get(b, [])])
            daily[archive.name] = {
                b: (pd.concat(daily_parts[b], axis=1).mean(axis=1)
                    if daily_parts[b] else pd.Series(dtype=float))
                for b in band_names}
            sph_coverage[archive.name] = coverage
        for label in band_names:      # 同一频带内跨模型配对；指标即频带，故逐频带各做一轮
            frame = pd.DataFrame({n: daily[n][label] for n in names})
            sph_tests[label] = paired_tests(_pair(frame))
        sph_band_logrms = pd.DataFrame(logrms_rows).reindex(index=band_names)
        sph_band_ratio = pd.DataFrame(ratio_rows).reindex(index=band_names)
        sph_variable = pd.DataFrame(variable_rows).reindex(index=variables)
    sph_rms = pd.Series({n: sph_rms.get(n, float("nan")) for n in names})
    sph_bias = pd.Series({n: sph_bias.get(n, float("nan")) for n in names})

    return WaveBundle(
        base=base, spectra=spectra,
        zonal_full=zonal_full, zonal_matched=zonal_matched,
        zonal_per_date=zonal_per_date,
        band_logrms=band_logrms, band_ratio=band_ratio,
        variable_logrms=variable_logrms, variable_matched=variable_matched,
        zonal_tests=zonal_tests, primary=primary, reference=reference,
        sph_available=sph_available, sph_rms=sph_rms, sph_bias=sph_bias,
        sph_band_logrms=sph_band_logrms, sph_band_ratio=sph_band_ratio,
        sph_variable=sph_variable, sph_tests=sph_tests,
        sph_coverage=sph_coverage,
    )


# ====================================================================== 出图


def _save_figure(figure, path: Path) -> Path:
    """落盘一张图。字体设置与保存约定跟 :mod:`visualization.det_plots` 对齐。"""
    plt.rcParams["figure.dpi"] = 150
    plt.rcParams["savefig.dpi"] = 150
    plt.rcParams["savefig.bbox"] = "tight"
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path)
    plt.close(figure)
    print(f"已保存: {path}")
    return path


def _bar_panels(out_dir: Path, filename: str, panels: Sequence[Tuple[str, pd.Series]],
                *, suptitle: str, ylabel: str = "") -> Path:
    """一排柱状子图：每个面板一个指标，柱子是各模型。"""
    apply_style()
    columns = len(panels)
    width = snap_width(3.2 * columns)
    sizes = font_sizes(width)
    figure, axes = plt.subplots(
        1, columns, figsize=(width, 4.0), squeeze=False
    )
    axes = axes.ravel()
    for axis, (title, series) in zip(axes, panels):
        series = series.dropna()
        colors = model_colors(list(series.index))
        axis.bar([str(i) for i in series.index], series.values,
                 color=[colors[str(i)] for i in series.index], width=0.62)
        axis.set_title(title, fontsize=sizes["title"])
        axis.set_ylabel(ylabel or title, fontsize=sizes["label"])
        axis.grid(axis="y", linewidth=0.6)
        axis.tick_params(labelsize=sizes["tick"])
        for tick in axis.get_xticklabels():
            tick.set_rotation(30)
            tick.set_ha("right")
    figure.suptitle(suptitle, fontsize=sizes["suptitle"])
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    return _save_figure(figure, Path(out_dir) / filename)


def _grouped_bars(out_dir: Path, filename: str, frame: pd.DataFrame, *,
                  suptitle: Optional[str], ylabel: str,
                  ref_line: Optional[float] = None,
                  y_grid: bool = True) -> Path:
    """分组柱状图：index 是分组，columns 是模型。

    ``suptitle=None`` 时不画顶部标题；``y_grid=False`` 时去掉横向参考线，
    只留柱子和坐标轴（集合报告的球谐带图用这一档）。图例一律横向排在整张图
    底部（``LEGEND_FRAME`` 只描灰色细边），作为图的一部分，不占柱子所在的
    面板。
    """
    apply_style()
    groups = [str(i) for i in frame.index]
    models = [str(c) for c in frame.columns]
    colors = model_colors(models)
    bar_width = 0.8 / max(1, len(models))
    positions = np.arange(len(groups))
    width = snap_width(max(8.0, 1.7 * len(groups)))
    sizes = font_sizes(width)
    figure, axis = plt.subplots(figsize=(width, 5.2))
    for offset, model in enumerate(models):
        values = pd.to_numeric(frame[model], errors="coerce").values
        axis.bar(positions + offset * bar_width - 0.4 + bar_width / 2, values,
                 width=bar_width, color=colors[model], label=model)
    if ref_line is not None:
        axis.axhline(ref_line, color=NEUTRAL["reference"], linestyle="--", linewidth=1.0)
    axis.set_xticks(positions)
    axis.set_xticklabels(groups, fontsize=sizes["tick"])
    axis.set_ylabel(ylabel, fontsize=sizes["label"])
    axis.tick_params(axis="y", labelsize=sizes["tick"])
    if y_grid:
        axis.grid(axis="y", linewidth=0.6)
    else:
        axis.grid(False)
    if suptitle:
        axis.set_title(suptitle, fontsize=sizes["suptitle"])
    # 图例横排在整图底部（不在柱子上方压数据），只描灰色细边。一行放不下
    # 就折成多行，并按行数多留一点底部空间。
    handles, labels = axis.get_legend_handles_labels()
    if handles:
        ncols = min(len(labels), 6)
        rows = max(1, (len(labels) + ncols - 1) // ncols)
        legend = figure.legend(
            handles, labels, loc="lower center", ncols=ncols,
            fontsize=sizes["legend"], handlelength=1.8, columnspacing=1.6,
            labelspacing=0.5, borderaxespad=0.2, **LEGEND_FRAME,
        )
        legend.get_frame().set_linewidth(0.8)
        figure.tight_layout(rect=(0.0, 0.06 if rows == 1 else 0.13, 1.0, 1.0))
    else:
        figure.tight_layout()
    return _save_figure(figure, Path(out_dir) / filename)


def _plot_mean_spectrum(out_dir: Path, filename: str, spectra: Mapping[str, pd.DataFrame], *,
                        variable: str, suptitle: str) -> Path:
    """平均功率谱：双对数，各模型预测谱 + 一条共享观测谱。"""
    plotter = DetPlotter()
    path = Path(out_dir) / filename
    plotter.plot_model_spectrum(
        dict(spectra), variable, suptitle=suptitle, save_path=path,
    )
    plt.close("all")
    return path


def _plot_spectrum_ratio(out_dir: Path, filename: str, spectra: Mapping[str, pd.DataFrame], *,
                         variable: str, suptitle: str) -> Path:
    """pred/obs 比值曲线，纵轴限定 0.1–10。"""
    apply_style()
    colors = model_colors(list(spectra))
    width = snap_width(9)
    sizes = font_sizes(width)
    figure, axis = plt.subplots(figsize=(width, 5.5))
    for name, frame in spectra.items():
        data = frame.copy()
        data.index = pd.to_numeric(data.index, errors="coerce")
        data = data[(data.index > 0)].sort_index()
        ratio = (data["pred"] / data["obs"]).replace([np.inf, -np.inf], np.nan).dropna()
        ratio = ratio[ratio > 0]
        if ratio.empty:
            continue
        axis.plot(ratio.index, ratio.values, color=colors[name], linewidth=1.6, label=name)
    axis.axhline(1.0, color=NEUTRAL["observation"], linestyle="--", linewidth=1.2,
                 label="理想值 = 1")
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_ylim(0.1, 10.0)
    axis.set_xlabel("纬向波数 k", fontsize=sizes["label"])
    axis.set_ylabel("pred / obs", fontsize=sizes["label"])
    axis.tick_params(labelsize=sizes["tick"])
    axis.grid(which="both", linewidth=0.6)
    axis.legend(loc="best", fontsize=sizes["legend"])
    axis.set_title(suptitle, fontsize=sizes["suptitle"])
    figure.tight_layout()
    return _save_figure(figure, Path(out_dir) / filename)


# ====================================================================== 渲染






def _section_conclusion(bundle: WaveBundle) -> List[str]:
    base = bundle.base
    names = bundle.names
    order = base.relative_composite.sort_values().index.tolist()
    best = order[0]
    zonal_best = bundle.zonal_full.idxmin()
    zonal_worst = bundle.zonal_full.idxmax()

    # 补充报告：精度与幅度只在第 3 节给基线值，结论段不重复主报告的判断，
    # 只讲这一份独有的两套谱。
    items = [
        f"{cn_number(len(names))}个产物完整性检查通过，全部模型 `n_ok = n_dates`，"
        f"无失败起报点；综合相对 RMSE、ACC、FA 只在第 3 节列基线值，"
        f"明细与验收建议见原输出级评估报告。",
        f"纬向 FFT 谱（全波数 1–{FULL_K_MAX}）log-RMS×100 最小 {zonal_best}"
        f"（{fmt_number(bundle.zonal_full[zonal_best], 2)}），最大 {zonal_worst}"
        f"（{fmt_number(bundle.zonal_full[zonal_worst], 2)}）；"
        f"匹配尺度 1–{MATCHED_K_MAX} 口径下最小 {bundle.zonal_matched.idxmin()}"
        f"（{fmt_number(bundle.zonal_matched.min(), 2)}）。",
    ]
    if bundle.sph_available:
        items.append(
            f"球谐带功率谱（{cn_number(SPHERICAL_BAND_COUNT)}个球谐总阶数频带）"
            f"log-RMS×100 最小 {bundle.sph_rms.idxmin()}"
            f"（{fmt_number(bundle.sph_rms.min(), 2)}），最大 {bundle.sph_rms.idxmax()}"
            f"（{fmt_number(bundle.sph_rms.max(), 2)}）；比偏差最小 {bundle.sph_bias.idxmin()}"
            f"（{fmt_number(bundle.sph_bias.min(), 2)}%），最大 {bundle.sph_bias.idxmax()}"
            f"（{fmt_number(bundle.sph_bias.max(), 2)}%）。"
        )
    else:
        items.append(f"球谐带功率谱：{SPHERICAL_ABSENT}")
    if bundle.primary and bundle.reference:
        primary, reference = bundle.primary, bundle.reference
        delta = float(bundle.zonal_full[primary]) - float(bundle.zonal_full[reference])
        direction = "改善" if delta < 0 else "变差"
        items.append(
            f"主模型 {primary} 相对 {reference}：纬向谱 log-RMS×100 "
            f"{fmt_number(bundle.zonal_full[reference], 2)} → {fmt_number(bundle.zonal_full[primary], 2)}，"
            f"{direction} {fmt_number(abs(delta), 2)}；综合相对 RMSE "
            f"{fmt_number(base.relative_composite[reference])} → {fmt_number(base.relative_composite[primary])}。"
        )
    if own_dates_mode(bundle):
        # 谱比随季节、随天气过程变化明显，各模型又取的是不同时段——这段差异里
        # 有多少是模式差异、多少是样本时段差异，本报告分不开。必须写在摘要里。
        items.append(
            f"日期口径：每个模型都用各自全部起报日"
            f"（{date_scope_counts(bundle)}），谱与 log-RMS 的均值来自**不同时段**，"
            f"跨模型的相对位次因此同时包含模式差异与时段差异，"
            f"**不是同一批日期上的公平对比**；只有配对检验"
            f"退回 {len(bundle.dates)} 个共同日期（{date_span(bundle.dates)}）。"
        )
    items.append(
        f"选型取向：谱保真度优先看纬向谱（{zonal_best}）"
        + (f"与球谐带（{bundle.sph_rms.idxmin()}）" if bundle.sph_available else "")
        + f"，与精度类指标（综合相对 RMSE 最低 {best}）不一定指向同一模型。"
    )
    return [
        "# 1. 结论摘要",
        "",
        # 先交代评估基本信息（评的是谁、多少天、什么指标），再给结论
        f"本报告评估{'、'.join(sorted(names))}共{cn_number(len(names))}个模型，"
        + (
            f"覆盖{len(bundle.dates)}天（{date_span(bundle.dates)}）；"
            if not own_dates_mode(bundle)
            else f"覆盖时段不等长（{date_scope_counts(bundle)}）；"
        )
        + f"口径为纬向 FFT 与球谐带功率谱，RMSE / ACC / FA 只作基线。"
        + (
            "本报告是补充分析，不修订、也不替代原输出级评估检验报告。"
            if not own_dates_mode(bundle)
            else "本报告中每个模型都用自己的全部起报日——各自的平均、各自的天数，"
            "各表各图并不是在同一批日期上对比（详见第 2 节）；"
            "各处的配对检验退回共同日期，那是全报告唯一公平的一处。"
            "本报告是补充分析，不修订、也不替代原输出级评估检验报告。"
        ),
        "",
        # 结论并成一段而不是逐条列点：读起来是判断，不是清单
        "".join(items),
        "",
    ]


def _section_scope(bundle: WaveBundle) -> List[str]:
    base = bundle.base
    names = bundle.names
    lines = ["# 2. 数据范围与检验口径", ""]

    rows = []
    nan_notes: List[str] = []
    for name in sorted(names):
        archive = base.archives[name]
        summary = archive.summary
        own_dates = sorted(set(summary["init_date"]))
        shared = len(set(own_dates) & set(bundle.dates))
        # 本次真正进了统计的那几天：默认口径下就是共同日期，各用各的时是自己的全部。
        used = list(bundle.dates_by_model[archive.name])
        rmse_columns = [c for c in summary.columns if c.startswith("rmse_")]
        # 缺文件按 stem 实探，不看 summary 有没有行——两者可能对不上。
        # 探的是**本次实际用的那几天**，不是共同日期：各用各的时两者不是一批。
        missing_dates = probe_dates(archive.root, used, "rmse")
        missing = "无" if not missing_dates else f"{len(missing_dates)} 天：{missing_dates[0]} 等"
        nan_mask = summary[rmse_columns].isna().any(axis=1) if rmse_columns else None
        nan_all = summary.loc[nan_mask, "init_date"].tolist() if nan_mask is not None else []
        # 这一列统计的是**该模型自己的全部日期**，不是共同日期——不写清楚会被读成
        # 「共同 168 天里有 15 天是 NaN」，那是另一回事。窗口按本次实际用的那几天算。
        nan_outside = [d for d in nan_all if d not in set(used)]
        # 注意这里要的是**窗口之内**的天数。nan_outside 是窗口之外的那一批，
        # 拿它的长度去填「落在窗口之内」的句子会整体说反：全是窗口外的日期时
        # 反而报成「其中 15 天落在共同窗口之内」。
        nan_inside = len(nan_all) - len(nan_outside)
        # 「窗口」指本次统计实际用的那几天。默认口径下就是共同日期，各用各的时
        # 是该模型自己的全部起报日——这时 NaN 天必然全在窗口内，正是要说清的事。
        window = "共同窗口" if not own_dates_mode(bundle) else "本次统计窗口"
        rows.append([
            name, len(own_dates), date_span(own_dates), shared,
            missing, f"{len(nan_all)}（全区间）" if nan_all else "无",
        ])
        if nan_all:
            nan_notes.append(
                f"{name} 的 {len(nan_all)} 天 RMSE 为 NaN"
                + (f"（{nan_all[0]}–{nan_all[-1]}）" if len(nan_all) > 1 else f"（{nan_all[0]}）")
                + (f"全部落在{window}之外，不影响本报告任何统计"
                   if not nan_inside else
                   f"，其中 {nan_inside} 天落在{window}**之内**，需先查明原因")
                + "。"
            )
    lines += table_lines(
        ["**模型**", "**日期数**", "**日期范围**", "**共同日期**", "**核心文件缺失**",
         "**RMSE NaN（全区间）**"],
        rows,
    )
    if nan_notes:
        lines += [""] + nan_notes

    first, last, leads = lead_span(base.rmse_by_lead[names[0]].index)
    variables = base.common_variables
    fa_variables = base.fa_variables
    spectrum_variables = bundle.spectrum_variables
    lines += [
        "",
        (
            f"所有统计均使用{cn_number(len(names))}个模型的共同{len(bundle.dates)}天和"
            f"{leads}个 lead（{first}–{last}h）。RMSE 使用{cn_number(len(variables))}个共同变量；"
            f"FA 使用所有模型都提供的{cn_number(len(fa_variables))}个变量："
            f"{'、'.join(fa_variables)}；功率谱使用{cn_number(len(spectrum_variables))}个变量："
            f"{'、'.join(spectrum_variables)}。全部{cn_number(len(names))}个模型在整个共同日期区间上"
            f"都有完整结果，无缺席模型。"
            if not own_dates_mode(bundle) else
            f"所有统计均按**每个模型自己的全部起报日**做：{date_scope_counts(bundle)}，"
            f"lead 一律 {leads} 个（{first}–{last}h）。RMSE 使用{cn_number(len(variables))}个共同变量；"
            f"FA 使用所有模型都提供的{cn_number(len(fa_variables))}个变量："
            f"{'、'.join(fa_variables)}；功率谱使用{cn_number(len(spectrum_variables))}个变量："
            f"{'、'.join(spectrum_variables)}。各表各图的天数不同、时段也不同，"
            f"**上表的「日期数」就是该模型本次实际用的天数**，"
            f"「共同日期」那一列才是各模型共有的、也就是下面配对检验能用的天数。"
        ),
        "",
        "## 2.1 指标定义",
        "",
    ]
    lines += table_lines(
        ["**指标**", "**计算方式**", "**方向**"],
        [
            ["综合相对 RMSE",
             f"每个日期、每个变量先除以{cn_number(len(names))}模型几何均值，再对{cn_number(len(variables))}个变量取几何平均；100 为{cn_number(len(names))}模型基线。",
             "越低越好"],
            ["ACC",
             f"{base.acc_variable} 每个日期{leads}个 lead 的平均 ACC，再对所有日期取平均。",
             "越高越好"],
            ["FA 偏差",
             f"{cn_number(len(fa_variables))}个变量、全部 lead 的 \\|pred/obs−1\\|，再按日期和要素平均，单位 pp。",
             "越低越好"],
            ["纬向谱 log-RMS×100",
             f"对{cn_number(len(spectrum_variables))}个变量的纬向 FFT 谱计算 RMS[ln(pred/obs)]×100；"
             f"报告同时给出全波数 1–{FULL_K_MAX} 和匹配尺度 1–{MATCHED_K_MAX}。",
             "越低越好"],
            ["球谐带 log-RMS×100",
             f"{cn_number(SPHERICAL_BAND_COUNT)}个球谐总阶数频带上计算 RMS[ln(pred/obs)]×100，是所有日期、变量、lead 的总均方根。",
             "越低越好；球谐带补充谱指标" if bundle.sph_available else "越低越好；本批未出"],
            ["球谐带比偏差",
             "所有频带、变量、lead 与日期上的 mean \\|pred/obs−1\\|，单位 %。",
             "越低越好" if bundle.sph_available else "越低越好；本批未出"],
        ],
    )
    pairs = len(names) * (len(names) - 1) // 2
    lines += [
        "",
        f"配对比较采用{len(bundle.dates)}个日度样本的 Wilcoxon 符号秩检验，"
        f"并对同一指标族的{pairs}个模型对做 Holm 多重校正。统计显著不代表业务重要性。"
        + ("" if not own_dates_mode(bundle) else
           f"**配对检验是全报告唯一在共同日期上做的比较**："
           f"配对要求两边在同一天有数，各模型自己的日期对不齐，"
           f"所以上面各表用各模型全部起报日、这里的样本却只有各模型共有的 "
           f"{len(bundle.dates)} 天，两者不是同一批，"
           f"表里的均值也**不会**等于检验时的均值。"),
        "",
    ]
    return lines


def _section_overall(bundle: WaveBundle) -> List[str]:
    base = bundle.base
    order = base.relative_composite.sort_values().index.tolist()
    lines = ["# 3. 总体结果", ""]
    rows = []
    for name in order:
        rows.append([
            name, fmt_number(base.relative_composite[name]), fmt_number(base.acc_mean[name]),
            fmt_number(base.fa_bias[name], 4), fmt_number(bundle.zonal_full[name], 2),
            fmt_number(bundle.sph_rms[name], 2) if bundle.sph_available else "—",
            fmt_number(bundle.sph_bias[name], 2) if bundle.sph_available else "—",
        ])
    lines += table_lines(
        ["**模型**", "**综合相对RMSE**", "**ACC (%)**", "**FA偏差 (pp)**",
         "**纬向谱 RMS×100**", "**球谐谱 RMS×100**", "**球谐比偏差 (%)**"],
        rows,
    )
    lines += ["", "相关图：", "", "![overall_summary](overall_summary.png)", "",
              "图 1：{模型数}模型的综合相对 RMSE、ACC、FA 偏差与纬向谱 log-RMS×100 对比"
              "（{日期口径}平均，柱高越低越好，ACC 除外）。"
              .replace("{模型数}", cn_number(len(bundle.names)))
              .replace("{日期口径}", date_scope_word(bundle)), ""]

    return lines


def _spherical_bands_section(bundle: WaveBundle) -> List[str]:
    """§4.1 球谐带分频结果：频带 × 模型，logRMS 与均值比各一组。"""
    names = bundle.names
    lines = ["## 4.1 球谐带分频结果", ""]
    if not bundle.sph_available:
        lines += [f"{SPHERICAL_ABSENT}因此本节无表、无图，保留节位以便将来补数后直接填入。", ""]
        return lines
    # 列序与 §4.5 一致：每个模型 logRMS / 均值比 两列相邻，「最佳」收尾
    header = ["**频带**"]
    for name in names:
        header += [f"**{name} logRMS**", f"**{name} 均值比**"]
    header += ["**最佳**"]
    rows = []
    for band in bundle.sph_band_logrms.index:
        logrms = bundle.sph_band_logrms.loc[band]
        ratios = bundle.sph_band_ratio.loc[band]
        rows.append([band] + [v for name in names
                              for v in (fmt_number(logrms[name], 2), fmt_number(ratios[name], 4))]
                    + [logrms.idxmin()])
    lines += table_lines(header, rows)
    band_list = list(bundle.sph_band_logrms.index)
    lines += [
        "",
        f"球谐带 log-RMS×100 最低的模型在"
        + "、".join(f"{band} 段是 {bundle.sph_band_logrms.loc[band].idxmin()}"
                    for band in band_list)
        + "；均值比是该频带内 pred/obs 的总平均，越接近 1 表示该尺度上的能量越不偏。"
        f"最高频带（{band_list[-1]}）的总阶数最高、观测功率最小，比值的对数对微小差异最敏感，"
        "该频带的 log-RMS 绝对值不宜与低频带直接比较——这是频谱类指标的固有特征，"
        "不是模型在该尺度上一定更差；均值比在同一频带仍有意义，因为它不依赖逐点的分母。"
        + f"按频带看，log-RMS 最低的模型分别是"
        + "、".join(
            f"{name} 拿下 {sum(1 for band in band_list if bundle.sph_band_logrms.loc[band].idxmin() == name)} 个频带"
            for name in names)
        + "。",
        "",
        "![spherical_bands_logrms](spherical_bands_logrms.png)",
        "",
        f"图 2：{cn_number(len(names))}模型分球谐总阶数频带（{' / '.join(band_list)}）"
        f"的球谐带 log-RMS×100 对比。",
        "",
        "![spherical_bands_ratio](spherical_bands_ratio.png)",
        "",
        f"图 3：{cn_number(len(names))}模型分球谐总阶数频带的 pred/obs 均值比对比（虚线为理想值 1）。",
        "",
    ]
    return lines


def _spherical_variables_section(bundle: WaveBundle) -> List[str]:
    """§4.2 球谐带分变量结果：变量 × 模型。"""
    names = bundle.names
    lines = ["## 4.2 球谐带分变量结果", ""]
    if not bundle.sph_available:
        lines += [f"{SPHERICAL_ABSENT}本节同 4.1，一并留空。", ""]
        return lines
    variables = sorted(bundle.sph_variable.index)
    rows = []
    for variable in variables:
        row = bundle.sph_variable.loc[variable]
        rows.append([variable] + [fmt_number(row[n], 2) for n in names] + [row.idxmin()])
    lines += table_lines(["**变量**"] + [f"**{n}**" for n in names] + ["**最佳**"], rows)
    by_variable = bundle.sph_variable.mean(axis=0)
    by_band = bundle.sph_band_logrms.mean(axis=0)
    order_variable = by_variable.sort_values().index.tolist()
    order_band = by_band.sort_values().index.tolist()
    lines += [
        "",
        f"逐变量的球谐带 log-RMS×100 排序为 "
        + " < ".join(f"{n}（{fmt_number(by_variable[n], 2)}）" for n in order_variable)
        + f"，按 §4.1 的频带口径汇总为 "
        + " < ".join(f"{n}（{fmt_number(by_band[n], 2)}）" for n in order_band)
        + ("，两者一致。" if order_variable == order_band else
           "，**两者不一致**。")
        + "两者用的是同一批带功率样本，只是聚合维度不同（这里按变量汇总全部频带，"
        "§4.1 按频带汇总全部变量），排序不一致时说明模型间的优劣集中在个别频带上，"
        "选型时要指明关注哪一段尺度。",
        "",
    ]
    return lines


def _spherical_tests_section(bundle: WaveBundle) -> List[str]:
    """§4.3 球谐带配对检验：指标（频带）× 模型对。"""
    names = bundle.names
    lines = ["## 4.3 球谐带配对检验", ""]
    if not bundle.sph_available:
        lines += [f"{SPHERICAL_ABSENT}本节同 4.1，一并留空。", ""]
        return lines
    rows = []
    for band in bundle.sph_band_logrms.index:
        # 骨架要求「先按指标出现顺序、再按模型对字典序」，paired_tests 按列序出，
        # 这里补一次字典序排序
        for test in sorted(bundle.sph_tests.get(band, []), key=lambda t: (t.a, t.b)):
            rows.append([
                band, f"{test.a} vs {test.b}", _signed(test.delta_pct),
                fmt_number(test.p_holm, 4), "是" if test.significant else "否", test.better,
            ])
    lines += table_lines(
        ["**指标**", "**模型对**", "**A相对B**", "**Holm p**", "**显著**", "**胜者**"],
        rows,
    )
    significant = [r for r in rows if r[4] == "是"]
    pair_count = len(rows) // max(1, len(bundle.sph_band_logrms.index))
    lines += [
        "",
        f"逐频带在{len(bundle.dates)}个日度样本上做 Wilcoxon 符号秩检验，"
        f"每个频带各自对{pair_count}个模型对做 Holm 校正（指标是频带本身，"
        "同一频带内的模型对构成一个检验族）。"
        + (f"全部 {len(rows)} 个「频带 × 模型对」组合里有 {len(significant)} 个达到显著，"
           "其余组合的差异被样本量或频带内的离散度淹没；"
           if significant else
           f"全部 {len(rows)} 个「频带 × 模型对」组合均未达显著，"
           "说明各模型在这些总阶数频带上的差异小于日度波动；")
        + "统计显著不代表业务重要性。",
        "",
    ]
    return lines


def _section_spectrum(bundle: WaveBundle, figures: Mapping[str, str]) -> List[str]:
    base = bundle.base
    names = bundle.names
    variables = bundle.spectrum_variables
    lines = [
        "# 4. 功率谱检验",
        "",
        f"产物保留两套功率谱定义：纬向 FFT 谱（`diagnostics/spectrum_by_init.csv`）"
        f"和球谐带功率（`scores.csv` 里 `spherical_bands` 的结果行）。"
        f"两者都只使用功率或功率谱，不包含相位；本报告分别给出完整结果，不用一套指标替代另一套。"
        + ("" if bundle.sph_available else SPHERICAL_ABSENT),
        "",
    ]
    coverage = sph_coverage_notes(bundle)
    if coverage:
        lines += coverage + [""]
    lines += _spherical_bands_section(bundle)
    lines += _spherical_variables_section(bundle)
    lines += _spherical_tests_section(bundle)
    lines += [
        f"## 4.4 纬向 FFT 功率谱（完整）",
        "",
        f"纬向 FFT 对每个纬圈做一维 rFFT，经度方向波数 k=1..{FULL_K_MAX}，k=0 的纬向平均已置零。"
        f"纬向谱与球谐谱不是同一种变换：纬向谱保留完整 1–{FULL_K_MAX} 波数，"
        f"球谐带只汇总到总阶数 {MATCHED_K_MAX}。为避免把超高频近零观测功率的影响与"
        f"可训练频带混为一谈，本节同时给出全波数和 1–{MATCHED_K_MAX} 匹配尺度结果。",
        "",
    ]
    rows = [
        [f"全波数 1–{FULL_K_MAX}"] + [fmt_number(bundle.zonal_full[n], 2) for n in names]
        + [bundle.zonal_full.idxmin(), "包含极高波数；受近零观测功率与数值端点影响较大"],
        [f"匹配尺度 1–{MATCHED_K_MAX}"] + [fmt_number(bundle.zonal_matched[n], 2) for n in names]
        + [bundle.zonal_matched.idxmin(), "与球谐损失的最高阶数同范围，适合与球谐带对照"],
    ]
    lines += table_lines(
        ["**口径**"] + [f"**{n}**" for n in names] + ["**最佳**", "**解释**"], rows,
    )

    lines += ["", f"## 4.5 {cn_number(len(WAVE_BANDS))}个波数带结果", ""]
    header = ["**波数带**"]
    for name in names:
        header += [f"**{name} logRMS**", f"**{name} 能量比**"]
    header += ["**最佳**"]
    rows = []
    for band in bundle.band_logrms.index:
        logrms = bundle.band_logrms.loc[band]
        ratios = bundle.band_ratio.loc[band]
        row = [band] + [v for name in names
                        for v in (fmt_number(logrms[name], 2), fmt_number(ratios[name], 4))]
        rows.append(row + [logrms.idxmin()])
    lines += table_lines(header, rows)
    lines += [
        "",
        f"波数带按行星尺度到小尺度划分。log-RMS 最低的模型在"
        + "、".join(
            f"{band.split()[0]}段是 {bundle.band_logrms.loc[band].idxmin()}"
            for band in bundle.band_logrms.index
        )
        + "；能量比为该波数带内预测谱与观测谱的平均功率之比（先沿波数取平均再相除），"
        "越接近 1 表示该尺度上的能量越不偏。"
        "高波数段（"
        + bundle.band_logrms.index[-1]
        + "）的 log-RMS 普遍高于低波数段，是纬向谱指标的固有特征："
        "高波数观测功率接近零，比值的对数发散，这一段的绝对值不宜与低波数段直接比较；"
        "能量比在同一段仍有意义，因为它不依赖逐波数的分母。",
        "",
        "![zonal_spectrum_bands](zonal_spectrum_bands.png)",
        "",
        f"图 4：{cn_number(len(names))}模型分波数带（{' / '.join(b.split()[0] for b in bundle.band_logrms.index)}）"
        f"的纬向谱 log-RMS×100 对比。",
        "",
        "## 4.6 平均功率谱与比值曲线",
        "",
        f"下列曲线是{date_scope_word(bundle)}平均出来的日度功率谱。黑线为 ERA5，"
        f"彩色线为{cn_number(len(names))}个模型的预测谱；比值为 pred/obs，理想值为 1，"
        f"图中断面显示范围限定为 0.1–10。变量取 {base.spectrum_variable}。"
        + ("" if not own_dates_mode(bundle) else
           "**各模型平均用的是各自那批日子**，所以曲线高低里同时含模式差异与"
           "时段差异，逐条曲线之间不是同一批样本。"),
        "",
        "![mean_power_spectrum](mean_power_spectrum.png)",
        "",
        f"图 5：{base.spectrum_variable} 纬向功率谱"
        f"（{date_scope_word(bundle)}平均，双对数坐标）。",
        "",
        "![spectrum_ratio](spectrum_ratio.png)",
        "",
        f"图 6：{base.spectrum_variable} 纬向谱 pred/obs 比值随波数的变化（纵轴限定 0.1–10）。",
        "",
        f"## 4.7 1–{MATCHED_K_MAX} 波数逐变量结果",
        "",
    ]
    rows = []
    for variable in sorted(variables):
        row = bundle.variable_matched.loc[variable]
        rows.append([variable] + [fmt_number(row[n], 2) for n in names] + [row.idxmin()])
    lines += table_lines(["**变量**"] + [f"**{n}**" for n in names] + ["**最佳**"], rows)

    full_rank = bundle.variable_logrms.mean(axis=0).sort_values().index.tolist()
    matched_rank = bundle.variable_matched.mean(axis=0).sort_values().index.tolist()
    lines += [
        "",
        f"全波数 1–{FULL_K_MAX} 口径下的模型排序为 {' < '.join(full_rank)}，"
        f"1–{MATCHED_K_MAX} 匹配尺度口径下为 {' < '.join(matched_rank)}，"
        + ("两者一致，说明高波数段没有改变模型间格局。"
           if full_rank == matched_rank else
           "两者不一致，说明高波数段（近零观测功率主导）会改变排序，"
           "涉及业务可训练尺度时应以匹配尺度口径为准。"),
        "",
    ]

    # --- 4.8 主模型专项（可选：只在显式指定了 primary / reference 时出） ---
    if bundle.primary and bundle.reference:
        lines += _section_primary(bundle)
    return lines


def _section_primary(bundle: WaveBundle) -> List[str]:
    """FGVP_pinpu 这类「相对某模型做了谱优化」的主模型专项对比。"""
    primary, reference = bundle.primary, bundle.reference
    base = bundle.base
    names = bundle.names
    others = [n for n in names if n not in (primary, reference)]

    lines = [f"## 4.8 {primary} 相对 {reference} 的频谱专项对比", "",
             f"本节为{primary}（主模型）与{reference}（对照）的逐项对比，"
             f"用于判断频谱优化是否落地。", ""]

    rows = []
    for band in bundle.band_logrms.index:
        before = float(bundle.band_logrms.loc[band, reference])
        after = float(bundle.band_logrms.loc[band, primary])
        delta = after - before
        rows.append([
            band, fmt_number(before, 2), fmt_number(after, 2),
            fmt_number(delta, 2),
            "改善" if delta < 0 else ("变差" if delta > 0 else "持平"),
        ])
    lines += table_lines(
        ["**波数带**", f"**{reference} logRMS**", f"**{primary} logRMS**",
         "**变化（负为好）**", "**判定**"], rows,
    )

    full_before = float(bundle.zonal_full[reference])
    full_after = float(bundle.zonal_full[primary])
    matched_before = float(bundle.zonal_matched[reference])
    matched_after = float(bundle.zonal_matched[primary])
    lines += [
        "",
        "**两种口径的总量**",
        "",
    ]
    lines += table_lines(
        ["**口径**", f"**{reference}**", f"**{primary}**", "**变化（负为好）**"],
        [
            [f"全波数 1–{FULL_K_MAX}", fmt_number(full_before, 2), fmt_number(full_after, 2),
             fmt_number(full_after - full_before, 2)],
            [f"匹配尺度 1–{MATCHED_K_MAX}", fmt_number(matched_before, 2), fmt_number(matched_after, 2),
             fmt_number(matched_after - matched_before, 2)],
            ["综合相对 RMSE", fmt_number(base.relative_composite[reference]),
             fmt_number(base.relative_composite[primary]),
             fmt_number(float(base.relative_composite[primary]) - float(base.relative_composite[reference]), 2)],
            ["ACC (%)", fmt_number(base.acc_mean[reference]), fmt_number(base.acc_mean[primary]),
             fmt_number(float(base.acc_mean[primary]) - float(base.acc_mean[reference]), 2)],
            ["FA 偏差 (pp)", fmt_number(base.fa_bias[reference], 4), fmt_number(base.fa_bias[primary], 4),
             fmt_number(float(base.fa_bias[primary]) - float(base.fa_bias[reference]), 4)],
        ],
    )

    test = next((t for t in bundle.zonal_tests
                 if {t.a, t.b} == {primary, reference}), None)
    if test is not None:
        lines += [
            "",
            f"**配对检验（纬向谱 log-RMS，逐日）**：Wilcoxon 符号秩检验 "
            f"p = {fmt_number(test.p_holm, 4)}（Holm 校正后），"
            + (f"显著优者为 {test.better}。" if test.significant else "未达显著。"),
        ]

    del others  # 只在上面挑对照模型时用过，这里不再需要
    lines += [
        "",
        f"在全部{cn_number(len(names))}个模型中，{primary} 的纬向谱 log-RMS×100 排名第 "
        f"{int(bundle.zonal_full.rank().loc[primary])}，{reference} 排名第 "
        f"{int(bundle.zonal_full.rank().loc[reference])}。",
        "",
    ]
    return lines


def _appendix(bundle: WaveBundle, figures: Mapping[str, str],
              archive: Optional[str], out_dir: Path) -> List[str]:
    base = bundle.base
    names = bundle.names
    first, last, leads = lead_span(base.rmse_by_lead[names[0]].index)
    try:
        step = float(np.diff(sorted(set(base.rmse_by_lead[names[0]].index)))[0])
    except (IndexError, ValueError):
        step = float("nan")

    total_bytes = 0
    for name in names:
        for path in base.archives[name].root.rglob("*"):
            if path.is_file():
                total_bytes += path.stat().st_size

    lines = ["# 附录：数据与复算证据", ""]
    lines += table_lines(
        ["**项目**", "**值**"],
        [
            ["产物目录", archive or "、".join(names)],
            ["产物大小", f"{total_bytes / 1024 / 1024:.1f} MB"],
            ["SHA-256", "—（多目录产物，未逐文件计算）"],
            ["模型", "、".join(names)],
            # 默认口径下保持老那一行的标签与格式不动；各用各的时没有「共同日期」
            # 可言，换成口径说明（天数清单）。
            (["共同日期", f"{len(bundle.dates)} 天，{date_span(bundle.dates)}"]
             if not own_dates_mode(bundle) else
             ["日期口径", date_scope_detail(bundle)]),
            ["lead", f"{leads} 个，{first}–{last}h，间隔 {step:g}h"],
            ["分析结果目录", str(out_dir)],
        ],
    )
    lines += [
        "",
        f"**本补充报告不修订、不替代原评估报告；全部结论仅来自用户指定产物目录中的"
        f"{cn_number(len(names))}个 single 输出"
        + ("，球谐带功率取自 scores.csv 里 `spherical_bands` 的结果行。**"
           if bundle.sph_available else
           "，产物里没有可用的球谐带功率结果，相关章节已标注「本批未出」。**"),
        "",
    ]
    # 骨架把可选块定在「附录那张表之后」，图件清单是渲染器自己加的一节，排在可选块后面。
    # 附 L（分纬度带）已随简化去掉——主报告的附 L 覆盖同一张表，本报告不再复制。
    lines += _season_block()
    lines += [
        "",
        "## 附：图件清单",
        "",
    ]
    for index, (key, filename) in enumerate(figures.items(), start=1):
        lines.append(f"{index}. `{filename}`（{key}）")
    lines.append("")
    return lines


def render_wave(bundle: WaveBundle, figures: Mapping[str, str], *,
                archive: Optional[str] = None, out_dir: Path = Path(".")) -> str:
    """把 :class:`WaveBundle` 渲染成骨架形状的完整 Markdown。"""
    names = bundle.names
    shown = sorted(names)
    lines: List[str] = []
    lines += [
        f"# 确定性预报{cn_number(len(names))}模型波谱评估报告",
        "",
        f"数据产物：{archive or '、'.join(names)}",
        "",
        f"报告日期：{report_date()}    模型：{'、'.join(shown)}    评估层级：输出级复核",
        "",
        f"**报告性质：补充分析。本报告不修订、不替代原输出级评估检验报告，"
        f"仅补充{cn_number(len(names))}个 single 输出的统一重算、纬向 FFT 和球谐带功率谱结果。**",
        "",
    ]
    for section in (
        _section_conclusion(bundle),
        _section_scope(bundle),
        _section_overall(bundle),
        _section_spectrum(bundle, figures),
        _appendix(bundle, figures, archive, out_dir),
    ):
        lines += section
        lines.append("")
    return "\n".join(lines)


# ====================================================================== 入口


def build_wave_report(archives: Sequence[Archive], out_dir: Path, *,
                      archive: Optional[str] = None,
                      own_dates: bool = False) -> Dict[str, Path]:
    """出图 + 写 ``REPORT.md``，返回产物名 → 路径。

    ``own_dates`` 见 :func:`analyze_wave`：每个模型按自己的全部起报日统计，
    配对检验仍退回共同日期。
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    bundle = analyze_wave(archives, own_dates=own_dates)
    names = bundle.names
    base = bundle.base

    # 图序即本字典的顺序。简化后只出谱图，编号由骨架的「图 N：」定死：
    #   1 总体柱状 → 2/3 球谐带（有球谐时才占号）→ 4 五波数带 → 5/6 平均谱与比值
    # 球谐缺席时 2/3 一并消失，后面整体前移——编号由渲染器现算，不写死。
    figures = {
        "overall_summary": "overall_summary.png",
        "zonal_spectrum_bands": "zonal_spectrum_bands.png",
        "mean_power_spectrum": "mean_power_spectrum.png",
        "spectrum_ratio": "spectrum_ratio.png",
    }
    if bundle.sph_available:
        figures = {
            "overall_summary": "overall_summary.png",
            "spherical_bands_logrms": "spherical_bands_logrms.png",
            "spherical_bands_ratio": "spherical_bands_ratio.png",
            "zonal_spectrum_bands": "zonal_spectrum_bands.png",
            "mean_power_spectrum": "mean_power_spectrum.png",
            "spectrum_ratio": "spectrum_ratio.png",
        }

    # 图 1：综合四指标柱状
    _bar_panels(
        out_dir, figures["overall_summary"],
        [("综合相对 RMSE", base.relative_composite),
         ("ACC (%)", base.acc_mean),
         ("FA 偏差 (pp)", base.fa_bias),
         ("纬向谱 RMS×100", bundle.zonal_full)],
        suptitle=f"{cn_number(len(names))}模型总体指标对比",
    )

    # 图 2 / 图 3：球谐带 log-RMS 与均值比
    if bundle.sph_available:
        _grouped_bars(
            out_dir, figures["spherical_bands_logrms"], bundle.sph_band_logrms.T,
            suptitle=f"{cn_number(len(names))}模型分球谐总阶数频带 log-RMS×100（越低越好）",
            ylabel="log-RMS×100",
        )
        _grouped_bars(
            out_dir, figures["spherical_bands_ratio"], bundle.sph_band_ratio.T,
            suptitle=f"{cn_number(len(names))}模型分球谐总阶数频带 pred/obs 均值比",
            ylabel="pred / obs", ref_line=1.0,
        )

    # 图 4：分波数带 log-RMS
    _grouped_bars(
        out_dir, figures["zonal_spectrum_bands"], bundle.band_logrms.T,
        suptitle=f"{cn_number(len(names))}模型分波数带纬向谱 log-RMS×100（越低越好）",
        ylabel="log-RMS×100",
    )

    # 图 5 / 图 6：平均功率谱与比值曲线（取优先变量）
    variable = base.spectrum_variable
    spectra = {name: frame[variable] for name, frame in bundle.spectra.items()}
    _plot_mean_spectrum(
        out_dir, figures["mean_power_spectrum"], spectra, variable=variable,
        suptitle=f"{variable} 纬向功率谱（{date_scope_word(bundle)}平均）",
    )
    _plot_spectrum_ratio(
        out_dir, figures["spectrum_ratio"], spectra, variable=variable,
        suptitle=f"{variable} 纬向谱 pred/obs 比值（理想值 1）",
    )

    # 简化后本报告只出谱图。热力图、单起报谱曲线、能力雷达图三张主报告的附录里
    # 已经出过，分纬度带整块随附 L 一起去掉，分时效段的 RMSE 图随 §4 RMSE 那节
    # 一起去掉——这里都不生成，免得留下没人引用的孤儿文件。
    text = render_wave(bundle, figures, archive=archive, out_dir=out_dir)
    report_path = out_dir / "REPORT.md"
    report_path.write_text(text, encoding="utf-8", newline="\n")
    print(f"已保存: {report_path}")

    artifacts: Dict[str, Path] = {"report": report_path}
    for key, filename in figures.items():
        artifacts[key] = out_dir / filename
    return artifacts
