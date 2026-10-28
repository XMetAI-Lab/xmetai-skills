#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""确定性多模型评估报告：加载评估产物 → 统计 → 渲染 Markdown。

输入是 ``output/store.py`` 写出的评估产物目录（每个模型一个）：``scores.csv``
长表（列见 ``output/table.py`` 的 ``SCORE_COLUMNS``）、``manifest.json``、
``diagnostics/spectrum_by_init.csv``。``analyze()`` 把长表透视回报告要的表，
指标定义与判读见 ``references/rmse-batch-evaluation.md`` §2。

取数两条硬规则：正文一律取 ``region`` 为空的全球行（:func:`_global`）；
``metric="rmse"`` 分确定性与集合展开两套，先按 ``product_kind`` 过滤
（:func:`_deterministic`）。正文与附录都用各模型共同的起报日。

章节骨架与集合报告一致：1 摘要 / 2 样本与数据质量 / 3 指标解释 / 4 总体结果
（4.1 RMSE、4.2 ACC、4.3 FA、4.4 纬向谱、4.5 球谐带、4.6 空间分布）/
5 分时效 / 6 统计检验 / 附录。显著性检验为逐日期配对 Wilcoxon + Holm。
"""

from __future__ import annotations

import json
import math
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, NamedTuple, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from reporting.core.common import (
    bullet_lines,
    cn_number,
    date_span,
    finite_or_none,
    fmt_number,
    lead_span,
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

try:
    from scipy.stats import wilcoxon as _scipy_wilcoxon
except ImportError:  # 推迟到真正做检验时报错，让"只出图"也能跑
    _scipy_wilcoxon = None


#: 几何均值基线：相对 RMSE 的 100 就是这个意思
BASELINE = 100.0

#: ACC / 谱图优先用的变量，缺了就退到字典序第一个
PREFERRED_VARIABLE = "z500"

#: ACC 在首时效高于这个百分数就算"饱和"，写进 6.3 的风险提示
SATURATED_ACC = 99.0

#: 分时效段的边界（左开右闭，单位 h）；空段不输出，段名按段内实际 lead 现写
LEAD_BANDS: Tuple[Tuple[float, float], ...] = (
    (0.0, 120.0),
    (120.0, 240.0),
    (240.0, 360.0),
)


# ====================================================================== 加载


#: 评估产物的固定文件名。新契约下**只认这一种**形状，不再有
#: ``summary.csv`` / ``batch_meta.json`` / ``<YYYYMMDD>/`` 那套批次归档。
SCORES_NAME = "scores.csv"
MANIFEST_NAME = "manifest.json"
DIAGNOSTICS_DIR = "diagnostics"

#: 逐起报谱表。``spectrum_<变量>.csv`` 是全体样本平均，只有这一张能给出
#: **每次起报各自**的谱，报告里按日期画/检验的谱曲线都取自它。
_SPECTRUM_BY_INIT_STEM = "spectrum_by_init"
SPECTRUM_BY_INIT_NAME = f"{_SPECTRUM_BY_INIT_STEM}.csv"

#: ``init_time`` 是 ISO 串（``2025-01-02T00:00:00.000000``），前 10 位即起报日。
_INIT_DATE_LEN = 10

#: 同一个产物目录在一个进程里可能被反复取数，长表又不小，所以缓存住。
#: 报告侧真正用到的长表列。完整长表 25 列、单份几 GB，``valid_time`` / ``storm`` /
#: ``protocol_id`` 这些报告一次也不查，``usecols`` 掉能省掉一截解析与内存。
#: ``region`` / ``group`` / ``status`` 是可选列，缺席时下游各自有兜底，所以先读表头。
_SCORE_COLUMNS = (
    "variable", "metric", "product_kind", "lead_h", "init_time", "value",
    "region", "group", "status", "unit",
)

_SCORES_CACHE: Dict[str, pd.DataFrame] = {}
_MANIFEST_CACHE: Dict[str, Dict[str, object]] = {}


class Archive(NamedTuple):
    """一个模型的评估产物。

    ``summary`` 与 ``meta`` 是为兼容渲染层保留的两个**派生视图**，新产物不落盘：

    ``summary``
        老 ``summary.csv`` 的形状——``init_date`` 一列，加每变量一列
        ``rmse_<v>_mean``，另带 ``variables`` / ``n_leads``。由 ``scores.csv``
        里 ``metric=rmse`` 的行按起报日、对全部时效平均现算。
    ``meta``
        老 ``batch_meta.json`` 的形状——``n_dates`` / ``n_ok`` / ``failures``。
        新产物没有「请求过的起报日数」这个概念，所以 ``n_dates`` 取**实际出了数的
        起报日数**，``n_ok`` 是其中全部行都 ``status == "success"`` 的那些。
    """

    name: str
    root: Path
    summary: pd.DataFrame
    meta: Dict[str, object]
    dates: List[str]  # scores.csv 里出现过的起报日（升序）


def _to_init_date(series: pd.Series) -> pd.Series:
    """``init_time`` / ``valid_time`` 的 ISO 串 → ``YYYYMMDD``。"""
    return series.astype(str).str.slice(0, _INIT_DATE_LEN).str.replace("-", "", regex=False)


def load_scores(root: Path) -> pd.DataFrame:
    """读 ``scores.csv`` 长表，附一列 ``init_date``。同一个目录只读一次。

    Raises:
        FileNotFoundError: 没有 ``scores.csv``——那不是一份评估产物目录。
        ValueError: 缺报告要用的列。
    """
    root = Path(root)
    key = str(root)
    cached = _SCORES_CACHE.get(key)
    if cached is not None:
        return cached
    path = root / SCORES_NAME
    if not path.is_file():
        raise FileNotFoundError(f"缺 {SCORES_NAME}（{path}）：这不是一份评估产物目录")
    # 先读表头：既拿来挑 ``usecols``，也用于缺列报错（不把整个文件读进来才发现）。
    header = pd.read_csv(path, nrows=0).columns
    missing = [
        column for column in ("variable", "metric", "product_kind", "lead_h", "init_time", "value")
        if column not in header
    ]
    if missing:
        raise ValueError(f"{path} 缺列 {missing}；现有列: {list(header)}")
    # low_memory=False：``unit`` / ``level`` / ``region`` 这些列是「字符串或空」，
    # 分块推断会在不同块里得出不同的 dtype（一串 DtypeWarning），读全再推断才一致。
    frame = pd.read_csv(
        path,
        usecols=[column for column in _SCORE_COLUMNS if column in header],
        low_memory=False,
    )
    frame["lead_h"] = pd.to_numeric(frame["lead_h"], errors="coerce")
    frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
    frame["init_date"] = _to_init_date(frame["init_time"])
    _SCORES_CACHE[key] = frame
    return frame


def load_manifest(root: Path) -> Dict[str, object]:
    """读 ``manifest.json``（运行记录），同一个目录只读一次。"""
    root = Path(root)
    key = str(root)
    cached = _MANIFEST_CACHE.get(key)
    if cached is not None:
        return cached
    path = root / MANIFEST_NAME
    if not path.is_file():
        raise FileNotFoundError(f"缺 {MANIFEST_NAME}（{path}）")
    with open(path, "r", encoding="utf-8") as handle:
        meta = json.load(handle)
    _MANIFEST_CACHE[key] = meta
    return meta


def _region_rows(scores: pd.DataFrame, region: Optional[str]) -> pd.DataFrame:
    """按纬度带取行：``region=None`` 只要全球行，给了带名只要**那一个带**的行。

    产物对同一 (起报, 时效, 指标) 会同时写全球行和 ``config_options.regions``
    里的区域行（``tropics`` 等），全球行的 ``region`` 是空串，读进来是 NaN。

    **两种口径不能混着平均**：全球平均和纬度带平均是两个量，把带行和全球行一起
    丢进 ``pivot_table`` 会让 ``aggfunc="mean"`` 悄悄把带间差异抹平，出来的数既不是
    全球的也不是任何一个带的，且没人看得出来。所以取数一律显式二选一。

    长表没有 ``region`` 列（这份产物不分区）时：要全球行就是全部行，
    要某个带就是空——**不静默退化成全球**。
    """
    if "region" not in scores.columns:
        return scores if region is None else scores.iloc[0:0]
    values = scores["region"]
    empty = values.isna() | (values.astype(str).str.strip() == "")
    if region is None:
        return scores[empty]
    return scores[~empty & (values.astype(str).str.strip() == str(region))]


def _global(scores: pd.DataFrame) -> pd.DataFrame:
    """只留全球口径的行（``region`` 为空）。见 :func:`_region_rows`。"""
    return _region_rows(scores, None)


#: ``manifest["regions"]`` 的条目形状：``"tropics[-20, 20]"``。
_REGION_ENTRY = re.compile(r"^(?P<name>.+?)\s*\[(?P<bounds>.+)\]$")


def region_names(root: Path) -> List[str]:
    """这份产物**真的落了数**的纬度带名（升序），取自 ``scores.csv`` 的 ``region`` 列。

    以数据为准而不是 manifest：manifest 里写着一堆 ``regions``、长表里却没有对应行
    （配置改了没重跑、或老产物）时，按数据来才不会端出一张空表。
    """
    scores = load_scores(root)
    if "region" not in scores.columns:
        return []
    values = scores["region"]
    filled = values[values.notna() & (values.astype(str).str.strip() != "")]
    return sorted({str(v).strip() for v in filled})


def region_labels(root: Path) -> Dict[str, str]:
    """``{带名: 边界串}``，边界取自 ``manifest["regions"]``（形如 ``tropics[-20, 20]``）。

    带名和边界都**照搬配置**——报告不翻译、也不写死任何带名。配置里叫
    ``mid_latitudes`` 就显示 ``mid_latitudes``；要换成「热带/中纬/高纬」那套切法，
    改配置的 ``options["regions"]`` 重跑即可，报告这边一个字都不用动。
    """
    labels: Dict[str, str] = {}
    for entry in load_manifest(root).get("regions") or []:
        match = _REGION_ENTRY.match(str(entry).strip())
        if match:
            labels[match.group("name").strip()] = match.group("bounds").strip()
    return labels


def region_display(name: str, labels: Mapping[str, str]) -> str:
    """带名的展示写法：``tropics [-20, 20]``；没有边界信息就只写带名。"""
    bounds = labels.get(name)
    return f"{name} [{bounds}]" if bounds else str(name)


def _deterministic(scores: pd.DataFrame) -> pd.DataFrame:
    """只留确定性指标的行。

    ``spread_error`` 展开出来的 ``rmse`` 与确定性 ``rmse`` **在长表里同名**，
    只能靠 ``product_kind``（``ensemble`` / ``deterministic``）区分。
    """
    if "product_kind" not in scores.columns:
        return scores
    return scores[scores["product_kind"].astype(str) == "deterministic"]


def metric_rows(
    root: Path,
    metric: str,
    *,
    product_kind: Optional[str] = None,
    region: Optional[str] = None,
) -> pd.DataFrame:
    """长表里某个指标的行；``product_kind`` 给了就再按它过滤。

    三个报告模块都从这里取数：确定性指标传 ``product_kind="deterministic"``，
    集合特有的（``crps`` / ``spread`` / ``spread_error_ratio``）传 ``"ensemble"``，
    ``activity`` 的四个展开量是 ``"specialized"``（不传就是不过滤）。

    ``region`` 省略即**全球口径**（只要 ``region`` 为空的行）；给了带名就只要那个带。
    两者互斥，没有「都要」的选项——见 :func:`_region_rows`。
    """
    rows = _region_rows(load_scores(root), region)
    rows = rows[rows["metric"] == str(metric)]
    if product_kind is not None and "product_kind" in rows.columns:
        rows = rows[rows["product_kind"].astype(str) == str(product_kind)]
    return rows


def by_date_frames(rows: pd.DataFrame, dates: Sequence[str], label: str = "") -> Dict[str, pd.DataFrame]:
    """长表行 → ``{起报日: lead × 变量 表}``。

    配对检验要**逐日起报**的样本（先按日平均再检验等于把样本量从 N 天压成 1 个点），
    所以这里保留逐日，不像 :func:`load_by_lead` 那样直接跨日平均。

    Raises:
        FileNotFoundError: 任何一个请求的起报日没有结果——按 SKILL.md 的硬性规则
            不静默跳过。
    """
    wanted = [str(date) for date in dates]
    stamp = rows["init_date"].astype(str)
    have = set(stamp)
    missing = [date for date in wanted if date not in have]
    if missing:
        raise FileNotFoundError(
            f"{label or '产物'}: {len(missing)} 个起报日没有结果（如 {missing[0]}）"
        )
    # 一次 groupby 切出所有日期。原先对每个日期都在整张表上做一次布尔掩码，
    # 348 天 × 数百万行 = 十几亿次比较，单个指标就要几十秒。
    subset = rows[stamp.isin(set(wanted))]
    grouped = {
        str(date): own
        for date, own in subset.groupby("init_date", sort=False)
    }
    frames: Dict[str, pd.DataFrame] = {}
    for date in wanted:
        # 顺序按调用方给的 ``dates``，不按 groupby 的原始顺序。
        frames[date] = grouped[date].pivot_table(
            index="lead_h", columns="variable", values="value", aggfunc="mean"
        ).sort_index()
    return frames


def _summary_from_scores(scores: pd.DataFrame) -> pd.DataFrame:
    """把长表透视成老 ``summary.csv`` 的形状。

    Raises:
        ValueError: 长表里没有 ``metric=rmse`` 的确定性行，分变量结果无从谈起。
    """
    rows = _deterministic(_global(scores))
    rows = rows[rows["metric"] == "rmse"]
    if rows.empty:
        raise ValueError("scores.csv 里没有确定性 rmse 的行，算不出分变量 RMSE")
    table = rows.pivot_table(index="init_date", columns="variable", values="value", aggfunc="mean")
    table = table.sort_index()
    table.columns = [f"rmse_{v}_mean" for v in table.columns]
    table = table.reset_index()
    # 老 summary 的 variables 列是「该模型算过的变量清单」，第 6.2 节拿它跟
    # --declared 对照。这里取长表里出现过的全部变量，不只 rmse 覆盖的那些。
    table["variables"] = ",".join(sorted({str(v) for v in scores["variable"].dropna().unique()}))
    table["n_leads"] = int(scores["lead_h"].dropna().nunique())
    return table


def _meta_from_scores(scores: pd.DataFrame) -> Dict[str, object]:
    """老 ``batch_meta.json`` 的等价物，喂给第 2 节的完整性检查。"""
    dates = sorted({str(d) for d in scores["init_date"].dropna().unique()})
    if "status" in scores.columns:
        ok = scores["status"].astype(str) == "success"
    else:
        ok = pd.Series(True, index=scores.index)
    failed = sorted({str(d) for d in scores.loc[~ok, "init_date"].dropna().unique()})
    return {
        "n_dates": len(dates),
        "n_ok": len(dates) - len(failed),
        "failures": failed,
        # 新产物没有「请求过的日期」这一说，留空让渲染层显式判空
        "dates": [],
    }


def load_archive(name: str, root: Path) -> Archive:
    """读一份评估产物目录。

    Raises:
        FileNotFoundError: 目录、``scores.csv`` 或 ``manifest.json`` 不存在。
        ValueError: ``scores.csv`` 没有数据行或缺列。
    """
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"产物目录不存在: {root}")
    scores = load_scores(root)
    if scores.empty:
        raise ValueError(f"{name}: {SCORES_NAME} 没有数据行")
    load_manifest(root)  # 缺 manifest 提前报，别拖到渲染层才炸
    dates = sorted({str(d) for d in scores["init_date"].dropna().unique()})
    return Archive(
        name=str(name), root=root, summary=_summary_from_scores(scores),
        meta=_meta_from_scores(scores), dates=dates,
    )


def common_dates(archives: Sequence[Archive]) -> List[str]:
    """所有模型**和它们的 summary.csv** 都有的日期（升序）。"""
    if not archives:
        raise ValueError("没有模型")
    shared: Optional[set] = None
    for archive in archives:
        own = set(archive.summary["init_date"])
        shared = own if shared is None else (shared & own)
    return sorted(shared or set())


def summary_variables(archive: Archive) -> List[str]:
    """``summary.csv`` 的 ``variables`` 列里声明过的变量（并集，字典序）。"""
    names: set = set()
    if "variables" in archive.summary.columns:
        for value in archive.summary["variables"].dropna():
            for item in str(value).split(","):
                item = item.strip()
                if item:
                    names.add(item)
    if not names:  # 没有 variables 列就退回从 rmse_<v>_mean 列名反推
        names = set(rmse_mean_columns(archive.summary))
    return sorted(names)


def rmse_mean_columns(summary: pd.DataFrame) -> List[str]:
    """``rmse_<v>_mean`` 列名里的变量（字典序）。"""
    found = []
    for column in summary.columns:
        match = re.fullmatch(r"rmse_(.+)_mean", str(column))
        if match:
            found.append(match.group(1))
    return sorted(found)


def _metric_by_lead(
    root: Path,
    metric: str,
    dates: Sequence[str],
    name: str = "",
    *,
    deterministic: bool = True,
    region: Optional[str] = None,
) -> pd.DataFrame:
    """长表里 ``metric`` 的行 → lead × 变量 表（跨起报日平均）。``region`` 给了就只算那个带。

    Raises:
        FileNotFoundError: 任何一个请求的起报日没有这个指标的结果——按 SKILL.md
            的硬性规则不静默跳过（跳过会让表和图的样本量对不上，且没人会发现）。
    """
    rows = metric_rows(
        root, metric, product_kind="deterministic" if deterministic else None, region=region,
    )
    frames = by_date_frames(rows, dates, f"{name or root}: {metric}")
    if not frames:
        raise ValueError(f"{name or root}: 没有可读的 {metric} 结果")
    stacked = pd.concat(frames.values(), axis=0)
    return stacked.groupby(level=0).mean().sort_index()


def probe_dates(root: Path, dates: Iterable[str], stem: str) -> List[str]:
    """这些起报日里，**没有** ``stem`` 结果的（升序）。用于第 2 节完整性检查。"""
    scores = _deterministic(_global(load_scores(root)))
    have = set(scores.loc[scores["metric"] == str(stem), "init_date"].astype(str))
    return sorted(str(date) for date in dates if str(date) not in have)


def load_by_lead(
    root: Path, stem: str, dates: Sequence[str], name: str = "", *, region: Optional[str] = None,
) -> pd.DataFrame:
    """``stem`` 指标按 lead 平均成一张 lead × 变量表；``region`` 给了就只算那个带。"""
    return _metric_by_lead(root, stem, dates, name, region=region)


def region_rmse_by_date(root: Path, region: str, dates: Sequence[str]) -> pd.DataFrame:
    """某个纬度带上 ``metric=rmse`` 的 **日期 × 变量** 表（对该带的全部 lead 平均）。

    全球口径的同名物是 :func:`_summary_from_scores` 里那张透视表；这里只是多一个
    ``region`` 过滤，口径与它完全一致（同样只取确定性 rmse、同样跨 lead 平均）。

    该带在这份产物里没有 rmse 行时返回**空表**而不是报错——分带块整体可以「本批未出」，
    但一台模型缺一个带不该把整份报告打掉，缺哪个带由调用方按空表判。
    """
    rows = metric_rows(root, "rmse", product_kind="deterministic", region=region)
    if rows.empty:
        return pd.DataFrame(index=[str(d) for d in dates], dtype=float)
    table = rows.pivot_table(index="init_date", columns="variable", values="value", aggfunc="mean")
    return table.reindex(index=[str(d) for d in dates])


def load_fa_ratio(root: Path, dates: Sequence[str], name: str = "") -> pd.DataFrame:
    """``activity_ratio`` → lead × 变量 的 FA 比值表。

    ``activity`` 指标在长表里展开成 ``activity_ratio`` / ``_bias`` /
    ``_forecast`` / ``_observation`` 四行，报告要的 ratio 只是其中一行。
    """
    return _metric_by_lead(root, "activity_ratio", dates, name, deterministic=False)


def spectrum_variables(root: Path) -> List[str]:
    """产物里出了纬向谱的变量（字典序）。

    以 ``diagnostics/spectrum_by_init.csv`` 的 ``variable`` 列为准——它有全部逐起报
    曲线；没有这张表就退回 ``diagnostics/spectrum_<变量>.csv`` 的文件名。
    """
    directory = Path(root) / DIAGNOSTICS_DIR
    by_init = directory / SPECTRUM_BY_INIT_NAME
    if by_init.is_file():
        raw = pd.read_csv(by_init, usecols=["variable"])
        return sorted({str(v) for v in raw["variable"].dropna().unique()})
    return sorted({
        path.stem[len("spectrum_"):]
        for path in directory.glob("spectrum_*.csv")
        if path.stem != _SPECTRUM_BY_INIT_STEM
    })


def load_spectrum(
    root: Path, dates: Sequence[str], variable: str, name: str = "",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """``diagnostics/spectrum_by_init.csv`` → ``(pred, obs)`` 两张 波数 × 起报日 表。

    这张表是**逐起报**的：每个 ``(变量, 起报时刻)`` 一条曲线，是该起报各时效曲线的
    按样本数加权平均——正是老归档 ``spectrum_<date>_<变量>.csv`` 的口径。

    Raises:
        FileNotFoundError: 产物没有 ``spectrum_by_init.csv``（配置里没挂 spectrum writer）。
        ValueError: 表里没有该变量，或请求的日期区间内没有它的谱。
    """
    path = Path(root) / DIAGNOSTICS_DIR / SPECTRUM_BY_INIT_NAME
    if not path.is_file():
        raise FileNotFoundError(
            f"{name or root}: 缺 {DIAGNOSTICS_DIR}/{SPECTRUM_BY_INIT_NAME}"
            f"（配置的 writers 里要有 spectrum 才会出这张表）"
        )
    raw = pd.read_csv(path)
    for column in ("variable", "init_time", "wavenumber", "pred", "obs"):
        if column not in raw.columns:
            raise ValueError(f"{path} 缺列 {column!r}；现有列: {list(raw.columns)}")
    raw = raw[raw["variable"].astype(str) == str(variable)]
    if raw.empty:
        available = sorted({str(v) for v in pd.read_csv(path, usecols=["variable"])["variable"].dropna()})
        raise ValueError(f"{path}: 没有变量 {variable} 的谱；现有变量: {available}")
    raw = raw.assign(init_date=_to_init_date(raw["init_time"]))
    raw = raw[raw["init_date"].isin({str(d) for d in dates})]
    if raw.empty:
        raise ValueError(f"{path}: 请求的起报日区间内没有 {variable} 的谱")
    pred = raw.pivot_table(
        index="wavenumber", columns="init_date", values="pred", aggfunc="mean"
    ).sort_index()
    obs = raw.pivot_table(
        index="wavenumber", columns="init_date", values="obs", aggfunc="mean"
    ).sort_index()
    return pred, obs


# ================================================================== 统计检验


def _wilcoxon_p(a: Sequence[float], b: Sequence[float]) -> float:
    """配对 Wilcoxon 符号秩检验的 p 值；样本不足或全部相等时返回 1.0。"""
    if _scipy_wilcoxon is None:
        raise ImportError(
            "做配对检验需要 scipy（scipy.stats.wilcoxon）；"
            "当前环境没有装，装上再跑，或不要调用报告生成"
        )
    left = np.asarray(a, dtype=float)
    right = np.asarray(b, dtype=float)
    mask = np.isfinite(left) & np.isfinite(right)
    left, right = left[mask], right[mask]
    if left.size < 2:
        return float("nan")
    if np.allclose(left, right, equal_nan=True):
        return 1.0
    try:
        return float(_scipy_wilcoxon(left, right).pvalue)
    except ValueError:
        return 1.0


def _holm(pvalues: Sequence[float]) -> List[float]:
    """Holm 逐步下降法校正；p 里带 nan 的原样返回 nan。"""
    index = [i for i, p in enumerate(pvalues) if math.isfinite(p)]
    ordered = sorted(index, key=lambda i: pvalues[i])
    adjusted = [float("nan")] * len(pvalues)
    running = 0.0
    total = len(ordered)
    for rank, i in enumerate(ordered):
        running = max(running, (total - rank) * pvalues[i])
        adjusted[i] = min(1.0, running)
    return adjusted


class PairTest(NamedTuple):
    """一对模型的检验结论。"""

    a: str
    b: str
    mean_a: float
    mean_b: float
    delta_pct: float  # (A − B) / B × 100
    p_raw: float
    p_holm: float
    significant: bool
    better: str  # 显著优者；不显著时是 "—"


def paired_tests(
    frame: pd.DataFrame,
    *,
    higher_is_better: bool = False,
) -> List[PairTest]:
    """``frame`` 是 日期 × 模型，两两配对做 Wilcoxon + Holm。

    Args:
        frame: index=配对样本（日期），columns=模型名。
        higher_is_better: ACC 这类"越大越好"的指标置 True。
    """
    names = [str(c) for c in frame.columns]
    comparisons = [(a, b) for i, a in enumerate(names) for b in names[i + 1:]]
    if not comparisons:
        return []

    raw, means = [], {}
    for name in names:
        values = pd.to_numeric(frame[name], errors="coerce")
        means[name] = float(values.mean()) if values.notna().any() else float("nan")
    for a, b in comparisons:
        raw.append(_wilcoxon_p(frame[a], frame[b]))
    holm = _holm(raw)

    tests = []
    for (a, b), p_raw, p_holm in zip(comparisons, raw, holm):
        significant = math.isfinite(p_holm) and p_holm < 0.05
        if significant:
            a_wins = means[a] > means[b] if higher_is_better else means[a] < means[b]
            better = a if a_wins else b
        else:
            better = "—"
        base = means[b]
        delta = BASELINE * (means[a] - means[b]) / base if base else float("nan")
        tests.append(PairTest(a, b, means[a], means[b], delta, p_raw, p_holm, significant, better))
    return tests


def variable_win_counts(
    by_variable: Mapping[str, pd.DataFrame],
    names: Sequence[str],
) -> pd.DataFrame:
    """逐变量配对检验 → 每个模型对的「A 显著更优 / B 显著更优 / 无差异」计数。

    Args:
        by_variable: ``{变量: 日期 × 模型 表}``（RMSE 这类越小越好）。
    """
    rows = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            wins_a = wins_b = ties = 0
            for frame in by_variable.values():
                if a not in frame.columns or b not in frame.columns:
                    continue
                p = _wilcoxon_p(frame[a], frame[b])
                if not (math.isfinite(p) and p < 0.05):
                    ties += 1
                elif float(frame[a].mean()) < float(frame[b].mean()):
                    wins_a += 1
                else:
                    wins_b += 1
            rows.append({
                "模型对": (a, b),
                "A 显著更优": wins_a,
                "B 显著更优": wins_b,
                "无显著差异": ties,
                "总变量数": wins_a + wins_b + ties,
            })
    return pd.DataFrame(rows)


# ================================================================== 派生指标


def _relative(table: pd.DataFrame) -> pd.DataFrame:
    """逐行（变量）做 100 × 值 / 几何均值，columns=模型、index=变量。"""
    geometric = np.exp(np.log(table.where(table > 0)).mean(axis=1))
    return BASELINE * table.div(geometric, axis=0)


def _composite(relative: pd.DataFrame) -> pd.Series:
    """对共享变量等权平均成每个模型一个数。"""
    return relative.mean(axis=0, skipna=True)


def _rank(series: pd.Series, *, higher_is_better: bool = False) -> pd.Series:
    """名次（1 最好）；nan 排最后。"""
    ordered = series.astype(float)
    return ordered.rank(ascending=not higher_is_better, method="min")


# ================================================================== 汇总分析


class Bundle(NamedTuple):
    """算完的一切。渲染只读它，不再碰文件。"""

    names: List[str]
    archives: Dict[str, Archive]
    dates: List[str]  # 共同日期（配对检验与 §2「有效共同日期」那一行用）
    #: 模型 -> 该模型**本次统计实际用的**起报日。默认等于 ``dates``（共同日期）；
    #: ``own_dates=True`` 时是各模型自己的全部起报日，长度可以互不相同。
    #: 凡是要给**单个模型**算平均的地方都读这里，别读 ``dates``。
    dates_by_model: Dict[str, List[str]]
    common_variables: List[str]
    acc_variable: str
    fa_variables: List[str]
    spectrum_variable: str
    spectrum_variables: List[str]
    rmse_by_date: Dict[str, pd.DataFrame]  # 模型 -> 日期 × 变量
    acc_by_lead: Dict[str, pd.DataFrame]
    fa_ratio_by_lead: Dict[str, pd.DataFrame]
    rmse_by_lead: Dict[str, pd.DataFrame]
    spectrum: Dict[str, pd.DataFrame]  # 模型 -> 波数 × {pred, obs}
    relative: pd.DataFrame  # 变量 × 模型
    per_date_relative: pd.DataFrame  # 日期 × 模型（综合相对 RMSE）
    relative_composite: pd.Series
    acc_mean: pd.Series
    fa_bias: pd.Series
    spectrum_rms: pd.Series
    rank_score: pd.Series
    per_date_acc: pd.DataFrame
    per_variable_rmse: Dict[str, pd.DataFrame]
    overall_tests: List[PairTest]
    acc_tests: List[PairTest]
    win_counts: pd.DataFrame
    # --- 球谐带功率（确定性单成员场的球谐分解；helper 与 collection 报告同源）---
    sph_available: bool
    sph_rms: pd.Series  # 模型 -> 全频带/变量/lead/日期总 log-RMS×100
    sph_bias: pd.Series  # 模型 -> mean|pred/obs-1|×100
    sph_band_logrms: pd.DataFrame  # 频带 × 模型
    sph_band_ratio: pd.DataFrame  # 频带 × 模型（均值比）
    sph_variable: pd.DataFrame  # 变量 × 模型
    sph_tests: Dict[str, List[PairTest]]  # 频带 -> 配对检验列表
    sph_coverage: Dict[str, Dict[str, Tuple[int, int]]]  # 模型 -> 变量 -> (用, 请求)
    sph_note: str  # 不可得时的原因，渲染进「本批未出」那句
    anomaly_variable: str
    anomaly_model: str
    anomaly_ratio: float
    lead_relative: pd.DataFrame  # lead × 模型（逐 lead 综合相对 RMSE）
    # --- 分纬度带（可选块「附 L」）---
    region_names: List[str]  # 各模型都有的带名（升序），空 = 本批出不了这一块
    region_labels: Dict[str, str]  # 带名 -> 边界串（来自 manifest）
    region_relative_composite: pd.DataFrame  # 带名 × 模型（带内综合相对 RMSE）
    region_lead_relative: Dict[str, pd.DataFrame]  # 带名 -> lead × 模型
    region_rmse_by_lead: Dict[str, Dict[str, pd.DataFrame]]  # 带名 -> 模型 -> lead × 变量
    region_note: str  # 出不了这一块时的原因，渲染进「本批未出」那句


def _lead_relative(rmse_by_lead: Mapping[str, pd.DataFrame], names: Sequence[str],
                   variables: Sequence[str]) -> pd.DataFrame:
    """逐 lead 的综合相对 RMSE（lead × 模型）。"""
    columns = {}
    for name in names:
        frame = rmse_by_lead[name].reindex(columns=variables)
        columns[name] = frame
    stacked = pd.concat(columns, axis=1, keys=names)  # 列 = (模型, 变量)
    per_variable = {v: stacked.xs(v, level=1, axis=1) for v in variables}
    relative = [_relative(per_variable[v]) for v in variables]
    return sum(relative) / len(relative)


def analyze(
    archives: Sequence[Archive],
    *,
    declared: Optional[Mapping[str, Sequence[str]]] = None,
    own_dates: bool = False,
) -> Bundle:
    """把 N 份产物算成一份 :class:`Bundle`。

    Args:
        declared: 调用方声明的共同变量，``None`` 表示由产物自己定。
        own_dates: ``False``（默认）时每个模型的每一项统计都只在**共同日期**上算
            ——各模型覆盖的时段不同时，比的是同一批日子。
            ``True`` 时每个模型的统计与曲线用**它自己的全部起报日**（各自的平均值、
            各自的天数），只有**配对检验**仍退回共同日期：日期不对齐的样本做不了
            Wilcoxon。这条路径用于各模型覆盖时段差很多、又要直接对比的情形——
            此时基线（跨模型的几何均值）是拿不同时段的平均在比，**不公平但看得见**，
            报告里必须点名（见 :mod:`ens_report` 的 §1/§2/§7）。

    Raises:
        ValueError: 共同日期太少（< 2）或没有共享的 RMSE 变量。
        FileNotFoundError: 用到的日期上缺逐日文件——不静默跳过。
    """
    if len(archives) < 2:
        raise ValueError("多模型报告至少要两个模型")
    names = [archive.name for archive in archives]
    by_name = {archive.name: archive for archive in archives}
    dates = common_dates(archives)
    if len(dates) < 2:
        raise ValueError(f"共同日期只有 {len(dates)} 天，做不了配对检验: {dates}")
    # 每个模型「本次统计实际用哪几天」。own_dates=False 时所有模型都是共同日期，
    # 逐字保持既有行为；True 时各用各的，长度可以不一样。
    dates_by_model = {
        archive.name: (list(archive.dates) if own_dates else dates) for archive in archives
    }
    if own_dates:
        # 配对检验要有足够的共同日期才有意义；共同日期本身在 :func:`common_dates`
        # 那里已经保证 ≥2，这里只负责让「谁的天数太少」这件事在报告里说得出来。
        empty = [name for name, own in dates_by_model.items() if not own]
        if empty:
            raise ValueError(f"这些模型自己的起报日为空，算不出任何统计: {empty}")

    # --- 变量集：所有模型 summary 里都有、且真的算出了数的 RMSE 变量 ---
    shared = set(rmse_mean_columns(archives[0].summary))
    for archive in archives[1:]:
        shared &= set(rmse_mean_columns(archive.summary))
    variables = sorted(
        v for v in shared
        if all(archive.summary[f"rmse_{v}_mean"].notna().any() for archive in archives)
    )
    if not variables:
        raise ValueError("这些模型没有共享的 rmse_<变量>_mean 列，算不出分变量结果")

    rmse_by_date: Dict[str, pd.DataFrame] = {}
    for archive in archives:
        indexed = archive.summary.set_index("init_date")
        columns = {v: pd.to_numeric(indexed[f"rmse_{v}_mean"], errors="coerce") for v in variables}
        rmse_by_date[archive.name] = pd.DataFrame(columns).reindex(
            index=dates_by_model[archive.name]
        )

    # --- 逐日文件（缺就报错，不静默跳过）---
    # acc_<date>_det.csv 里存的是相关系数本身（-1..1），报告统一按百分数呈现，
    # 所以在入口乘 100，下游（表、图、检验）拿到的都是 %。
    acc_by_lead = {
        a.name: load_by_lead(a.root, "acc", dates_by_model[a.name], a.name) * BASELINE
        for a in archives
    }
    fa_ratio_by_lead = {
        a.name: load_fa_ratio(a.root, dates_by_model[a.name], a.name) for a in archives
    }
    rmse_by_lead = {
        a.name: load_by_lead(a.root, "rmse", dates_by_model[a.name], a.name) for a in archives
    }

    acc_columns = set(acc_by_lead[names[0]].columns)
    for name in names[1:]:
        acc_columns &= set(acc_by_lead[name].columns)
    acc_variable = _pick_variable(acc_columns, variables)

    fa_columns = set(fa_ratio_by_lead[names[0]].columns)
    for name in names[1:]:
        fa_columns &= set(fa_ratio_by_lead[name].columns)
    fa_variables = sorted(fa_columns)
    if not fa_variables:
        raise ValueError("这些模型没有共享的 <变量>_ratio 列，算不出 FA 偏差")

    spectrum_vars = None
    for archive in archives:
        own = set(spectrum_variables(archive.root))
        spectrum_vars = own if spectrum_vars is None else (spectrum_vars & own)
    spectrum_vars = sorted(spectrum_vars or set())
    if not spectrum_vars:
        raise ValueError(
            "这些模型的产物里都没有纬向谱（diagnostics/"
            f"{SPECTRUM_BY_INIT_NAME}），算不出频谱指标"
        )
    spectrum_variable = _pick_variable(set(spectrum_vars), variables)

    spectrum: Dict[str, pd.DataFrame] = {}
    spectrum_rms_values: Dict[str, float] = {}
    for archive in archives:
        pred, obs = load_spectrum(
            archive.root, dates_by_model[archive.name], spectrum_variable, archive.name
        )
        spectrum[archive.name] = pd.DataFrame({"pred": pred.mean(axis=1), "obs": obs.mean(axis=1)})
        per_date = []
        for date in pred.columns:
            p = pred[date].where(lambda s: s > 0)
            o = obs[date].where(lambda s: s > 0)
            ratio = (p / o).dropna()
            ratio = ratio[ratio.index > 0]
            if not ratio.empty:
                per_date.append(float(np.sqrt(np.mean(np.log10(ratio.values) ** 2))))
        spectrum_rms_values[archive.name] = BASELINE * float(np.mean(per_date)) if per_date else float("nan")

    # --- 派生指标 ---
    # 跨模型的**几何均值基线**（``_relative`` 的分母）逐行算，所以每一行（变量／日期）
    # 必须所有模型都在场，否则这一行的基线是拿残缺的模型集算的。own_dates=True 时
    # 各模型天数不同，**只有配对检验那些帧**需要按下面的 ``_paired`` 裁齐，
    # 按变量取平均的 ``mean_rmse`` 不受影响：每个模型各自的均值都在。
    mean_rmse = pd.DataFrame({name: rmse_by_date[name].mean(axis=0) for name in names}).reindex(variables)
    relative = _relative(mean_rmse)
    relative_composite = _composite(relative)

    #: 逐日配对帧统一走这个：裁到共同日期；``own_dates`` 时还要把任何模型缺数的
    #: 那一天整行丢掉——日期不齐会让「谁跟谁配对」这件事说不清。默认路径
    #: **不丢**，逐字保持既有行为。
    def _pair(frame: pd.DataFrame) -> pd.DataFrame:
        frame = frame.reindex(index=[str(d) for d in dates])
        return frame.dropna(how="any") if own_dates else frame

    per_variable_relative = {}
    for variable in variables:
        columns = {name: rmse_by_date[name][variable] for name in names}
        per_variable_relative[variable] = _relative(_pair(pd.DataFrame(columns))).reindex(index=dates)
    per_date_relative = sum(per_variable_relative.values()) / len(variables)
    # ``per_variable_rmse`` 进 Bundle，§4.1 拿它的**均值**填表——own_dates 时那就是
    # 各模型自己的全部日期；「显著优胜」那一列另由共同日期上的检验给，
    # 两者口径不同，报告里写明。
    per_variable_rmse = {
        variable: pd.DataFrame({name: rmse_by_date[name][variable] for name in names})
        for variable in variables
    }

    # 逐日 ACC 直接读逐日文件，而不是从"先按 lead 平均、再按日期平均"的表里取——
    # 两者数值上等价，但配对检验要的是逐日样本，不能只有均值。
    per_date_acc = pd.DataFrame({
        name: _acc_by_date(by_name[name].root, dates_by_model[name], acc_variable, name)
        for name in names
    })
    acc_mean = per_date_acc.mean(axis=0)

    deviation = pd.DataFrame({
        name: (fa_ratio_by_lead[name][fa_variables] - 1.0).abs().mean(axis=1) for name in names
    })
    fa_bias = BASELINE * deviation.mean(axis=0)
    spectrum_rms = pd.Series(spectrum_rms_values)

    rank_score = (
        _rank(relative_composite)
        + _rank(acc_mean, higher_is_better=True)
        + _rank(fa_bias)
        + _rank(spectrum_rms)
    ) / 4.0

    # 检验一律在共同日期上做（见上方 ``_pair``）：§6 表里印的均值因此是**共同日期**
    # 上的均值，与 §4 各模型全量日期的均值**不是同一个数**，报告里两处都写明。
    overall_tests = paired_tests(per_date_relative)
    acc_tests = paired_tests(_pair(per_date_acc), higher_is_better=True)
    win_counts = variable_win_counts(
        {v: _pair(frame) for v, frame in per_variable_rmse.items()}, names
    )

    anomaly_variable, anomaly_model, anomaly_ratio = _detect_anomaly(mean_rmse, names)
    lead_relative = _lead_relative(rmse_by_lead, names, variables)

    # --- 分纬度带（可选块「附 L」）：各模型都有的带才出，一个都没有就整块「本批未出」---
    shared_regions: Optional[set] = None
    for archive in archives:
        own = set(region_names(archive.root))
        shared_regions = own if shared_regions is None else (shared_regions & own)
    region_list = sorted(shared_regions or set())
    region_label_map = region_labels(archives[0].root)
    region_rmse_by_lead: Dict[str, Dict[str, pd.DataFrame]] = {}
    region_lead_relative: Dict[str, pd.DataFrame] = {}
    per_region_composite: Dict[str, pd.Series] = {}
    region_note = ""
    for region in region_list:
        own_models = {
            archive.name: region_rmse_by_date(
                archive.root, region, dates_by_model[archive.name]
            )
            .reindex(columns=variables).mean(axis=0)
            for archive in archives
        }
        # 带内**自己的**几何均值基线：跨带比绝对值没有意义（热带与极区差一个量级），
        # 除成本带基线之后，「哪个模型在这个带上更稳」才是可比的。
        per_region_composite[region] = _composite(_relative(pd.DataFrame(own_models).reindex(index=variables)))
        region_rmse_by_lead[region] = {
            archive.name: _metric_by_lead(
                archive.root, "rmse", dates_by_model[archive.name], archive.name, region=region
            )
            for archive in archives
        }
        region_lead_relative[region] = _lead_relative(region_rmse_by_lead[region], names, variables)

    if not region_list:
        region_note = (
            "这批产物的 scores.csv 里没有共同的 region 行——"
            "评测配置的 config_options.regions 为空，或者各模型的分带对不上。"
        )
    region_relative_composite = pd.DataFrame(per_region_composite).T

    # --- 球谐带功率（§4.5）：与纬向谱同一组变量 ---
    # wave_report 在模块顶层 import det_report，这里**必须在函数内**延迟导入，
    # 否则形成循环导入，``import ens_report`` 会直接失败。
    from reporting.reports.wave_report import (
        SPHERICAL_BANDS,
        _band_label,
        _band_tag,
        _bias_pct,
        _mean_ratio,
        _pooled_logrms,
        band_daily_logrms,
        band_dates,
        load_band_power,
        log_ratio_arrays,
        probe_bands,
    )

    band_names = [_band_label(lo, hi) for lo, hi in SPHERICAL_BANDS]
    sph_variables = list(spectrum_vars)
    probes = {
        archive.name: probe_bands(archive.root, dates_by_model[archive.name], sph_variables)
        for archive in archives
    }
    sph_available = bool(sph_variables) and all(probes.values())
    sph_tests: Dict[str, List[PairTest]] = {}
    sph_rms_values: Dict[str, float] = {}
    sph_bias_values: Dict[str, float] = {}
    sph_band_logrms = pd.DataFrame(index=band_names, columns=names, dtype=float)
    sph_band_ratio = pd.DataFrame(index=band_names, columns=names, dtype=float)
    sph_variable = pd.DataFrame(index=sph_variables, columns=names, dtype=float)
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
        # ``{模型: {频带: 值}}`` 直接转 DataFrame 就是「频带 × 模型」，
        # **不要 .T**：那会把 index 变成模型名，reindex(index=频带) 之后全成 NaN。
        sph_band_logrms = pd.DataFrame(logrms_rows).reindex(
            index=band_names, columns=names
        )
        sph_band_ratio = pd.DataFrame(ratio_rows).reindex(
            index=band_names, columns=names
        )
        sph_variable = pd.DataFrame(variable_rows).reindex(
            index=sph_variables, columns=names
        )
        sph_note = ""
    else:
        missing = [name for name, ok in probes.items() if not ok]
        sph_note = (
            "这些模型/变量的产物里没有可用的球谐带功率行"
            f"（``metric=spherical_band_power_*``、``group`` 为频带）：{'、'.join(missing)}。"
            "旧档案的 scores.csv 只有 23 列、没有 ``group`` 列，也会落到这里。"
        )
    sph_rms = pd.Series({name: sph_rms_values.get(name, float("nan")) for name in names})
    sph_bias = pd.Series({name: sph_bias_values.get(name, float("nan")) for name in names})

    return Bundle(
        names=names, archives=by_name, dates=dates, dates_by_model=dates_by_model,
        common_variables=variables,
        acc_variable=acc_variable, fa_variables=fa_variables,
        spectrum_variable=spectrum_variable, spectrum_variables=spectrum_vars,
        rmse_by_date=rmse_by_date, acc_by_lead=acc_by_lead, fa_ratio_by_lead=fa_ratio_by_lead,
        rmse_by_lead=rmse_by_lead, spectrum=spectrum,
        relative=relative, per_date_relative=per_date_relative,
        relative_composite=relative_composite, acc_mean=acc_mean, fa_bias=fa_bias,
        spectrum_rms=spectrum_rms, rank_score=rank_score, per_date_acc=per_date_acc,
        per_variable_rmse=per_variable_rmse,
        overall_tests=overall_tests, acc_tests=acc_tests, win_counts=win_counts,
        sph_available=sph_available, sph_rms=sph_rms, sph_bias=sph_bias,
        sph_band_logrms=sph_band_logrms, sph_band_ratio=sph_band_ratio,
        sph_variable=sph_variable, sph_tests=sph_tests, sph_coverage=sph_coverage,
        sph_note=sph_note,
        anomaly_variable=anomaly_variable, anomaly_model=anomaly_model,
        anomaly_ratio=anomaly_ratio, lead_relative=lead_relative,
        region_names=region_list, region_labels=region_label_map,
        region_relative_composite=region_relative_composite,
        region_lead_relative=region_lead_relative,
        region_rmse_by_lead=region_rmse_by_lead, region_note=region_note,
    )


def _acc_by_date(root: Path, dates: Sequence[str], variable: str, name: str) -> pd.Series:
    """逐起报日的 ACC 变量在全部 lead 上的平均（百分数），index=起报日。"""
    acc = metric_rows(root, "acc", product_kind="deterministic")
    rows = acc[acc["variable"].astype(str) == str(variable)]
    if rows.empty:
        available = sorted({str(v) for v in acc["variable"].dropna()})
        raise ValueError(f"{name}: 产物里没有变量 {variable} 的 acc 结果；现有变量: {available}")
    per_date = rows.groupby(rows["init_date"].astype(str))["value"].mean()
    wanted = [str(date) for date in dates]
    missing = [date for date in wanted if date not in per_date.index]
    if missing:
        raise FileNotFoundError(
            f"{name}: {len(missing)} 个起报日没有 {variable} 的 acc 结果（如 {missing[0]}）"
        )
    return BASELINE * per_date.reindex(wanted)


def _pick_variable(candidates: Iterable[str], fallback: Sequence[str]) -> str:
    """优先 z500，其次候选里字典序最小的，最后退回 fallback 的第一个。"""
    options = sorted(str(c) for c in candidates)
    if PREFERRED_VARIABLE in options:
        return PREFERRED_VARIABLE
    if options:
        return options[0]
    return str(fallback[0])


def _detect_anomaly(mean_rmse: pd.DataFrame, names: Sequence[str]) -> Tuple[str, str, float]:
    """挑「跨模型量级最离谱」的变量：max / median 比值最大的那个。"""
    ratios = {}
    for variable in mean_rmse.index:
        row = mean_rmse.loc[variable].dropna()
        median = float(row.median())
        ratios[variable] = float(row.max()) / median if median > 0 else float("nan")
    finite = {v: r for v, r in ratios.items() if math.isfinite(r)}
    if not finite:
        variable = str(mean_rmse.index[0])
    else:
        variable = max(finite, key=lambda v: finite[v])
    row = mean_rmse.loc[variable].dropna()
    model = str(row.idxmax())
    return variable, model, float(row.max()) / float(row.median()) if float(row.median()) > 0 else float("nan")


# ==================================================================== 渲染


def _signed(value, digits: int = 2) -> str:
    number = finite_or_none(value)
    return "—" if number is None else f"{number:+.{digits}f}"


# ------------------------------------------------------------------ 各章节


def _section_samples(bundle: Bundle, declared: Optional[Mapping[str, Sequence[str]]]) -> List[str]:
    """§2：样本口径、完整性、变量覆盖合并成一张表 + 两句话。"""
    lines = ["# 2. 样本口径与数据质量", ""]
    rows = []
    counts: Dict[str, int] = {}
    for name in bundle.names:
        archive = bundle.archives[name]
        meta = archive.meta
        own = sorted(set(archive.summary["init_date"]))
        common = set(own) & set(bundle.dates)
        used = bundle.dates_by_model.get(name, bundle.dates)
        # scores.csv 里出现过、却没有确定性 rmse 结果的起报日。正常产物是空的；
        # 非空说明这次只算了一部分指标，那些日期也就进不了 summary 和共同日期。
        missing = probe_dates(archive.root, archive.dates, "rmse")
        nan_variables = [
            v for v in rmse_mean_columns(archive.summary)
            if archive.summary[f"rmse_{v}_mean"].isna().all()
        ]
        n_dates = meta.get("n_dates")
        n_ok = meta.get("n_ok")
        problems = []
        if missing:
            problems.append(f"缺 RMSE {len(missing)} 天")
        if nan_variables:
            problems.append("NaN：" + "、".join(nan_variables[:4]))
        actual = summary_variables(archive)
        counts[name] = len(actual)
        rows.append([
            name,
            date_span(own),
            str(len(own)),
            str(len(used)),
            str(len(common)),
            f"{n_ok}/{n_dates}" if n_ok is not None and n_dates is not None else "—",
            "；".join(problems) if problems else "无",
            str(len(actual)),
        ])
    lines += table_lines(
        ["**模型**", "**日期范围**", "**起报日数**", "**本次使用**", "**共同日期**",
         "**完整性 n_ok/n_dates**", "**缺失**", "**变量数**"],
        rows,
    )
    lines.append("")

    failures = {name: bundle.archives[name].meta.get("failures") or [] for name in bundle.names}
    broken = {name: items for name, items in failures.items() if items}
    if broken:
        detail = "；".join(f"{name} 有 {len(items)} 个起报点失败" for name, items in broken.items())
    else:
        detail = "所有模型的产物里每一行 status 都是 success，n_ok = n_dates，没有失败起报点"
    lines.append(
        f"{detail}。正文与附录统一使用 {len(bundle.dates)} 个共同日期"
        f"（{date_span(bundle.dates)}），非共同日期不参与任何统计与曲线，"
        f"因此各表的样本量一致、可直接横向比较。"
    )
    lines.append("")

    if len(set(counts.values())) == 1:
        cover = (
            f"各模型的产物都覆盖 {list(counts.values())[0]} 个变量，变量覆盖一致；"
            f"本报告只对 {len(bundle.common_variables)} 个共享 RMSE 变量做横向比较。"
        )
    else:
        cover = (
            "变量覆盖不一致（"
            + "、".join(f"{name} {n} 个" for name, n in sorted(counts.items()))
            + f"），本报告只对 {len(bundle.common_variables)} 个共享变量比较，"
            "缺变量模型的结论不完整。"
        )
    declared = declared or {}
    unknown = [name for name in bundle.names if name not in declared]
    if not declared:
        cover += "本次未提供声明变量清单（`--declared`），变量覆盖只看产物实际算出的变量。"
    else:
        gaps = []
        for name in bundle.names:
            if name not in declared:
                continue
            claimed = {str(v).strip() for v in declared[name]}
            gap = sorted(claimed - set(summary_variables(bundle.archives[name])))
            if gap:
                gaps.append(f"{name} 缺 {'、'.join(gap)}")
        if unknown:
            cover += f"{'、'.join(unknown)} 未提供声明变量清单。"
        if gaps:
            cover += "声明变量未全部落盘：" + "；".join(gaps) + "。"
        elif not unknown:
            cover += "各模型的声明变量清单与产物实际变量一致。"
    lines.append(cover)
    lines.append("")
    return lines


def _section_metrics(bundle: Bundle) -> List[str]:
    """§3：指标表 + 口径说明（静态文本，只随球谐可得性改方向列）。"""
    pending = "" if bundle.sph_available else "；本批未出"
    rows = [
        ["RMSE", "预报与观测的均方根误差（逐变量、逐 lead）", "越低越好"],
        ["综合相对RMSE",
         "100 × 变量RMSE / 各模型该变量几何均值，再对共享变量等权平均（100 = 几何均值基线）",
         "越低越好"],
        ["ACC (%)", f"距平相关系数 ×100；本批使用 {bundle.acc_variable} 的 centered=False 口径",
         "越高越好"],
        ["FA ratio 偏差 (pp)",
         "100 × mean|pred/obs − 1|；全程 lead、不分时段（FA 指 Forecast Activity）", "越低越好"],
        ["频谱对数 RMS×100",
         "100 × RMS_k[log10(pred_k/obs_k)]，纬向 FFT 谱（仅主谱变量）", "越低越好"],
        [f"球谐谱 RMS×100",
         "100 × RMS[ln(pred/obs)]，对球谐总阶数频带池化（自然对数）", "越低越好" + pending],
        ["球谐比偏差 (%)", "100 × mean|pred/obs − 1|，跨频带、变量与 lead", "越低越好" + pending],
        ["平均排名分", "综合相对RMSE、ACC、FA偏差、频谱对数RMS 四项名次的平均", "越低越好"],
    ]
    lines = ["# 3. 评估指标与解释", "", "**指标口径**", ""]
    lines += table_lines(["**指标**", "**定义/口径**", "**方向**"], rows)
    lines += [
        "",
        f"报告类型为确定性单成员预报（{len(bundle.names)} 个模型）。"
        "RMSE、ACC、FA 与谱指标全部是**单成员场**口径，不是集合平均，也不含 CRPS/离散度。",
        "",
        "纬向谱的「频谱对数 RMS×100」取 **log10**；球谐带的「球谐谱 RMS×100」取 "
        "**自然对数 ln**，两者数值相差 2.3026 倍，**不是同一口径，不能跨表比较**。",
        "",
        f"逐日指标覆盖 {lead_span(bundle.rmse_by_lead[bundle.names[0]].index)[2]} 个 lead"
        f"（{lead_span(bundle.rmse_by_lead[bundle.names[0]].index)[0]}–"
        f"{lead_span(bundle.rmse_by_lead[bundle.names[0]].index)[1]}h）；"
        f"FA 只做全程统计、不分时效段，以免与降水二分类的 False Alarm 混淆。",
        "",
        f"显著性一律用共同日期（{len(bundle.dates)} 天）上的逐日配对 Wilcoxon 符号秩检验，"
        "多重比较用 Holm 逐步校正，Holm p < 0.05 才记「显著」。",
        "",
    ]
    return lines


def _section_overall(bundle: Bundle, figures: Mapping[str, str]) -> List[str]:
    lines = ["# 4. 总体评估结果", "", f"**{cn_number(len(bundle.names))}模型总体指标**", ""]
    order = bundle.relative_composite.sort_values().index.tolist()
    rank = _rank(bundle.relative_composite)
    sph_columns = bundle.sph_available
    rows = []
    for name in order:
        rows.append([
            f"{name}（第{int(rank[name])}）",
            fmt_number(bundle.relative_composite[name]),
            fmt_number(bundle.acc_mean[name]),
            fmt_number(bundle.fa_bias[name], 4),
            fmt_number(bundle.spectrum_rms[name], 2),
            fmt_number(bundle.sph_rms[name], 2) if sph_columns else "—",
            fmt_number(bundle.sph_bias[name], 2) if sph_columns else "—",
            fmt_number(bundle.rank_score[name], 1),
        ])
    lines += table_lines(
        [f"**模型（相对排序）**", "**综合相对RMSE**", f"**ACC（{bundle.acc_variable}）**",
         "**FA偏差(全程)**", "**频谱对数RMS×100**", "**球谐谱RMS×100**",
         "**球谐比偏差(%)**", "**平均排名分**"],
        rows,
    )
    lines.append("")

    items = [
        f"综合相对 RMSE 最低 {order[0]}（{fmt_number(bundle.relative_composite[order[0]])}），"
        f"最高 {order[-1]}（{fmt_number(bundle.relative_composite[order[-1]])}）；"
        f"100 为{cn_number(len(bundle.names))}模型几何均值基线。",
        f"ACC 最高 {bundle.acc_mean.idxmax()}（{fmt_number(bundle.acc_mean.max())}%），"
        f"FA 偏差最小 {bundle.fa_bias.idxmin()}（{fmt_number(bundle.fa_bias.min(), 4)} pp），"
        f"频谱对数 RMS×100 最小 {bundle.spectrum_rms.idxmin()}"
        f"（{fmt_number(bundle.spectrum_rms.min(), 2)}）。",
        "平均排名分：" + "、".join(
            f"{name} {fmt_number(bundle.rank_score[name], 1)}"
            for name in bundle.rank_score.sort_values().index
        ) + "。",
    ]
    if sph_columns:
        items.append(
            f"球谐谱 RMS×100 最低 {bundle.sph_rms.idxmin()}"
            f"（{fmt_number(bundle.sph_rms.min(), 2)}），"
            f"球谐比偏差最小 {bundle.sph_bias.idxmin()}"
            f"（{fmt_number(bundle.sph_bias.min(), 2)}%）。"
        )
    lines += bullet_lines(items)
    lines += [
        "",
        f"![overall_heatmap]({figures['overall_heatmap']})",
        "",
        f"图 1：{cn_number(len(bundle.names))}模型总体指标热力图。"
        "列为指标（ACC 越高越好，其余越低越好），行按综合相对 RMSE 从好到差排列；"
        "格内数字为原始值。颜色是该列在本批模型内的归一化相对表现（1 = 最优，0 = 最差），"
        "**各列量纲不同，颜色只在同一列内可比**，不能跨列比较。",
        "",
    ]
    return lines


def _section_rmse(bundle: Bundle, figures: Mapping[str, str]) -> List[str]:
    lines = ["## 4.1 RMSE", "", "**分变量RMSE（共同日期平均）**", ""]
    rows = []
    for variable in bundle.common_variables:
        values = pd.Series({name: bundle.rmse_by_date[name][variable].mean() for name in bundle.names})
        rows.append([variable] + [fmt_number(values[name], 4) for name in bundle.names]
                    + [_best_cell(values, 4)])
    lines += table_lines(["**变量**"] + list(bundle.names) + ["**Best**"], rows)
    lines += [
        "",
        f"![rmse_vs_lead]({figures['rmse_plot']})",
        "",
        f"图 2：{len(bundle.common_variables)}个变量 RMSE 随预报时效的变化"
        f"（{date_span(bundle.dates)} 共同日期平均）。纵轴按各变量数据范围缩放，"
        "便于分辨模型间差异；绝对数值见上表。",
        "",
    ]

    items = []
    for variable in bundle.common_variables:
        values = pd.Series({name: bundle.rmse_by_date[name][variable].mean() for name in bundle.names})
        finite = values.dropna()
        if finite.empty:
            continue
        low, high = float(finite.min()), float(finite.max())
        items.append(
            f"{variable}：最低 {finite.idxmin()}（{fmt_number(low, 4)}），"
            f"最高 {finite.idxmax()}（{fmt_number(high, 4)}），极差 {fmt_number(high - low, 4)}。"
        )
    lines += bullet_lines(items)
    return lines


def _section_acc(bundle: Bundle, figures: Mapping[str, str]) -> List[str]:
    lines = ["## 4.2 ACC", "", "**分模型平均ACC（%）**", ""]
    order = bundle.acc_mean.sort_values(ascending=False)
    rows = [[name, fmt_number(bundle.acc_mean[name]), str(int(i + 1))]
            for i, name in enumerate(order.index)]
    lines += table_lines(["**模型**", f"**ACC（{bundle.acc_variable}，%）**", "**排名**"], rows)
    lines += [
        "",
        f"![acc_vs_lead]({figures['acc_plot']})",
        "",
        f"图 3：{bundle.acc_variable} ACC 随预报时效的变化（{date_span(bundle.dates)} 共同日期平均）。"
        "纵轴按数据范围缩放，越高越好。",
        "",
    ]

    spread = float(order.iloc[0] - order.iloc[-1])
    items = [
        f"ACC 最高 {order.index[0]}（{fmt_number(order.iloc[0])}%），"
        f"最低 {order.index[-1]}（{fmt_number(order.iloc[-1])}%），极差 {fmt_number(spread)} pp。"
    ]
    frame = bundle.acc_by_lead[bundle.names[0]]
    if bundle.acc_variable in frame.columns and not frame.empty:
        first_lead = float(frame[bundle.acc_variable].iloc[0])
        if first_lead > SATURATED_ACC:
            items.append(
                f"首时效 {bundle.acc_variable} ACC 已到 {fmt_number(first_lead)}%，接近饱和"
                f"（阈值 {SATURATED_ACC:g}%），近端对模型差异几乎不敏感，"
                "不能单独作为选型依据。"
            )
    return lines


def _section_fa(bundle: Bundle, figures: Mapping[str, str]) -> List[str]:
    first, last, _ = lead_span(bundle.rmse_by_lead[bundle.names[0]].index)
    lines = [
        "## 4.3 全程FA ratio",
        "",
        f"FA仅指Forecast Activity，全程{first}–{last}h统计，不进行时效分段，"
        "以避免与降水False Alarm混淆。",
        "",
        "**全程FA ratio绝对偏差（pp）**",
        "",
    ]
    rows = []
    for variable in bundle.fa_variables:
        values = pd.Series({
            name: BASELINE * float(
                (bundle.fa_ratio_by_lead[name][variable] - 1.0).abs().mean()
            )
            for name in bundle.names
        })
        rows.append([variable] + [fmt_number(values[name], 4) for name in bundle.names]
                    + [_best_cell(values, 4)])
    lines += table_lines(["**变量**"] + list(bundle.names) + ["**更接近1**"], rows)
    lines += [
        "",
        f"![fa_ratio_vs_lead]({figures['fa_plot']})",
        "",
        f"图 4：{len(bundle.fa_variables)}个变量 FA ratio 随预报时效的变化，"
        f"覆盖全程 {first}–{last}h，虚线为理想值 1；纵轴按数据范围缩放。",
        "",
    ]

    items = []
    for variable in bundle.fa_variables:
        values = pd.Series({
            name: BASELINE * float(
                (bundle.fa_ratio_by_lead[name][variable] - 1.0).abs().mean()
            )
            for name in bundle.names
        })
        finite = values.dropna()
        if finite.empty:
            continue
        items.append(
            f"{variable}：绝对偏差最小 {finite.idxmin()}（{fmt_number(finite.min(), 4)} pp），"
            f"最大 {finite.idxmax()}（{fmt_number(finite.max(), 4)} pp）。"
        )
    lines += bullet_lines(items)
    return lines


def _section_spectrum(bundle: Bundle, figures: Mapping[str, str]) -> List[str]:
    lines = ["## 4.4 纬向功率谱", "", "**纬向谱绝对对数偏差（log10）**", ""]
    rows = [[
        bundle.spectrum_variable,
        *[fmt_number(bundle.spectrum_rms[name], 2) for name in bundle.names],
        _best_cell(bundle.spectrum_rms, 2),
    ]]
    lines += table_lines(
        ["**谱变量**"] + list(bundle.names) + ["**更优**"], rows,
    )
    lines += [
        "",
        f"![spectrum]({figures['spectrum_plot']})",
        "",
        f"图 5：{bundle.spectrum_variable} 的纬向功率谱（双对数，"
        f"{date_span(bundle.dates)} 共同日期平均），每条颜色曲线为一个模型的预测谱，"
        "黑色虚线为观测谱。",
        "",
    ]
    items = [
        f"{bundle.spectrum_variable} 的 100×RMS[log10(pred/obs)] 最小 "
        f"{bundle.spectrum_rms.idxmin()}（{fmt_number(bundle.spectrum_rms.min(), 2)}），"
        f"最大 {bundle.spectrum_rms.idxmax()}（{fmt_number(bundle.spectrum_rms.max(), 2)}）。",
        "这是 log10 口径；§4.5 球谐带用自然对数 ln，两者不可直接比大小。",
    ]
    lines += bullet_lines(items)
    return lines


def _section_spherical(
    bundle: Bundle, figures: Mapping[str, str], *, primary: Optional[str] = None
) -> List[str]:
    """§4.5：球谐带功率（表 + 图 + 判读），口径为确定性单成员场。

    §4.5.3 只列主模型 ``primary`` 与其余模型的对照；Holm 校正仍在每个频带的
    全部模型对构成的检验族内完成，因此 p 值与列出全部模型对时一致。
    """
    primary = primary or bundle.names[0]
    # 与 analyze 相同：延迟导入，避开 wave_report -> det_report 的循环。
    from reporting.reports.wave_report import (
        SPHERICAL_ABSENT,
        SPHERICAL_BAND_COUNT,
    )

    names = bundle.names
    lines = ["## 4.5 球谐带功率", ""]
    if not bundle.sph_available:
        lines += [
            f"{SPHERICAL_ABSENT}{bundle.sph_note}",
            "因此本节无表、无图，保留节位以便将来补数后直接填入。",
        ]
        return lines

    band_names = list(bundle.sph_band_logrms.index)
    lines += [
        f"球谐带功率把确定性预报场分解到 {SPHERICAL_BAND_COUNT} 个球谐总阶数频带；"
        "log-RMS×100 越小表示该频带能量越接近观测，均值比越接近 1 越好。"
        "它与 §4.4 的纬向 FFT 谱不是同一分解，不能互相替代。",
        "",
        "### 4.5.1 分频带结果",
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
        f"图 6：{cn_number(len(names))}模型分球谐总阶数频带"
        f"（{' / '.join(band_names)}）的球谐带 log-RMS×100 对比。",
        "",
        f"![spherical_bands_ratio]({figures['spherical_bands_ratio']})",
        "",
        f"图 7：{cn_number(len(names))}模型分球谐总阶数频带的 pred/obs 均值比对比"
        "（虚线为理想值 1）。",
        "",
    ]
    lines += bullet_lines(_spherical_band_notes(bundle))

    lines += ["", "### 4.5.2 分变量结果", "", "**分变量球谐带 log-RMS×100**", ""]
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

    lines += ["", "### 4.5.3 配对检验", ""]
    lines.append(
        f"下表只列主模型 {primary} 与其余 {len(bundle.names) - 1} 个模型的对照；"
        "Holm 校正仍在每个频带的全部模型对构成的检验族内完成，p 值与列出全部模型对时一致。"
    )
    lines.append("")
    test_rows = []
    for band in band_names:
        subset = [
            test for test in bundle.sph_tests.get(band, [])
            if primary in (test.a, test.b)
        ]
        for test in sorted(
            subset, key=lambda t: t.b if t.a == primary else t.a
        ):
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
    lines += bullet_lines(_spherical_test_notes(bundle, primary))
    return lines


def _spherical_band_notes(bundle: Bundle) -> List[str]:
    band_list = list(bundle.sph_band_logrms.index)
    if not band_list:
        return []
    best = {}
    for band in band_list:
        finite = bundle.sph_band_logrms.loc[band].dropna()
        best[band] = finite.idxmin() if not finite.empty else "—"
    return [
        "球谐带 log-RMS×100 最低的模型在"
        + "、".join(f"{band} 段是 {best[band]}" for band in band_list)
        + "；均值比是该频带内 pred/obs 的总平均，越接近 1 表示该尺度上的能量越不偏。",
        f"最高频带（{band_list[-1]}）的总阶数最高、观测功率最小，"
        "比值的对数对微小差异最敏感，该频带的 log-RMS 绝对值不宜与低频带直接比较——"
        "这是频谱类指标的固有特征，不是模型在该尺度上一定更差；"
        "均值比在同一频带仍有意义，因为它不依赖逐点的分母。",
        "按频带看，log-RMS 最低的模型分别是 "
        + "、".join(
            f"{name} 拿下 {sum(1 for band in band_list if best[band] == name)} 个频带"
            for name in bundle.names
        )
        + "。",
    ]


def _spherical_variable_notes(bundle: Bundle) -> List[str]:
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
        + "，按 §4.5.1 的频带口径汇总为 "
        + " < ".join(f"{n}（{fmt_number(by_band[n], 2)}）" for n in order_band)
        + ("，两者一致。" if order_variable == order_band else "，**两者不一致**。")
        + "两者用的是同一批带功率样本，只是聚合维度不同（这里按变量汇总全部频带，"
        "§4.5.1 按频带汇总全部变量），排序不一致时说明模型间的优劣集中在个别频带上，"
        "选型时要指明关注哪一段尺度。"
    ]


def _spherical_test_notes(bundle: Bundle, primary: Optional[str] = None) -> List[str]:
    primary = primary or bundle.names[0]
    band_list = list(bundle.sph_band_logrms.index)
    rows = sum(len(bundle.sph_tests.get(band, [])) for band in band_list)
    significant = sum(
        1 for band in band_list for test in bundle.sph_tests.get(band, [])
        if test.significant
    )
    shown = [
        test for band in band_list for test in bundle.sph_tests.get(band, [])
        if primary in (test.a, test.b)
    ]
    shown_ok = sum(1 for test in shown if test.significant)
    pair_count = len(bundle.names) * (len(bundle.names) - 1) // 2
    tail = (
        f"全部 {rows} 个「频带 × 模型对」组合里有 {significant} 个达到显著，"
        f"表中 {len(shown)} 个主模型对照有 {shown_ok} 个显著；"
        "其余组合的差异被样本量或频带内的离散度淹没；"
        if significant
        else f"全部 {rows} 个「频带 × 模型对」组合均未达显著，"
        "说明各模型在这些总阶数频带上的差异小于日度波动；"
    )
    return [
        f"逐频带在 {len(bundle.dates)} 个日度样本上做 Wilcoxon 符号秩检验，"
        f"每个频带各自对全部 {pair_count} 个模型对做 Holm 校正"
        "（指标是频带本身，同一频带内的模型对构成一个检验族），"
        f"表中只列主模型 {primary} 的对照。"
        + tail
        + "统计显著不代表业务重要性。"
    ]


def _variable_units(bundle: Bundle) -> Dict[str, str]:
    """从 RMSE 行里取各变量的物理单位；取不到就返回空表（轴标签不写单位）。"""
    root = bundle.archives[bundle.names[0]].root
    rows = metric_rows(root, "rmse", product_kind="deterministic")
    if "unit" not in rows.columns:
        return {}
    units: Dict[str, str] = {}
    pairs = rows[["variable", "unit"]].dropna().drop_duplicates()
    for variable, unit in pairs.itertuples(index=False):
        text = str(unit).strip()
        # 同一变量会同时出现 "1"（无量纲指标）和物理单位；跳过 "1"。
        if text in ("", "1", "nan", "None"):
            continue
        variable = str(variable)
        units.setdefault(variable, text.replace("^2", "²").replace("^3", "³"))
    return units


def _lead_band_table(bundle: Bundle) -> pd.DataFrame:
    """分时效段相对综合 RMSE：index=时效段（从短到长）、columns=模型名。"""
    bands: Dict[str, pd.Series] = {}
    for low, high in LEAD_BANDS:
        leads = [
            lead for lead in bundle.lead_relative.index
            if low < float(lead) <= high
        ]
        if not leads:
            continue
        span = f"{float(min(leads)):g}–{float(max(leads)):g}h"
        bands[span] = bundle.lead_relative.loc[leads].mean(axis=0)
    return pd.DataFrame(bands).T.reindex(columns=bundle.names)


def _section_lead(
    bundle: Bundle, figures: Mapping[str, str], figure_number: int = 8
) -> List[str]:
    """§5：分时效段相对综合 RMSE + 点图 + 异常变量敏感性。"""
    variable = bundle.anomaly_variable
    lines = [
        "# 5. 分时效表现",
        "",
        f"分时效分析只覆盖RMSE；FA不参与分段。综合相对RMSE以"
        f"{cn_number(len(bundle.names))}模型的几何均值为100。",
        "",
        f"**分时效段{cn_number(len(bundle.names))}模型指标**",
        "",
    ]
    segment = _lead_band_table(bundle)
    band_order = {span: segment.loc[span] for span in segment.index}
    rows = []
    for span, band in band_order.items():
        for name in band.sort_values().index:
            rows.append([span, str(name), fmt_number(band[name])])
    lines += table_lines(["**时效**", "**模型**", "**相对综合RMSE**"], rows)
    lines.append("")
    items = []
    for span, band in band_order.items():
        order = band.sort_values()
        items.append(
            f"{span}：{order.index[0]} 最低（{fmt_number(order.iloc[0])}），"
            f"{order.index[-1]} 最高（{fmt_number(order.iloc[-1])}）；排序 "
            + " < ".join(f"{n} {fmt_number(band[n])}" for n in order.index) + "。"
        )
    lines += bullet_lines(items)
    lines += [
        "",
        f"![lead_segment_dot]({figures['lead_segment_dot']})",
        "",
        f"图 {figure_number}：分时效段综合相对 RMSE 点图（100 = "
        f"{cn_number(len(bundle.names))}模型几何均值；点越靠左越好）。",
    ]

    lines += ["", "**异常变量敏感性**", ""]
    lines.append(
        f"{variable} 在模型间存在量级差异（值见下表），为确认它是否主导了总体排序，"
        f"下表给出「含 {variable}」与「不含 {variable}」两种口径下的综合相对 RMSE。"
    )
    lines += ["", f"**含/不含{variable}两种口径**", ""]
    without = [v for v in bundle.common_variables if v != variable]
    if without:
        recomputed = _composite(_relative(
            pd.DataFrame({
                name: bundle.rmse_by_date[name][without].mean(axis=0) for name in bundle.names
            }).reindex(without)
        ))
        without_order = recomputed.sort_values().index.tolist()
        without_note = " < ".join(f"{n} {fmt_number(recomputed[n])}" for n in without_order)
        without_cells = [fmt_number(recomputed[name]) for name in bundle.names]
    else:
        without_order = None
        without_note = "没有其他共享变量可比"
        without_cells = ["—"] * len(bundle.names)

    with_order = bundle.relative_composite.sort_values().index.tolist()
    with_note = " < ".join(f"{n} {fmt_number(bundle.relative_composite[n])}" for n in with_order)
    lines += table_lines(
        ["**口径**"] + list(bundle.names) + ["**结论**"],
        [
            [f"含{variable}"] + [fmt_number(bundle.relative_composite[n]) for n in bundle.names] + [with_note],
            [f"不含{variable}"] + without_cells + [without_note],
        ],
    )
    lines.append("")
    if without_order is None or with_order == without_order:
        lines.append(
            f"两种口径下排序完全一致，说明 {variable} 的量级差异虽然明显，"
            f"但没有改变模型间的相对格局，总体结论对该变量不敏感。"
        )
    else:
        lines.append(
            f"两种口径下排序发生变化，说明 {variable} 主导了总体排序；"
            f"在 {variable} 的单位与定义核对清楚之前，不应据此对相关模型下最终结论。"
        )
    return lines


def _section_tests(bundle: Bundle) -> List[str]:
    """§6：综合/ACC/逐变量配对检验与关键差异。"""
    lines = [
        "# 6. 统计检验",
        "",
        f"主检验为共同日期（{len(bundle.dates)} 天）上的**配对 Wilcoxon 符号秩检验**，"
        "每个指标项上把所有模型两两配对；同一项内的多重比较用 **Holm 逐步校正**，"
        "Holm p < 0.05 才记「显著」。检验用逐日样本而不是平均后的单点，"
        f"样本量等于共同日期数。球谐带的逐频带检验见 §4.5.3。",
        "",
        "**综合相对 RMSE 配对检验**",
        "",
    ]
    rows = []
    for test in bundle.overall_tests:
        rows.append([
            f"{test.a} vs {test.b}", _signed(test.delta_pct),
            "是" if test.significant else "否", test.better, fmt_number(test.p_holm, 4),
        ])
    lines += table_lines(
        ["**比较**", "**相对变化 A vs B (%)**", "**Holm 显著**", "**显著优者**", "**Holm p**"],
        rows,
    )

    lines += ["", "**ACC 配对检验**", ""]
    rows = []
    for test in bundle.acc_tests:
        rows.append([
            f"{test.a} vs {test.b}", _signed(test.mean_a - test.mean_b, 3),
            "是" if test.significant else "否", test.better, fmt_number(test.p_holm, 4),
        ])
    lines += table_lines(
        ["**比较**", "**ACC 差 A-B (pp)**", "**Holm 显著**", "**显著优者**", "**Holm p**"],
        rows,
    )

    lines += ["", "**逐变量 RMSE 配对检验汇总**", ""]
    rows = []
    for _, record in bundle.win_counts.iterrows():
        a, b = record["模型对"]
        rows.append([
            f"{a} vs {b}", str(int(record["A 显著更优"])),
            str(int(record["B 显著更优"])), str(int(record["无显著差异"])),
            str(int(record["总变量数"])),
        ])
    lines += table_lines(
        ["**模型对**", "**A 显著更优变量数**", "**B 显著更优变量数**",
         "**无显著差异**", "**总变量数**"],
        rows,
    )

    lines += ["", "**关键配对差异**", ""]
    rows = []
    for test in bundle.overall_tests:
        record = bundle.win_counts[
            bundle.win_counts["模型对"].apply(lambda pair: set(pair) == {test.a, test.b})
        ]
        if record.empty:
            wins = "—"
        else:
            wins_a = int(record.iloc[0]["A 显著更优"])
            wins_b = int(record.iloc[0]["B 显著更优"])
            wins = f"{test.a} {wins_a} : {wins_b} {test.b}"
        explain = (
            f"{test.better} 在综合口径上显著更优"
            if test.significant
            else "未通过 Holm 校正，两者的总体差距不足以判定优劣"
        )
        rows.append([
            f"{test.a} vs {test.b}",
            f"综合相对RMSE 差异 {_signed(test.delta_pct)}%，Holm p = {fmt_number(test.p_holm, 4)}",
            wins, explain,
        ])
    lines += table_lines(["**模型对**", "**综合差异**", "**显著更优变量数**", "**解释**"], rows)
    lines.append("")
    lines.append(
        "逐变量检验用共同日期上的逐日 RMSE 做配对 Wilcoxon，未做多重校正，"
        "「显著」只作为方向性提示；校正后的结论以综合相对 RMSE 与 ACC 两张表为准。"
    )
    return lines


def _season_block() -> List[str]:
    """「附 S 分季节结果（可选）」：固定输出「本批未出」的静态说明。

    这一块**不预生成**：长表里没有季节列，季节要从 ``init_time`` 现推，而分季节
    对比是按需的（用户点名要哪一季、哪个纬度带才切）。口径 2026-09-22 已定，
    工序写在 ``references/rmse-batch-evaluation.md`` §6；这里只留节位与指路，
    **纯静态文本、不含任何占位符**，与 ``weather_rmse_single.md`` 的「附 S」逐字一致。
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
        "  「共同日期 = 共同 `init_date`」上，一组 `init_date` 必须干净地属于一个季节；",
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


def _best_names(series: pd.Series, digits: int = 3, *, mode: str = "min") -> List[str]:
    """并列最优的**全体**名字（``mode="max"`` 改成取最大）。

    两个模型出自同一批产物时相对值会精确相等，只取 ``idxmin`` / ``idxmax``
    挑中的那一个，报告读起来就是「A 全面领先」，而事实是分不出高下。按**显示精度**
    判并列，保证表里印出来的数字和这一列说的是同一件事。
    """
    finite = series.dropna()
    if finite.empty:
        return []
    target = finite.max() if mode == "max" else finite.min()
    key = f"{float(target):.{digits}f}"
    return [
        str(name) for name, value in finite.items() if f"{float(value):.{digits}f}" == key
    ]


def _best_cell(series: pd.Series, digits: int = 3, *, mode: str = "min") -> str:
    """附表「Best」/「更接近1」/「更优」列：并列者逐个列出，不只报挑中的那个。"""
    names = _best_names(series, digits, mode=mode)
    if not names:
        return "—"
    return " / ".join(names) + ("（并列）" if len(names) > 1 else "")


class RegionTable(NamedTuple):
    """「分纬度带综合相对RMSE」那张表的三件套，外加全球口径的冠军。"""

    rows: List[List[str]]                 # 表体（不含表头），一行一个带
    best_of: Dict[str, List[str]]         # 带名 -> 并列第一的全体模型名
    levels: Dict[str, Dict[str, float]]   # 带名 -> 变量 -> 该带该变量平均 RMSE
    best_global: List[str]                # 全球口径并列第一的全体模型名


def region_table(bundle: Bundle) -> RegionTable:
    """构造「分纬度带综合相对RMSE」表。

    det / ens / wave 三份报告的「附 L」第一张表完全同构，共用这一个构造函数——
    各写一遍的话，并列判定之类的修订迟早只在其中一份上生效。
    """
    rows: List[List[str]] = []
    best_of: Dict[str, List[str]] = {}
    levels: Dict[str, Dict[str, float]] = {}
    for region in bundle.region_names:
        series = bundle.region_relative_composite.loc[region]
        best_of[region] = _best_names(series)
        rows.append(
            [region_display(region, bundle.region_labels)]
            + [fmt_number(series[name]) for name in bundle.names]
            + [_best_cell(series)]
        )
        # index=(模型, lead)、columns=变量 → 按变量对「模型 × lead」平均，
        # 得到「这个带里这个变量的 RMSE 量级」。**不跨变量平均**，见 _region_reading。
        stacked = pd.concat(bundle.region_rmse_by_lead[region].values())
        levels[region] = {str(key): float(value) for key, value in stacked.mean(axis=0).items()}
    return RegionTable(rows, best_of, levels, _best_names(bundle.relative_composite))


def _region_reading(
    bundle: Bundle,
    best_global: Mapping[str, Any],
    best_of: Mapping[str, Mapping[str, Any]],
    levels: Mapping[str, Mapping[str, float]],
) -> str:
    """分带判读句：点名名次反转的带 + 带间绝对量级差。

    ``levels`` 是 ``{带名: {变量: 该带该变量的 RMSE}}``。量级差**逐变量算比值再取
    中位数**，不跨变量平均 RMSE——不同变量单位不同（``z500`` 是 m²/s²、``t2m`` 是 K），
    先平均再比大小等于把量纲加在一起，出来的倍数没有意义。
    """
    global_names = list(best_global)
    tied = len(global_names) > 1
    head_global = "、".join(global_names) if global_names else "—"
    tie_note = "（并列，分不出高下）" if tied else ""
    global_value = fmt_number(bundle.relative_composite[global_names[0]]) if global_names else "—"

    flips = [
        (region, list(best_of[region]))
        for region in bundle.region_names
        if set(best_of[region]) != set(global_names)
    ]
    if flips:
        detail = "；".join(
            f"{region_display(region, bundle.region_labels)} 的第一名是 {'、'.join(names)}"
            f"（{fmt_number(bundle.region_relative_composite.loc[region, names[0]])}）"
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

    variables = sorted({v for per_band in levels.values() for v in per_band})
    spreads: List[float] = []
    high_votes: Dict[str, int] = {}
    low_votes: Dict[str, int] = {}
    for variable in variables:
        finite = {
            region: per_band[variable]
            for region, per_band in levels.items()
            if np.isfinite(per_band.get(variable, np.nan)) and per_band.get(variable, 0) > 0
        }
        if len(finite) < 2:
            continue
        high = max(finite, key=lambda key: finite[key])
        low = min(finite, key=lambda key: finite[key])
        high_votes[high] = high_votes.get(high, 0) + 1
        low_votes[low] = low_votes.get(low, 0) + 1
        spreads.append(finite[high] / finite[low])
    if spreads:
        high_band = max(high_votes, key=lambda key: high_votes[key])
        low_band = max(low_votes, key=lambda key: low_votes[key])
        head += (
            f" 各带的绝对 RMSE 量级差（同一变量在各带之间的极差，"
            f"{len(spreads)}个变量的中位数）是 {fmt_number(float(np.median(spreads)), 1)} 倍，"
            f"最大出现在 {region_display(high_band, bundle.region_labels)}、"
            f"最小出现在 {region_display(low_band, bundle.region_labels)}"
            f"——带与带之间只能比相对值，比绝对值没有意义。"
        )
    return head


def _region_block(bundle: Bundle, figures: Mapping[str, str]) -> List[str]:
    """「附 L 分纬度带结果（可选）」。

    没有共同的 region 行时保留节位、写一句「本批未出」并说明原因，不静默省略、
    也不留一张空表——与 §4.5 球谐带同一套约定。
    """
    count = len(bundle.names)
    word = cn_number(count)
    lines = ["## 附 L 分纬度带结果（可选）", ""]
    if not bundle.region_names:
        lines += [f"**本块本批未出。** {bundle.region_note}", ""]
        return lines

    table = region_table(bundle)
    rows, best_of, levels, best_global = (
        table.rows, table.best_of, table.levels, table.best_global,
    )

    lines += [
        "本块把长表里 `region` 非空的行**单独汇总**，全球行**不参与**——全球平均与纬度带平均",
        "是两个量，混在一起算会把带间差异整个抹平，这一块也就没有意义了。",
        "带名与边界照搬评测配置 `options[\"regions\"]`：**报告不翻译、也不写死任何带名**，",
        "配置里叫 `mid_latitudes` 就显示 `mid_latitudes`；要换一套切法改配置重跑即可，",
        "报告这边一个字都不用动。表里的相对RMSE是**对本带基线**取的（100 = 该带内的",
        f"{word}模型几何均值），不是全球基线——热带和极区的绝对 RMSE 差一个量级，",
        "不除本带基线就没法横向比。",
        "",
        "**分纬度带综合相对RMSE**",
        "",
    ]
    lines += table_lines(["纬度带"] + list(bundle.names) + ["Best"], rows)
    lines += ["", _region_reading(bundle, best_global, best_of, levels), ""]
    lines += [
        f"![lat_band_rmse_vs_lead]({figures['lat_band_lead']})",
        "",
        f"图 L1：各纬度带综合相对RMSE随预报时效的变化"
        f"（100 = 该带内{word}模型几何均值；{date_span(bundle.dates)} 共同日期平均）",
        "",
        f"![lat_band_summary]({figures['lat_band_summary']})",
        "",
        f"图 L2：分纬度带综合相对RMSE"
        f"（100 = 该带内{word}模型几何均值；{date_span(bundle.dates)} 共同日期平均）",
    ]
    return lines


def _appendix(
    bundle: Bundle, figures: Mapping[str, str], last_figure: int = 8
) -> List[str]:
    """附录：补充诊断图（附 L、附 S）。"""
    count = len(bundle.names)
    lines = [
        "# 附录：补充诊断图",
        "",
        f"本附录使用{count}个模型共同的{len(bundle.dates)}个日期"
        f"（{bundle.dates[0]} 至 {bundle.dates[-1]}），与正文全部统计同一批日期。"
        f"正文图 1–{last_figure} 已经承载主结论；这里只放**不参与主排序**的诊断图："
        "可选的分纬度带与分季节结果。",
        "",
        "各块的图号独立编号（L1–L2、S1–S2），整块可缺——缺时写「本批未出」"
        "及原因，**不留空号、也不指不存在的文件**。",
        "",
    ]
    lines += _region_block(bundle, figures)
    lines += [""]
    lines += _season_block()
    return lines


def _overall_heatmap(out_dir: Path, filename: str, bundle: Bundle) -> Path:
    """总体指标热力图（``图 1``）：行是模型、列是指标，格内数字是原始值。

    颜色是**列内**批内归一化的相对表现（1 = 本批最优、0 = 最差），归一化复用
    能力雷达图的 ``radar_scores``（TS 报告也走同一份实现）。各指标量纲不同，
    颜色**只在同一列内可比**——图注必须写明，否则红蓝会被当成一把尺子。

    行按综合相对 RMSE 从好到差排；ACC 越高越好，其余指标越低越好。球谐指标
    不可得时那两列整列不画（不是画成空白列）。
    """
    names = sorted(bundle.names, key=lambda name: bundle.relative_composite[name])
    columns = [
        ("RMSE", bundle.relative_composite, False, 2),
        ("ACC (%)", bundle.acc_mean, True, 2),
        ("FA bias (pp)", bundle.fa_bias, False, 2),
        ("Spectrum dev.", bundle.spectrum_rms, False, 2),
    ]
    if bundle.sph_available:
        columns += [
            ("Sph. RMS", bundle.sph_rms, False, 2),
            ("Sph. bias (%)", bundle.sph_bias, False, 2),
        ]

    figure = plot_overall_heatmap(names, columns)
    path = Path(out_dir) / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path)
    plt.close(figure)
    print(f"已保存: {path}")
    return path


def _check_figure_contract(
    text: str,
    count: int = 8,
    required: Sequence[int] = (1, 2, 3, 4, 5, 8),
    paired: Sequence[int] = (6, 7),
) -> None:
    """骨架的编号契约：``图 N：`` 的分布必须合法。

    - ``图 N：``（``N`` 在 ``1..count``）**最多出现一次**，且首现位置递增；
    - ``required`` 里的图必出（1 总体热力图、2–5 §4.1–§4.4、8 §5 分时效点图）；
    - ``paired`` 是球谐带的两张图，**成对出现或成对缺席**：球谐不可得时整块
      写「本批未出」，不能只画一张；
    - ``图 L1..：`` 从 L1 起连续（分纬度带整块可缺）。
    """
    required = tuple(int(index) for index in required)
    paired = tuple(int(index) for index in paired)
    positions = []
    for index in range(1, count + 1):
        marker = f"图 {index}："
        found = text.count(marker)
        if found > 1:
            raise ValueError(f"图注契约被破坏：{marker} 出现 {found} 次，最多 1 次")
        if found == 0:
            if index in required:
                raise ValueError(f"图注契约被破坏：{marker} 缺失（必出图）")
            continue
        positions.append((index, text.index(marker)))
    if [where for _, where in positions] != sorted(where for _, where in positions):
        raise ValueError(f"图注契约被破坏：图 1..{count} 的首现位置不是递增的")

    present = [f"图 {index}：" for index in paired if f"图 {index}：" in text]
    if present and len(present) != len(paired):
        raise ValueError(
            f"图注契约被破坏：球谐带两张图必须成对出现，现在是 {present}"
        )

    band =[f"图 L{index}：" for index in range(1, 3)]
    present = [marker for marker in band if marker in text]
    # 允许缺席，但必须是**从 L1 起的连续段**，不能有 L1 没 L2 之外的跳号。
    if present != band[: len(present)]:
        raise ValueError(
            f"图注契约被破坏：分纬度带的图号必须从 图 L1 起连续，现在是 {present}"
        )
    for marker in present:
        if text.count(marker) != 1:
            raise ValueError(f"图注契约被破坏：{marker} 出现 {text.count(marker)} 次")


# ------------------------------------------------------------------ 装配


def report_date() -> str:
    """报告**生成**日期（``YYYYMMDD``）。

    抬头那个「报告日期」是跑这份报告的当天，**不是数据末日期**——两者很容易
    被看成一回事，但含义完全不同：数据覆盖到哪天，正文第 1/2 节已经写了
    「覆盖 N 天（起，止）」，抬头再写一遍等于把同一个信息抄两处，还挤掉了
    「这份报告是什么时候出的」这个只有抬头能承载的信息。

    代价是产物**不再逐位可复现**：同一天跑两次一样，隔天跑就不同。
    """
    return datetime.now().strftime("%Y%m%d")


def render(
    bundle: Bundle,
    *,
    title: Optional[str] = None,
    change: Optional[str] = None,
    archive: Optional[str] = None,
    declared: Optional[Mapping[str, Sequence[str]]] = None,
    primary: Optional[str] = None,
    figures: Optional[Mapping[str, str]] = None,
    spatial: Optional[SpatialPayload] = None,
) -> str:
    """把 :class:`Bundle` 渲染成完整 Markdown（六章骨架 + 附录）。

    ``figures`` 只传实际落盘的图；没传的键用这里的默认文件名兜底。默认名与
    :func:`build_det_report` 的落盘名一致（均为 SVG）。
    ``primary`` 是 §4.5.3 配对检验里作为主对象的模型；None 用 ``--model``
    给出的第一个模型。
    ``spatial`` 是 §4.6 的月平均空间分布；None 时该节保留并写「本批未出」，
    图号也不动（§5 仍是图 8）。
    """
    primary = primary or bundle.names[0]
    if primary not in bundle.names:
        raise ValueError(f"主模型 {primary!r} 不在模型列表 {bundle.names} 里")
    count = len(bundle.names)
    defaults = {
        "overall_heatmap": "overall_heatmap.svg",
        "rmse_plot": "rmse_vs_lead.svg",
        "acc_plot": "acc_vs_lead.svg",
        "fa_plot": "fa_ratio_vs_lead.svg",
        "spectrum_plot": f"spectrum_{bundle.spectrum_variable}.svg",
        "spherical_bands_logrms": "spherical_bands_logrms.svg",
        "spherical_bands_ratio": "spherical_bands_ratio.svg",
        "lead_segment_dot": "lead_segment_dot.svg",
        "lat_band_lead": "lat_band_rmse_vs_lead.svg",
        "lat_band_summary": "lat_band_summary.svg",
    }
    figures = {**defaults, **dict(figures or {})}
    if spatial is not None:
        # 空间图键由 payload 自带，避免渲染器和装配端各写一份文件名规则。
        figures.update({key: path for key, path in spatial.figures.items()})
    spatial_k = spatial.data.figure_count if spatial is not None else 0
    lead_figure = 8 + spatial_k
    first, last, leads = lead_span(bundle.rmse_by_lead[bundle.names[0]].index)

    lines: List[str] = []
    lines.append(title or f"# 确定性预报{cn_number(count)}模型综合评估报告")
    lines += ["", f"**产物：{archive or '、'.join(bundle.names)}**", ""]
    lines += [
        f"报告日期：{report_date()}    评估层级：输出级复核",
        "",
    ]
    lines += _conclusion_summary(bundle, change, first=first, last=last, leads=leads)
    lines += [""]
    lines += _section_samples(bundle, declared)
    lines += [""]
    lines += _section_metrics(bundle)
    lines += [""]
    lines += _section_overall(bundle, figures)
    lines += [""]
    lines += _section_rmse(bundle, figures)
    lines += [""]
    lines += _section_acc(bundle, figures)
    lines += [""]
    lines += _section_fa(bundle, figures)
    lines += [""]
    lines += _section_spectrum(bundle, figures)
    lines += [""]
    lines += _section_spherical(bundle, figures, primary=primary)
    lines += [""]
    lines += build_spatial_section(
        spatial, heading="## 4.6 空间分布图", figure_start=8
    )
    lines += [""]
    lines += _section_lead(bundle, figures, figure_number=lead_figure)
    lines += [""]
    lines += _section_tests(bundle)
    lines += [""]
    lines += _appendix(bundle, figures, last_figure=lead_figure)
    lines.append("")

    text = "\n".join(lines)
    _check_figure_contract(
        text,
        count=lead_figure,
        required=(1, 2, 3, 4, 5, *range(8, 9 + spatial_k)),
        paired=(6, 7),
    )
    return text


def _conclusion_summary(
    bundle: Bundle,
    change: Optional[str],
    *,
    first: float,
    last: float,
    leads: int,
) -> List[str]:
    order = bundle.relative_composite.sort_values().index.tolist()
    best, worst = order[0], order[-1]
    items = [
        f"综合相对 RMSE 最好 {best}（{fmt_number(bundle.relative_composite[best])}），"
        f"最差 {worst}（{fmt_number(bundle.relative_composite[worst])}）。",
        f"ACC 最好 {bundle.acc_mean.idxmax()}（{fmt_number(bundle.acc_mean.max())}%），"
        f"FA 偏差最小 {bundle.fa_bias.idxmin()}（{fmt_number(bundle.fa_bias.min(), 4)} pp），"
        f"频谱对数 RMS×100 最小 {bundle.spectrum_rms.idxmin()}（{fmt_number(bundle.spectrum_rms.min(), 2)}）。",
        f"平均排名分："
        + "、".join(f"{name} {fmt_number(bundle.rank_score[name], 1)}" for name in bundle.rank_score.sort_values().index)
        + "。",
    ]
    if bundle.sph_available:
        items.append(
            f"球谐带功率上，{bundle.sph_rms.idxmin()} 的总 log-RMS×100 最低"
            f"（{fmt_number(bundle.sph_rms.min(), 2)}），分频带结论见 §4.5。"
        )
    significant = [t for t in bundle.overall_tests if t.significant]
    if significant:
        items.append(
            "综合相对 RMSE 上通过 Holm 校正的显著差异："
            # 方向只能来自检验结论（t.better），不能拿参数顺序 t.a/t.b 当优劣——
            # 否则 B 更优时会写成「A 优于 B」，和第三节的表直接打架。
            + "；".join(
                f"{t.better} 优于 {t.b if t.better == t.a else t.a}（p={fmt_number(t.p_holm, 4)}）"
                for t in significant
            )
            + "。"
        )
    else:
        items.append(
            "综合相对 RMSE 上没有任何模型对通过 Holm 校正（p < 0.05），"
            "模型间的总体差距在统计上不稳健，选型应更依赖分维度表现。"
        )
    items.append(
        f"{bundle.anomaly_variable} 存在模型间量级差异"
        f"（{bundle.anomaly_model} 为其余模型中位数的 {fmt_number(bundle.anomaly_ratio, 2)} 倍），"
        f"已在 §5 的敏感性分析中单独说明。"
    )
    return [
        "# 1. 摘要",
        "",
        # 先交代评估基本信息（评的是谁、多少天、什么指标），再给结论
        f"本报告评估{'、'.join(bundle.names)}共{len(bundle.names)}个模型，"
        f"覆盖{len(bundle.dates)}天（{date_span(bundle.dates)}）；"
        f"口径为RMSE、ACC、FA、纬向谱与球谐带功率。正文与附录统一使用同一批共同日期，"
        f"逐日结果按 lead 平均后比较，非共同日期不参与任何统计。"
        + (f"本次相对上一版的改动：{change}" if change else ""),
        "",
        # 结论并成一段而不是逐条列点：读起来是判断，不是清单
        "".join(items),
        "",
        f"本报告中的FA指Forecast Activity，统一使用全程{first}–{last}h、"
        f"{leads}个lead，不对FA做短中长时效分段。",
        "",
    ]


# ------------------------------------------------------------------ 入口


def build_det_report(
    archives: Sequence[Archive],
    out_dir: Path,
    *,
    title: Optional[str] = None,
    change: Optional[str] = None,
    archive: Optional[str] = None,
    declared: Optional[Mapping[str, Sequence[str]]] = None,
    primary: Optional[str] = None,
    spatial_csv: Optional[Path] = None,
    spatial_model: Optional[str] = None,
) -> Dict[str, Path]:
    """出图 + 写 ``REPORT.md``，返回产物名 → 路径。

    ``primary`` 只影响 §4.5.3 配对检验列哪些模型对；None 用第一个模型。
    ``spatial_csv`` 是 §4.6 的月平均宽表（单模型）；None 时该节写「本批未出」。
    ``spatial_model`` 是图里的预报名称，缺省为 ``primary``。

    Raises:
        ValueError / FileNotFoundError: 见 :func:`analyze`。
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 先读空间 CSV 再算全量统计：CSV 缺列/网格错序时不该让人等完 analyze。
    spatial_data = load_spatial_csv(Path(spatial_csv)) if spatial_csv else None

    bundle = analyze(archives, declared=declared)
    plotter = DetPlotter()

    figures = {
        "overall_heatmap": "overall_heatmap.svg",
        "rmse_plot": "rmse_vs_lead.svg",
        "acc_plot": "acc_vs_lead.svg",
        "fa_plot": "fa_ratio_vs_lead.svg",
        "spectrum_plot": f"spectrum_{bundle.spectrum_variable}.svg",
        "lead_segment_dot": "lead_segment_dot.svg",
    }

    # 图 1：总体指标热力图。归一化与 TS 报告的能力雷达共用 radar_scores。
    _overall_heatmap(out_dir, figures["overall_heatmap"], bundle)

    # 图 2：4 行 × 3 列。10 个变量占 10 格，图例放第 4 行最右的空格，
    # 中间那格留白。实线、无标记点、不画总标题；面板标题用 (a)(b)…
    # 左对齐（与集合报告同一套），字号比默认档再大一档。每个面板左上角
    # 加一个 6–11h 的放大 inset（灰实线框圈出前期的一小段，引导线连到 inset）。
    rmse_series = {name: bundle.rmse_by_lead[name] for name in bundle.names}
    plotter.plot_model_panels(
        rmse_series, bundle.common_variables, ylabel="RMSE",
        units=_variable_units(bundle),
        ncols=3, panel_size=(3.6, 2.9), linewidth=1.8, marker=None,
        zoom_y=True, xlabel="Forecast lead time (h)",
        tick_labelsize=12, label_fontsize=14, title_fontsize=14,
        legend_fontsize=12, legend_ncols=2, legend_cell=0,
        zoom_windows=[(6.0, 11.0)],
        zoom_positions=[(0.04, 0.55, 0.33, 0.37)],
        zoom_linewidth=1.2, zoom_tick_labelsize=9, zoom_x_tick_step=2.0,
        grid=False, minimal_spines=True, panel_letters=True,
        save_path=out_dir / figures["rmse_plot"],
    )

    acc_series = {
        name: bundle.acc_by_lead[name].reindex(columns=[bundle.acc_variable])
        for name in bundle.names
    }
    plotter.plot_model_panels(
        acc_series, [bundle.acc_variable], ylabel="ACC", ncols=1,
        units={bundle.acc_variable: "%"},
        ref_line=100.0, ref_line_label=None,
        panel_size=(7.2, 3.6), linewidth=1.8, marker=None,
        zoom_y=True, xlabel="Forecast lead time (h)",
        grid=False, minimal_spines=True,
        legend_box=True, legend_ncols=min(len(bundle.names), 6),
        save_path=out_dir / figures["acc_plot"],
    )

    # 图 4：5 个变量排 2 列，最后一张**左对齐**在第 3 行左侧，图例放它右边的
    # 空格里（不再居中占满整行）。面板标题同样用 (a)(b)… 左对齐。
    plotter.plot_model_panels(
        {name: bundle.fa_ratio_by_lead[name] for name in bundle.names},
        bundle.fa_variables, ylabel="FA ratio", ref_line=1.0,
        ref_line_label=None,
        ncols=2, panel_size=(3.6, 2.9),
        linewidth=1.8, marker=None, zoom_y=True,
        xlabel="Forecast lead time (h)",
        grid=False, minimal_spines=True, panel_letters=True,
        save_path=out_dir / figures["fa_plot"],
    )

    plotter.plot_model_spectrum(
        bundle.spectrum, bundle.spectrum_variable,
        save_path=out_dir / figures["spectrum_plot"],
    )

    # 图 6/7：球谐带功率（成对出现或成对缺席）。`_grouped_bars` 在 wave_report 里，
    # 而 wave_report 顶层 import 了本模块——只能函数内延迟导入，否则循环导入。
    if bundle.sph_available:
        from reporting.reports.wave_report import _grouped_bars

        figures["spherical_bands_logrms"] = "spherical_bands_logrms.svg"
        _grouped_bars(
            out_dir, figures["spherical_bands_logrms"], bundle.sph_band_logrms,
            suptitle=None,
            ylabel="log-RMS×100",
            y_grid=False,
        )
        figures["spherical_bands_ratio"] = "spherical_bands_ratio.svg"
        _grouped_bars(
            out_dir, figures["spherical_bands_ratio"], bundle.sph_band_ratio,
            suptitle=None,
            ylabel="pred / obs",
            ref_line=1.0,
            y_grid=False,
        )

    # 图 8：分时效段点图。与 §5 的表用同一个 helper，避免两处各算一遍。
    plotter.plot_segment_dots(
        _lead_band_table(bundle),
        xlabel="Relative RMSE Index",
        ylabel="Lead-time band",
        save_path=out_dir / figures["lead_segment_dot"],
    )

    # 分纬度带（可选块）：产物里没有共同的 region 行就**不出图**，也不往 artifacts 里
    # 塞不存在的路径——报告那边会写「本批未出」，指一张没生成的图比不指更糟。
    if bundle.region_names:
        band_labels = [
            region_display(region, bundle.region_labels) for region in bundle.region_names
        ]
        figures["lat_band_lead"] = "lat_band_rmse_vs_lead.svg"
        figures["lat_band_summary"] = "lat_band_summary.svg"
        plotter.plot_model_panels(
            {
                name: pd.DataFrame({
                    label: bundle.region_lead_relative[region][name]
                    for region, label in zip(bundle.region_names, band_labels)
                })
                for name in bundle.names
            },
            band_labels,
            ylabel="综合相对 RMSE（100 = 该带内几何均值）",
            ref_line=BASELINE,
            ncols=min(len(band_labels), 3),
            suptitle="分纬度带综合相对 RMSE 随预报时效的变化",
            save_path=out_dir / figures["lat_band_lead"],
        )
        summary = bundle.region_relative_composite.copy()
        summary.index = band_labels
        plotter.plot_group_bars(
            summary,
            ylabel="综合相对 RMSE（100 = 该带内几何均值）",
            ref_line=BASELINE,
            suptitle="分纬度带综合相对 RMSE",
            save_path=out_dir / figures["lat_band_summary"],
        )

    # §4.6 空间分布：确定性版固定三行（CSV 里即便有 spread 列也不画）。
    spatial = None
    if spatial_data is not None:
        model = spatial_model or primary or bundle.names[0]
        paths = render_spatial(spatial_data, out_dir, model=model, mode="det")
        figures.update({key: path.name for key, path in paths.items()})
        spatial = SpatialPayload(
            data=spatial_data,
            model=model,
            mode="det",
            figures={key: path.name for key, path in paths.items()},
        )

    # DetPlotter 不关图（既有行为），这里统一收一次，避免长批次累积。
    plt.close("all")

    text = render(
        bundle, title=title, change=change, archive=archive,
        declared=declared, primary=primary, figures=figures, spatial=spatial,
    )
    report_path = out_dir / "REPORT.md"
    report_path.write_text(text, encoding="utf-8", newline="\n")
    print(f"已保存: {report_path}")

    artifacts: Dict[str, Path] = {"report": report_path}
    for key, filename in figures.items():
        artifacts[key] = out_dir / filename
    return artifacts
