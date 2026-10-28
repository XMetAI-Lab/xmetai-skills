#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""连续场评分（RMSE / ACC / 活跃度 / 纬向谱）的出图器。

输入是 ``weather_field_scores`` 流程的 ``scores.csv``（至少含
``variable / metric / lead_h / value``），输出 PNG；报告正文见 :mod:`field_report`。

四条与"一张图一根轴画到底"不同的做法：折线画**跨起报均值**（逐条连线只是
按起报顺序穿插，看不出形态），离散程度用分位带；RMSE **按单位分面**，不混轴；
热力图用**相对首时效的倍数**，跨变量才可比；谱读
``diagnostics/spectrum_<变量>.csv``，只有要看单个起报时才读 ``spectrum_by_init.csv``。
参考线只在语义表给了 ``ref_result`` 时画。
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from reporting.reports.field_report import (
    SKILL_METRICS,
    build_field_report,
    global_rows,
    metric_label,
    metric_ref,
    requested_from,
)
from reporting.core.style import (
    CMAP_SEQ,
    NEUTRAL,
    apply_style,
    font_sizes,
    snap_width,
)

REQUIRED_COLUMNS = ("lead_h", "variable", "metric", "value")

#: 折线图里跨起报离散带取的分位。用 10–90 而不是 min–max：后者由单次极端起报
#: 决定上下沿，带子会宽得看不出形态。
_SPREAD_QUANTILES = (0.10, 0.90)

#: 谱曲线产物的位置与表头。逐波数曲线**不在** ``scores.csv`` 里（长表一行一个
#: 变量的总量），走 ``MetricResult.curve``，由 ``components.py`` 落到
#: ``<产物目录>/diagnostics/`` 下，两个文件：
#:
#: * ``spectrum_<变量>.csv``：``wavenumber, wavelength_km, pred_mean, obs_mean``，
#:   全时段加权平均，一个变量一张——面板与分波段表用它；
#: * ``spectrum_by_init.csv``：``variable, init_time, wavenumber, wavelength_km,
#:   pred, obs``，逐起报，一个文件装所有变量——点名看某一个起报用它。
SPECTRUM_DIRNAME = "diagnostics"
SPECTRUM_GLOBAL_PREFIX = "spectrum_"
SPECTRUM_BY_INIT_NAME = "spectrum_by_init.csv"
_SPECTRUM_GLOBAL_COLUMNS = ("wavenumber", "pred_mean", "obs_mean")
_SPECTRUM_BY_INIT_COLUMNS = ("variable", "init_time", "wavenumber", "pred", "obs")

#: 赤道周长（km）。纬向波数 k 的波长按 ``40075 / k`` 算，与参考实现的
#: ``mean_spectrum_<var>.csv`` 里 ``wavelength_km`` 列同口径（k=1 → 40075km）。
EQUATOR_CIRCUMFERENCE_KM = 40075.0


def prepare_field_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """校验并规范化连续场长表。

    保留 ``status != success`` 的行（报告要数它们），出图时由各方法自行过滤。

    Raises:
        ValueError: 缺少必需列，或数据为空 / 没有可用的 ``lead_h``。
    """
    data = global_rows(df.copy())
    data.columns = [str(column).strip() for column in data.columns]

    missing = [column for column in REQUIRED_COLUMNS if column not in data.columns]
    if missing:
        raise ValueError(f"连续场长表缺少必需列 {missing}；现有列: {list(data.columns)}")
    if data.empty:
        raise ValueError("连续场长表是空的")

    for column in ("lead_h", "value"):
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data = data.dropna(subset=["lead_h"])
    if data.empty:
        raise ValueError("连续场长表里没有可用的 lead_h 数值")

    data["variable"] = data["variable"].astype(str)
    data["metric"] = data["metric"].astype(str)
    if "unit" not in data.columns:
        data["unit"] = ""
    data["unit"] = data["unit"].fillna("").astype(str)
    if "status" not in data.columns:
        data["status"] = "success"
    data["status"] = data["status"].fillna("success").astype(str)
    return data.sort_values(["metric", "variable", "lead_h"])


def _usable(data: pd.DataFrame) -> pd.DataFrame:
    """只留 ``status=success`` 且 ``value`` 非空的行。"""
    return data[(data["status"] == "success") & data["value"].notna()]


def _unit_of(chunk: pd.DataFrame) -> str:
    texts = [str(unit) for unit in chunk["unit"].dropna().unique() if str(unit)]
    return texts[0] if texts else ""


def _axis_label(metric: str, unit: str) -> str:
    """纵轴标签：无量纲的 ``1`` 不写出来，其余原样带上方括号。"""
    label = metric_label(metric)
    if not unit or unit == "1":
        return label
    return f"{label} [{unit}]"


def lead_aggregate(chunk: pd.DataFrame) -> pd.DataFrame:
    """把「逐起报」的行压成「逐时效」的均值 / 分位带 / 起报数。

    口径就是 ``summary.csv`` 里 ``scope=lead`` 那一档（跨起报算术平均），
    所以图上的线能和落盘的数字对上。

    Args:
        chunk: 同一 ``变量 × 指标`` 的可用行，要有 ``lead_h`` 与 ``value``。

    Returns:
        以 ``lead_h`` 升序为索引的 DataFrame，列为
        ``mean`` / ``low`` / ``high`` / ``n``（``n`` 是该时效的起报数）。
    """
    grouped = chunk.groupby("lead_h", observed=True)["value"]
    stats = pd.DataFrame(
        {
            "mean": grouped.mean(),
            "low": grouped.quantile(_SPREAD_QUANTILES[0]),
            "high": grouped.quantile(_SPREAD_QUANTILES[1]),
            "n": grouped.size(),
        }
    )
    return stats.sort_index()


def _spectrum_directory(directory: Path) -> Path:
    """定位谱曲线所在的目录：给产物目录就进 ``diagnostics/``，给的已经是它就原样用。"""
    directory = Path(directory)
    nested = directory / SPECTRUM_DIRNAME
    return nested if nested.is_dir() else directory


def spectrum_files(directory: Path) -> Dict[str, Path]:
    """``{变量: 全时段平均谱文件}``，来自 ``diagnostics/spectrum_<变量>.csv``。

    只认**逐变量**那一种；``spectrum_by_init.csv`` 是同目录下的另一个文件
    （逐起报），前缀相同，要显式排掉——否则它会被当成一个叫 ``by_init`` 的变量。
    """
    root = _spectrum_directory(directory)
    found: Dict[str, Path] = {}
    for path in sorted(root.glob(f"{SPECTRUM_GLOBAL_PREFIX}*.csv")):
        if path.name == SPECTRUM_BY_INIT_NAME:
            continue
        variable = path.stem[len(SPECTRUM_GLOBAL_PREFIX) :]
        if variable:
            found[variable] = path
    return found


def _spectrum_long(frame: pd.DataFrame, variable: str, pred: str, obs: str) -> pd.DataFrame:
    """``variable / wavenumber / field / value`` 长表——本模块谱图的统一入口口径。"""
    wide = frame.rename(columns={pred: "power_forecast", obs: "power_observation"})
    long = wide.melt(
        id_vars=["wavenumber"],
        value_vars=["power_forecast", "power_observation"],
        var_name="field",
        value_name="value",
    )
    long["variable"] = str(variable)
    long["wavenumber"] = pd.to_numeric(long["wavenumber"], errors="coerce")
    long["value"] = pd.to_numeric(long["value"], errors="coerce")
    return long.dropna(subset=["wavenumber", "value"])


def load_wavenumber_spectra(directory: Path) -> pd.DataFrame:
    """读产物目录里 ``diagnostics/spectrum_<变量>.csv`` 的逐波数谱曲线。

    这是**已经跨起报、跨时效加权平均**好的一张表（跟参考实现的
    ``mean_spectrum_<var>.csv`` 同口径），所以一个变量一张谱、一次读全。

    口径差别要说清楚：这里的谱是**全时段平均**的，看不出谱随预报时效怎么退化；
    要看某一个起报的谱，用 :func:`load_init_spectra` 那条路。

    Args:
        directory: 产物目录（含 ``scores.csv`` 与 ``diagnostics/``），
            也可以直接给 ``diagnostics`` 目录本身。

    Returns:
        长表，列为 ``variable / wavenumber / field / value``，
        ``field`` ∈ {``power_forecast``, ``power_observation``}。

    Raises:
        ValueError: 一个 ``spectrum_<变量>.csv`` 都没找到，或文件缺列。
    """
    files = spectrum_files(directory)
    if not files:
        raise ValueError(
            f"{_spectrum_directory(directory)} 下没有 {SPECTRUM_GLOBAL_PREFIX}<变量>.csv："
            f"这条流程的配置里没写 `zonal_spectrum` / `spectrum`，"
            f"或者纬度方向的格点数不足以做纬向谱"
        )
    frames = []
    for variable, path in files.items():
        frame = pd.read_csv(path)
        missing = [name for name in _SPECTRUM_GLOBAL_COLUMNS if name not in frame.columns]
        if missing:
            raise ValueError(f"{path} 缺少列 {missing}；现有列: {list(frame.columns)}")
        frames.append(_spectrum_long(frame, variable, "pred_mean", "obs_mean"))
    return (
        pd.concat(frames, ignore_index=True)
        .sort_values(["variable", "wavenumber"])
        .reset_index(drop=True)
    )


def load_init_spectra(
    directory: Path, variable: str, init: Optional[str] = None
) -> pd.DataFrame:
    """读 ``diagnostics/spectrum_by_init.csv`` 里某个变量某个起报的谱曲线。

    ``spectrum_by_init.csv`` 的每一行是该起报**跨时效加权平均**后的功率，
    不是某个具体时效的谱——长表那条链路的曲线只按起报存，没有逐时效的谱。

    Args:
        directory: 产物目录（含 ``diagnostics/``），也可以直接给 ``diagnostics``。
        variable: 变量名。
        init: 起报时刻，按**子串**匹配 ``init_time``（写 ``20250102`` 或整条
            ISO 串都行）。``None`` 取最后一个起报。

    Raises:
        ValueError: 文件不存在 / 缺列 / 该变量在该起报下没有谱。
    """
    path = _spectrum_directory(directory) / SPECTRUM_BY_INIT_NAME
    if not path.is_file():
        raise ValueError(f"{path} 不存在")
    frame = pd.read_csv(
        path, usecols=lambda name: name in _SPECTRUM_BY_INIT_COLUMNS
    )
    missing = [name for name in _SPECTRUM_BY_INIT_COLUMNS if name not in frame.columns]
    if missing:
        raise ValueError(f"{path} 缺少列 {missing}；现有列: {list(frame.columns)}")
    frame = frame[frame["variable"].astype(str) == str(variable)].copy()
    if frame.empty:
        known = sorted(set(pd.read_csv(path, usecols=["variable"])["variable"].astype(str)))
        raise ValueError(
            f"{path} 里没有 variable={variable!r} 的谱；有谱的是 {'、'.join(known) or '（无）'}"
        )
    frame["init_time"] = frame["init_time"].astype(str)
    if init is None:
        chosen = frame["init_time"].max()
    else:
        # 去掉日期里的连字符再匹配：``init_time`` 是 ISO 串，让人写
        # ``20250102``（跟别的产物目录名同一种写法）比逼人抄整条 ISO 串友好。
        haystack = frame["init_time"].str.replace("-", "", regex=False)
        needle = str(init).replace("-", "")
        matches = frame.loc[haystack.str.contains(needle, regex=False), "init_time"]
        if matches.empty:
            raise ValueError(
                f"{variable} 没有起报时刻含 {init!r} 的谱；"
                f"现有起报：{'、'.join(sorted(set(frame['init_time'])))}"
            )
        chosen = matches.max()
    frame = frame[frame["init_time"] == chosen]
    long = _spectrum_long(frame, variable, "pred", "obs")
    if long["wavenumber"].le(0).all():
        raise ValueError(f"{variable} @ {chosen} 的谱线只有 k=0，没有可画的波数")
    long = long.sort_values(["variable", "wavenumber"]).reset_index(drop=True)
    # 真正选中的那个起报（调用方可能只给了子串、或者干脆没给），回传出去给图题用
    long.attrs["init_time"] = str(chosen)
    return long


def spectrum_series(spectra: pd.DataFrame, variable: str) -> pd.DataFrame:
    """把一个变量的谱帧整成 ``wavenumber × {power_forecast, power_observation}``。

    Raises:
        ValueError: 该变量缺预报或实况其中一条谱线。
    """
    subset = spectra[spectra["variable"].astype(str) == str(variable)]
    wide = subset.pivot_table(
        index="wavenumber", columns="field", values="value", aggfunc="mean", observed=True
    ).sort_index()
    for field in ("power_forecast", "power_observation"):
        if field not in wide.columns:
            raise ValueError(
                f"{variable} 的谱只有 {list(wide.columns)}，缺 {field}，画不了两条线"
            )
    return wide


class FieldScorePlotter:
    """连续场评分结果的绘图器。"""

    def __init__(self, style: Optional[str] = None):
        """
        Args:
            style: 可选 matplotlib 样式名；默认不套样式，
                避免样式把中文字体设置覆盖掉（降水那边就栽在这）。
        """
        self.font = apply_style()
        if style:
            plt.style.use(style)
            self.font = apply_style()
        plt.rcParams["figure.dpi"] = 150
        plt.rcParams["savefig.dpi"] = 150
        plt.rcParams["savefig.bbox"] = "tight"

    # ------------------------------------------------------------------ #
    def _save(self, fig, save_path: Optional[Path]) -> None:
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path)
            print(f"已保存: {save_path}")

    def _subset(self, df: pd.DataFrame, metric: str) -> pd.DataFrame:
        data = _usable(prepare_field_dataframe(df))
        subset = data[data["metric"] == str(metric)]
        if subset.empty:
            raise ValueError(f"长表里没有 metric={metric!r} 的可用数据")
        return subset

    # ------------------------------------------------------------------ #
    def plot_metric_vs_lead(
        self,
        df: pd.DataFrame,
        metric: str,
        *,
        variables: Optional[Sequence[str]] = None,
        title: Optional[str] = None,
        save_path: Optional[Path] = None,
    ):
        """一个变量一条线；**按单位分面**，一个单位一个子图。

        线是**跨起报均值**（口径同 ``summary.csv`` 的 ``scope=lead``），
        阴影是跨起报的 10–90 分位。每个时效只有 1 个起报时不画阴影——
        带子宽度恒为 0，画了只会让人以为"离散很小"。
        """
        subset = self._subset(df, metric)
        if variables is not None:
            subset = subset[subset["variable"].isin([str(name) for name in variables])]
            if subset.empty:
                raise ValueError(f"指定的 variables 在 metric={metric!r} 里一个都没有")

        grouped: Dict[str, List[str]] = {}
        for variable, chunk in subset.groupby("variable", observed=True):
            grouped.setdefault(_unit_of(chunk), []).append(str(variable))
        units = sorted(grouped)

        cycle = list(plt.rcParams["axes.prop_cycle"].by_key().get("color", ["C0"]))
        ref = metric_ref(metric)
        spread_drawn = False
        most_inits = 0
        width = snap_width(12)
        sizes = font_sizes(width)
        figure, axes = plt.subplots(
            len(units), 1, figsize=(width, max(3.4 * len(units), 3.6)),
            sharex=True, squeeze=False,
        )
        axes = [row[0] for row in axes]
        for axis, unit in zip(axes, units):
            for index, variable in enumerate(sorted(grouped[unit])):
                stats = lead_aggregate(subset[subset["variable"] == variable])
                color = cycle[index % len(cycle)]
                most_inits = max(most_inits, int(stats["n"].max()))
                if bool((stats["n"] > 1).any()):
                    axis.fill_between(
                        stats.index,
                        stats["low"],
                        stats["high"],
                        color=color,
                        alpha=0.18,
                        linewidth=0,
                    )
                    spread_drawn = True
                axis.plot(
                    stats.index,
                    stats["mean"],
                    color=color,
                    marker="o" if len(stats) <= 20 else None,
                    markersize=3.5,
                    linewidth=1.6,
                    label=variable,
                )
            axis.set_ylabel(_axis_label(metric, unit), fontsize=sizes["label"])
            axis.tick_params(labelsize=sizes["tick"])
            axis.grid(linewidth=0.6)
            axis.legend(loc="best", fontsize=sizes["legend"], ncol=2, framealpha=0.85)
            if ref is not None:
                axis.axhline(ref, color=NEUTRAL["reference"], linestyle="--", linewidth=1)
                axis.annotate(
                    f"参考 {ref:g}",
                    xy=(0.995, ref),
                    xycoords=("axes fraction", "data"),
                    ha="right",
                    va="bottom",
                    fontsize=sizes["annot"],
                    color=NEUTRAL["muted"],
                )
        axes[-1].set_xlabel("预报时效 (h)", fontsize=sizes["label"])
        figure.suptitle(
            title or f"{metric_label(metric)} 随预报时效变化",
            fontsize=sizes["suptitle"],
        )
        notes = []
        if spread_drawn:
            notes.append(
                f"实线为跨起报均值，阴影为跨起报 10–90 分位（每个时效最多 {most_inits} 个起报）。"
            )
        else:
            notes.append("每个时效只有 1 个起报，只有均值曲线，没有分位带。")
        if len(units) > 1:
            notes.append("按单位分面，各面纵轴不同，面与面之间不可直接比大小。")
        figure.text(
            0.01, 0.005, "注：" + "".join(notes),
            fontsize=sizes["annot"], color=NEUTRAL["muted"],
        )
        figure.tight_layout(rect=(0, 0.02, 1, 0.96))
        self._save(figure, save_path)
        return figure

    # ------------------------------------------------------------------ #
    def plot_metric_heatmap(
        self,
        df: pd.DataFrame,
        metric: str,
        *,
        variables: Optional[Sequence[str]] = None,
        title: Optional[str] = None,
        save_path: Optional[Path] = None,
    ):
        """变量 × 时效热力图，填色是**相对首时效的倍数**（单位无关，行间可比）。

        格子里是**跨起报均值**（与折线图同一口径），否则一个格子要装 184 个数。
        """
        subset = self._subset(df, metric)
        if variables is not None:
            subset = subset[subset["variable"].isin([str(name) for name in variables])]
            if subset.empty:
                raise ValueError(f"指定的 variables 在 metric={metric!r} 里一个都没有")

        pivot = subset.pivot_table(
            index="variable", columns="lead_h", values="value", aggfunc="mean", observed=True
        ).sort_index(axis=1)
        if pivot.shape[1] < 2:
            raise ValueError(f"metric={metric!r} 只有 1 个时效，画不了热力图")
        base = pivot.iloc[:, 0].replace(0.0, np.nan)
        ratios = pivot.divide(base, axis=0)
        order = ratios.iloc[:, -1].sort_values(ascending=False).index
        ratios = ratios.loc[order]
        base = base.loc[order]

        labels = []
        for variable in order:
            unit = _unit_of(subset[subset["variable"] == variable])
            suffix = f" {unit}" if unit and unit != "1" else ""
            labels.append(f"{variable}（首时效 {base[variable]:.4g}{suffix}）")

        leads = [float(value) for value in ratios.columns]
        width = snap_width(max(10, 0.22 * len(leads) + 4))
        sizes = font_sizes(width)
        figure, axis = plt.subplots(
            figsize=(width, 0.42 * len(order) + 3)
        )
        mesh = axis.pcolormesh(
            np.asarray(leads),
            np.arange(len(order)),
            ratios.values,
            cmap=CMAP_SEQ,
            shading="nearest",
        )
        axis.set_yticks(np.arange(len(order)))
        axis.set_yticklabels(labels, fontsize=sizes["tick"])
        axis.invert_yaxis()
        axis.set_xlabel("预报时效 (h)", fontsize=sizes["label"])
        axis.tick_params(axis="x", labelsize=sizes["tick"])
        axis.set_title(
            title or f"{metric_label(metric)} 相对首时效的倍数（行标签括号内为首时效绝对值）",
            fontsize=sizes["suptitle"],
        )
        bar = figure.colorbar(mesh, ax=axis, pad=0.02)
        bar.set_label(
            f"{metric_label(metric)} 相对首时效的倍数", fontsize=sizes["label"]
        )
        bar.ax.tick_params(labelsize=sizes["annot"])
        figure.tight_layout()
        self._save(figure, save_path)
        return figure

    # ------------------------------------------------------------------ #
    def plot_spectrum_curve(
        self,
        spectra: pd.DataFrame,
        variable: str,
        label: str,
        *,
        title: Optional[str] = None,
        save_path: Optional[Path] = None,
    ):
        """波数 vs 功率，预报/实况两条线 + 总功率比标注（log 纵轴）。

        Args:
            spectra: 长表（``variable / wavenumber / field / value``），
                由 :func:`load_init_spectra` 或 :func:`load_wavenumber_spectra` 读来。
            label: 图题里那个"这一条是什么"的标注（起报时刻或"全时段平均"）。
        """
        wide = spectrum_series(spectra, variable)
        wide = wide[wide.index > 0]  # k=0 的功率恒为 0，log 轴画不了
        if wide.empty:
            raise ValueError(f"{variable} @ {label} 的谱线只有 k=0，没有可画的波数")

        forecast_total = float(wide["power_forecast"].sum())
        observation_total = float(wide["power_observation"].sum())
        ratio = forecast_total / observation_total if observation_total else float("nan")

        width = snap_width(11)
        sizes = font_sizes(width)
        figure, axis = plt.subplots(figsize=(width, 6))
        axis.plot(wide.index, wide["power_forecast"], linewidth=1.6, label="预报")
        axis.plot(wide.index, wide["power_observation"], linewidth=1.6, label="实况")
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_xlabel("纬向波数 k（对数轴）", fontsize=sizes["label"])
        axis.set_ylabel("功率（对数轴）", fontsize=sizes["label"])
        axis.tick_params(labelsize=sizes["tick"])
        axis.grid(which="both", linewidth=0.6)
        axis.legend(loc="best", fontsize=sizes["legend"])
        axis.set_title(
            title
            or f"{variable} 纬向谱 @ {label}（总功率比 预报/实况 = {ratio:.3f}）",
            fontsize=sizes["suptitle"],
        )
        axis.annotate(
            f"总功率比 = {ratio:.3f}\n（全波数求和后相除，与长表里的 "
            f"spectrum_power_ratio 同口径）",
            xy=(0.99, 0.02),
            xycoords="axes fraction",
            ha="right",
            va="bottom",
            fontsize=sizes["annot"],
            color=NEUTRAL["text"],
            bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": NEUTRAL["grid"]},
        )
        figure.tight_layout()
        self._save(figure, save_path)
        return figure

    # ------------------------------------------------------------------ #
    def plot_spectrum_panel(
        self,
        spectra: pd.DataFrame,
        *,
        variables: Optional[Sequence[str]] = None,
        title: Optional[str] = None,
        save_path: Optional[Path] = None,
    ):
        """一张多面板图：一个变量一个子图，预报/实况两条谱线（log 纵轴）。

        输入是 :func:`load_wavenumber_spectra` 读出来的逐波数表（跨起报平均），
        每个子图右上角标该变量的**全波数总功率比**——它和长表里的
        ``spectrum_power_ratio`` 同口径，所以图上的线和表里的数字对得上。
        """
        names = sorted(
            str(name) for name in (variables if variables is not None else spectra["variable"].unique())
        )
        if not names:
            raise ValueError("谱表里没有任何变量")

        columns = min(3, len(names))
        rows = -(-len(names) // columns)
        width = snap_width(5.0 * columns)
        sizes = font_sizes(width)
        figure, axes = plt.subplots(
            rows, columns, figsize=(width, 3.6 * rows), squeeze=False
        )
        flat = [axis for row in axes for axis in row]
        for axis, variable in zip(flat, names):
            wide = spectrum_series(spectra, variable)
            wide = wide[wide.index > 0]  # k=0 的功率恒为 0，log 轴画不了
            forecast_total = float(wide["power_forecast"].sum())
            observation_total = float(wide["power_observation"].sum())
            ratio = forecast_total / observation_total if observation_total else float("nan")
            axis.plot(wide.index, wide["power_forecast"], linewidth=1.4, label="预报")
            axis.plot(wide.index, wide["power_observation"], linewidth=1.4, label="实况")
            axis.set_yscale("log")
            axis.set_title(
                f"{variable}（总功率比 {ratio:.3f}）", fontsize=sizes["title"]
            )
            axis.tick_params(labelsize=sizes["tick"])
            axis.grid(which="both", linewidth=0.6)
        for axis in flat[len(names):]:
            axis.axis("off")
        for axis in flat[: len(names)][-columns:]:
            axis.set_xlabel("纬向波数 k", fontsize=sizes["label"])
        for index, axis in enumerate(flat[: len(names)]):
            if index % columns == 0:
                axis.set_ylabel("功率（对数轴）", fontsize=sizes["label"])
        flat[0].legend(loc="lower left", fontsize=sizes["legend"])
        figure.suptitle(
            title or "纬向功率谱（跨起报平均）", fontsize=sizes["suptitle"]
        )
        figure.text(
            0.01,
            0.005,
            "注：口径同 `mean_spectrum_<var>.csv`——跨起报、跨时效平均后的功率谱，"
            "看不出谱随预报时效怎么退化；要指定时效的谱请读 scores_detail.csv。"
            "标题里的总功率比是**全波数求和后**再相除，它正常不代表分布正常。",
            fontsize=sizes["annot"],
            color=NEUTRAL["muted"],
        )
        figure.tight_layout(rect=(0, 0.03, 1, 0.96))
        self._save(figure, save_path)
        return figure

    # ------------------------------------------------------------------ #
    def create_report(
        self,
        scores_df: pd.DataFrame,
        output_dir: Path,
        *,
        spectrum_dir: Optional[Path] = None,
        spectrum_variable: Optional[str] = None,
        spectrum_init: Optional[str] = None,
        model_name: str = "model",
        sources: Optional[Dict[str, str]] = None,
        manifest_info: Optional[Dict[str, object]] = None,
        change_description: Optional[str] = None,
    ) -> Dict[str, Path]:
        """出图 + 写 Markdown 报告，返回 ``{名称: 路径}``。

        Args:
            spectrum_dir: 产物目录（含 ``diagnostics/``）。给了就出一张全变量的
                纬向谱面板，并让报告多一节分波段功率比；目录里没有谱文件就照常
                出别的图，不报错。
            spectrum_variable / spectrum_init: 点名看**某一个起报**的谱曲线时才给；
                ``spectrum_init`` 省略时取最后一个起报。
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        data = prepare_field_dataframe(scores_df)
        present = set(_usable(data)["metric"])
        print(f"\n生成 {model_name} 连续场评测报告（输出 {output_dir}）")
        artifacts: Dict[str, Path] = {}

        for metric in SKILL_METRICS:
            if metric not in present:
                print(f"  跳过 {metric}：长表里没有这个指标")
                continue
            path = output_dir / f"{metric}_vs_lead.png"
            self.plot_metric_vs_lead(
                data,
                metric,
                title=f"{model_name} {metric_label(metric)} 随预报时效变化",
                save_path=path,
            )
            plt.close()
            artifacts[f"{metric}_vs_lead"] = path

        if "rmse" in present:
            path = output_dir / "rmse_heatmap.png"
            self.plot_metric_heatmap(
                data,
                "rmse",
                title=f"{model_name} RMSE 相对首时效的倍数（行标签括号内为首时效绝对值）",
                save_path=path,
            )
            plt.close()
            artifacts["rmse_heatmap"] = path

        spectra = None
        if spectrum_dir:
            try:
                spectra = load_wavenumber_spectra(Path(spectrum_dir))
            except ValueError as error:
                print(f"  跳过纬向谱面板：{error}")
        if spectra is not None and not spectra.empty:
            path = output_dir / "spectrum_panel.png"
            self.plot_spectrum_panel(
                spectra,
                title=f"{model_name} 纬向功率谱（全时段加权平均）",
                save_path=path,
            )
            plt.close()
            artifacts["spectrum_panel"] = path

        spectrum_note = None
        if spectrum_variable and spectrum_dir:
            try:
                one = load_init_spectra(Path(spectrum_dir), spectrum_variable, spectrum_init)
            except ValueError as error:
                spectrum_note = f"要求了谱曲线但没出图：{error}"
            else:
                stamp = pd.to_datetime(one.attrs.get("init_time") or "", errors="coerce")
                ok = pd.notna(stamp)
                shown = f"{stamp:%Y-%m-%d %H}" if ok else str(spectrum_init or "")
                token = f"{stamp:%Y%m%d%H}" if ok else "unknown"
                path = output_dir / f"spectrum_curve_{spectrum_variable}_{token}.png"
                self.plot_spectrum_curve(
                    one,
                    spectrum_variable,
                    shown or "最后一个起报",
                    title=f"{model_name} {spectrum_variable} 纬向谱 @ {shown} 起报",
                    save_path=path,
                )
                plt.close()
                artifacts[path.stem] = path
                spectrum_note = (
                    f"谱曲线：`{path.name}`（{spectrum_variable}，{shown} 起报）—— 双对数坐标，"
                    f"两条线分别是预报与实况的功率谱，总功率比见左上角标注。"
                    f"注意这是**该起报跨时效加权平均**后的谱，不是某个时效的谱。"
                )
        elif spectrum_variable and not spectrum_dir:
            spectrum_note = (
                "要求了谱曲线但没给产物目录，未出图"
                "（逐波数谱在 `diagnostics/spectrum_by_init.csv` 里）。"
            )

        report_path = output_dir / "REPORT.md"
        build_field_report(
            # 给**原始**帧而不是滤过区域的 data：报告要如实写"长表一共多少行、
            # 其中多少行是区域行没进统计"。两边的全球行口径一致，数字对得上。
            scores_df,
            report_path,
            model_name=model_name,
            requested=requested_from(manifest_info or {}),
            manifest_info=manifest_info,
            sources=sources,
            change_description=change_description,
            artifacts=artifacts,
            spectrum_note=spectrum_note,
            spectra=spectra,
        )
        artifacts["report"] = report_path
        # 别在这两行用 ✓ / ⚠：Windows 控制台默认 GBK，编码不了这两个字符。报告和图
        # 这时**已经写完了**，却会在最后一行抛 UnicodeEncodeError、进程非 0 退出——
        # 看着像失败，其实什么都没坏。中文字符本身 GBK 是有的，只有这些符号会炸。
        print(f"报告完成：{report_path}（{len(artifacts) - 1} 张图 + 1 份报告）")
        if self.font is None:
            print("未找到中文字体，图中的中文可能显示为方块；请安装 Noto Sans CJK 或 SimHei")
        return artifacts


