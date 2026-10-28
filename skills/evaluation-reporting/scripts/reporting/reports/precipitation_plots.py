#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""降水分类检验（TS 系列）的可视化。

输入：一张 TS 结果表，列至少包含 ``lead_h / grade / TS``，通常还有
``window_h / threshold_mm / hits / misses / false_alarms / n_pairs / POD / FAR / BIAS``
（``漏报率`` 缺失时由 ``1 - POD`` 推导；多出来的列如 ``run_id`` 会被忽略）。

输出：若干 SVG，以及一份 Markdown 报告（报告正文见 ``ts_report`` 模块）。

三条与旧实现的区别（旧实现在服务器上会出豆腐块/空图/被截断的柱子）：

* 中文字体**自动探测**，找不到中文字体时退化为英文标签并给出警告；
* 时效**不写死**：默认取数据里最小的时效，也可显式指定；
* 每个指标用自己的纵轴范围，``BIAS`` 不截断并画 ``=1`` 参考线。
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from matplotlib.ticker import MultipleLocator

from reporting.core.style import (
    CMAP_SEQ,
    LEGEND_FRAME,
    NEUTRAL,
    apply_style,
    font_sizes,
    grade_colors,
    model_colors,
    snap_width,
)
from reporting.reports.ts_report import (
    box_stats,
    build_ts_report,
    collect_box_models,
    load_probability,
    ordered_grades,
)

try:  # seaborn 只用于热力图配色，缺了也能跑
    import seaborn as sns
except ImportError:  # pragma: no cover
    sns = None

REQUIRED_COLUMNS = ("lead_h", "grade", "TS")
OPTIONAL_COLUMNS = (
    "window_h",
    "threshold_mm",
    "hits",
    "misses",
    "false_alarms",
    "n_pairs",
    "POD",
    "FAR",
    "BIAS",
)


def prepare_ts_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """校验并规范化 TS 结果表。

    Raises:
        ValueError: 缺少必需列，或数据为空。
    """
    data = df.copy()
    data.columns = [str(column).strip() for column in data.columns]

    missing = [column for column in REQUIRED_COLUMNS if column not in data.columns]
    if missing:
        raise ValueError(
            f"TS 结果表缺少必需列 {missing}；现有列: {list(data.columns)}"
        )
    if data.empty:
        raise ValueError("TS 结果表是空的")

    if "threshold_mm" not in data.columns:
        data["threshold_mm"] = data["grade"].astype(str).str.replace("≥", "", regex=False)
        data["threshold_mm"] = pd.to_numeric(data["threshold_mm"], errors="coerce")
    if "漏报率" not in data.columns:
        if "miss_rate" in data.columns:
            data["漏报率"] = pd.to_numeric(data["miss_rate"], errors="coerce")
        elif "POD" in data.columns:
            data["漏报率"] = 1.0 - pd.to_numeric(data["POD"], errors="coerce")

    for column in OPTIONAL_COLUMNS + ("漏报率",):
        if column in data.columns:
            data[column] = pd.to_numeric(data[column], errors="coerce")

    data["lead_h"] = pd.to_numeric(data["lead_h"], errors="coerce")
    data = data.dropna(subset=["lead_h", "TS"])
    if data.empty:
        raise ValueError("TS 结果表里没有可用的 lead_h/TS 数值")

    data["grade"] = data["grade"].astype(str)
    if "threshold_mm" in data.columns and data["threshold_mm"].notna().any():
        order = data.groupby("grade")["threshold_mm"].first().sort_values().index
        data["grade"] = pd.Categorical(data["grade"], categories=list(order), ordered=True)
    return data.sort_values(["grade", "lead_h"])


def split_regions(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """宽表 → ``(全球行, {带名: 该带的行})``。

    分纬度带评测时同一张宽表里全球行与各带行并存（全球行的 ``region`` 空），
    **两者绝不能混着算**：各带平均再平均 ≠ 全球平均，混进去会把带间差异整个
    抹平。所以图与正文一律只看全球行，带行走 :func:`ts_report.build_ts_report`
    的「附 L」那一块。

    没有 ``region`` 列（这个能力出现之前的产物、或没配分带）时：全部算全球行、
    带集为空——**与加这个能力之前逐行一致**。函数本身是幂等的，传进来的已经是
    全球行时得到的还是它自己。
    """
    if "region" not in df.columns:
        return df, {}
    values = df["region"]
    empty = values.isna() | (values.astype(str).str.strip() == "")
    if bool(empty.all()):
        return df, {}
    bands = {
        name: df[~empty & (values.astype(str).str.strip() == name)]
        for name in sorted(set(values[~empty].astype(str).str.strip()))
    }
    return df[empty], bands


def available_leads(df: pd.DataFrame) -> List[float]:
    """数据里出现过的时效（升序）。"""
    return sorted(float(value) for value in pd.unique(df["lead_h"]))


def lead_ticks(leads: Sequence[float], max_ticks: int = 5) -> List[float]:
    """从有效时效里挑一组 x 轴刻度：最多 ``max_ticks`` 个、首末必含。

    只标首/中/末三个时，24–360h 之间读不出衰减发生在哪一段。按等距索引抽稀
    （``linspace(0, n-1, max_ticks)`` 再取整），首末两档的间隔不会比其他档窄
    ——按固定步长抽会在末段剩一个 48h 的小间隔，两个三位数标签就叠在一起。
    刻度都落在真实采样点时效上：15 个时效 → 24/120/192/264/360。

    面板宽度不同，能放下的标签数也不同：12 in 宽、3 列的面板用 5 个，
    4 列的概率评分面板只用 4 个。
    """
    values = sorted({float(value) for value in leads})
    if len(values) <= max_ticks:
        return values
    indices = np.round(np.linspace(0, len(values) - 1, max_ticks)).astype(int)
    return [values[index] for index in dict.fromkeys(indices)]


class PrecipitationPlotter:
    """降水分类检验结果的绘图器。"""

    def __init__(self, style: Optional[str] = None):
        """
        Args:
            style: 可选 matplotlib 样式名；默认不套样式，
                避免样式把中文字体设置覆盖掉（旧实现就栽在这）。
        """
        self.font = apply_style()
        if style:
            plt.style.use(style)
            self.font = apply_style()
        plt.rcParams["figure.dpi"] = 150
        plt.rcParams["savefig.dpi"] = 150
        plt.rcParams["savefig.bbox"] = "tight"

    def _save(self, fig, save_path: Optional[Path]) -> None:
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path)
            print(f"已保存: {save_path}")

    def plot_metric_vs_lead(
        self,
        df: pd.DataFrame,
        metric: str = "TS",
        figsize: Tuple[int, int] = (12, 6),
        title: Optional[str] = None,
        save_path: Optional[Path] = None,
    ):
        """指标随时效变化（按阈值分线）。"""
        if metric not in df.columns:
            raise ValueError(f"结果表里没有指标列 '{metric}'")

        data = df.dropna(subset=[metric])
        colors = grade_colors(
            list(data["grade"].cat.categories)
            if hasattr(data["grade"], "cat")
            else sorted(data["grade"].unique())
        )
        # 事件数过少的等级用虚线并标注，避免"看着像结论"（样本少时方差极大）
        events = {}
        if {"hits", "misses"}.issubset(data.columns):
            totals = data.fillna({"hits": 0, "misses": 0}).groupby("grade", observed=True)[
                ["hits", "misses"]
            ].sum()
            events = {str(index): float(row["hits"] + row["misses"]) for index, row in totals.iterrows()}

        width = snap_width(figsize[0])
        sizes = font_sizes(width)
        fig, ax = plt.subplots(figsize=(width, float(figsize[1])))
        for grade, subset in data.groupby("grade", observed=True):
            subset = subset.sort_values("lead_h")
            sparse = events.get(str(grade), float("inf")) < 50
            ax.plot(
                subset["lead_h"],
                subset[metric],
                marker="o",
                markersize=5,
                linewidth=2,
                linestyle="--" if sparse else "-",
                label=f"{grade}（样本少，仅参考）" if sparse else str(grade),
                color=colors.get(str(grade), NEUTRAL["muted"]),
            )

        ax.set_xlabel("预报时效 (小时)", fontsize=sizes["label"])
        ax.set_ylabel(metric, fontsize=sizes["label"])
        ax.tick_params(labelsize=sizes["tick"])
        ax.set_title(
            title or f"{metric} 随预报时效变化", fontsize=sizes["suptitle"]
        )
        ax.grid(True, linestyle="--")
        ax.set_xlim(left=0)
        if metric in ("TS", "POD", "FAR", "漏报率"):
            ax.set_ylim(0, 1)
        if metric == "BIAS":
            ax.axhline(1.0, color=NEUTRAL["reference"], linestyle="--", linewidth=1, alpha=0.8)
            ax.text(
                0.01, 1.0, " BIAS=1 (无偏差)",
                transform=ax.get_yaxis_transform(),
                va="bottom", ha="left", fontsize=sizes["annot"], alpha=0.7,
            )
        ax.legend(
            title="降水等级", loc="best", frameon=True,
            fontsize=sizes["legend"],
        )
        fig.tight_layout()
        self._save(fig, save_path)
        return fig

    def plot_metric_by_threshold(
        self,
        df: pd.DataFrame,
        lead_h: Optional[float] = None,
        metrics: Sequence[str] = ("TS", "POD", "FAR"),
        figsize: Tuple[int, int] = (12, 6),
        save_path: Optional[Path] = None,
    ):
        """指定时效下、各阈值的指标柱状对比（每个指标一个子图）。"""
        leads = available_leads(df)
        lead = float(lead_h) if lead_h is not None else leads[0]
        subset = df[df["lead_h"] == lead].sort_values("threshold_mm")
        if subset.empty:
            raise ValueError(f"结果表里没有 {lead}h 时效的数据（可用: {leads}）")

        metrics = [metric for metric in metrics if metric in subset.columns]
        if not metrics:
            raise ValueError("没有可绘制的指标列")

        width = snap_width(figsize[0])
        sizes = font_sizes(width)
        fig, axes = plt.subplots(1, len(metrics), figsize=(width, float(figsize[1])))
        axes = np.atleast_1d(axes)
        x = np.arange(len(subset))
        bar_colors = grade_colors([str(grade) for grade in subset["grade"]])
        for ax, metric in zip(axes, metrics):
            ax.bar(
                x, subset[metric], alpha=0.85,
                color=[bar_colors[str(grade)] for grade in subset["grade"]],
            )
            ax.set_title(f"{metric} @ {lead:g}h", fontsize=sizes["title"])
            ax.set_xticks(x)
            ax.set_xticklabels(
                [str(value) for value in subset["grade"]],
                fontsize=sizes["tick"],
            )
            ax.set_xlabel("降水等级", fontsize=sizes["label"])
            ax.tick_params(axis="y", labelsize=sizes["tick"])
            ax.grid(True, axis="y", linestyle="--")
            if metric in ("TS", "POD", "FAR", "漏报率"):
                ax.set_ylim(0, 1)
            elif metric == "BIAS":
                ax.axhline(1.0, color=NEUTRAL["reference"], linestyle="--", linewidth=1, alpha=0.8)
            for index, value in enumerate(subset[metric]):
                if pd.notna(value):
                    ax.text(
                        index, value, f"{value:.2f}", ha="center", va="bottom",
                        fontsize=sizes["annot"],
                    )
        fig.suptitle(
            f"预报时效 {lead:g}h 各降水等级的指标对比",
            fontsize=sizes["suptitle"],
        )
        fig.tight_layout()
        self._save(fig, save_path)
        return fig

    def plot_heatmap(
        self,
        df: pd.DataFrame,
        metric: str = "TS",
        figsize: Tuple[int, int] = (12, 7),
        cmap=None,
        save_path: Optional[Path] = None,
    ):
        """阈值 × 时效 热力图；``cmap`` 不传就用统一的单向色带。"""
        cmap = CMAP_SEQ if cmap is None else cmap
        pivot = df.pivot_table(index="grade", columns="lead_h", values=metric, observed=True)
        width = snap_width(figsize[0])
        sizes = font_sizes(width)
        fig, ax = plt.subplots(figsize=(width, float(figsize[1])))
        upper = 1.0 if metric != "BIAS" else max(2.0, float(np.nanmax(pivot.values)))
        if sns is not None:
            sns.heatmap(
                pivot, annot=True, fmt=".3f", cmap=cmap, linewidths=0.5,
                linecolor=NEUTRAL["grid"], ax=ax, vmin=0, vmax=upper,
                annot_kws={"fontsize": sizes["annot"]},
                cbar_kws={"label": metric},
            )
            bar = ax.collections[0].colorbar
            if bar is not None:
                bar.set_label(metric, fontsize=sizes["label"])
                bar.ax.tick_params(labelsize=sizes["annot"])
            # 数字颜色随底色深浅自动切换（浅底黑字、深底白字）
            values = np.asarray(pivot.values, dtype="f8")
            for text in ax.texts:
                x, y = text.get_position()
                row, column = int(round(float(y) - 0.5)), int(round(float(x) - 0.5))
                if 0 <= row < values.shape[0] and 0 <= column < values.shape[1]:
                    value = values[row, column]
                    if np.isfinite(value):
                        text.set_color("white" if value / upper > 0.6 else NEUTRAL["observation"])
        else:  # pragma: no cover
            mesh = ax.pcolormesh(pivot.values, cmap=cmap, vmin=0, vmax=upper)
            bar = fig.colorbar(mesh, ax=ax)
            bar.set_label(metric, fontsize=sizes["label"])
            bar.ax.tick_params(labelsize=sizes["annot"])
            ax.set_xticks(np.arange(len(pivot.columns)) + 0.5)
            ax.set_xticklabels([f"{value:g}" for value in pivot.columns])
            ax.set_yticks(np.arange(len(pivot.index)) + 0.5)
            ax.set_yticklabels([str(value) for value in pivot.index])
        ax.tick_params(labelsize=sizes["tick"])
        ax.set_xlabel("预报时效 (小时)", fontsize=sizes["label"])
        ax.set_ylabel("降水等级", fontsize=sizes["label"])
        ax.set_title(
            f"{metric} 热力图（时效 × 降水等级）", fontsize=sizes["suptitle"]
        )
        fig.tight_layout()
        self._save(fig, save_path)
        return fig

    def plot_performance_diagram(
        self,
        df: pd.DataFrame,
        lead_h: Optional[float] = None,
        figsize: Tuple[int, int] = (12, 9),
        save_path: Optional[Path] = None,
    ):
        """性能图：横轴 1-FAR，纵轴 POD，灰色为 TS 等值线。"""
        leads = available_leads(df)
        lead = float(lead_h) if lead_h is not None else leads[0]
        subset = df[(df["lead_h"] == lead) & df["POD"].notna() & df["FAR"].notna()]
        if subset.empty:
            raise ValueError(f"结果表里没有 {lead}h 的 POD/FAR 数据")

        width = snap_width(figsize[0])
        # aspect="equal" 让坐标区是个正方形，保存时两侧留白被 tight bbox 裁掉，
        # 成图实际宽度≈figsize[1]。字号按这个宽度算，插进报告（统一按 12 in
        # 显示）才和别的图一样大，不会凭空放大 27%。
        sizes = font_sizes(float(figsize[1]))
        fig, ax = plt.subplots(figsize=(width, float(figsize[1])))
        grid = np.linspace(0.001, 1, 200)
        sr, pod = np.meshgrid(grid, grid)
        with np.errstate(divide="ignore", invalid="ignore"):
            ts = 1.0 / (1.0 / pod + 1.0 / sr - 1.0)
        contour = ax.contour(sr, pod, ts, levels=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                             colors=NEUTRAL["reference"], linewidths=1,
                             linestyles="--", alpha=0.8)
        ax.clabel(contour, inline=True, fontsize=sizes["annot"], fmt="TS=%.1f")
        for bias in (0.5, 1.0, 1.5, 2.0):
            ax.plot(grid, grid * bias, color=NEUTRAL["reference"], linestyle=":",
                    linewidth=1, alpha=0.8)
            # 标签放在这条线离开坐标框的那个边上：>1 的线从顶边出去，
            # 统一写在 (1,1) 会让 1.5 和 2.0 两个标签叠成一团。
            if bias >= 1.0:
                ax.text(
                    1.0 / bias, 1.0, f"BIAS={bias:g}", ha="center", va="top",
                    fontsize=sizes["annot"], color=NEUTRAL["muted"],
                )
            else:
                ax.text(
                    1.0, bias, f"BIAS={bias:g}", ha="right", va="bottom",
                    fontsize=sizes["annot"], color=NEUTRAL["muted"],
                )

        colors = grade_colors([str(value) for value in subset["grade"].unique()])
        for _, row in subset.iterrows():
            ax.scatter(1 - row["FAR"], row["POD"], s=130, zorder=10,
                       color=colors.get(str(row["grade"]), NEUTRAL["muted"]),
                       edgecolors=NEUTRAL["observation"], linewidths=1.2,
                       label=str(row["grade"]))
        handles, labels = ax.get_legend_handles_labels()
        seen = dict(zip(labels, handles))
        ax.legend(
            seen.values(), seen.keys(), title="降水等级", loc="lower left",
            frameon=True, fontsize=sizes["legend"],
        )
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        ax.tick_params(labelsize=sizes["tick"])
        ax.set_xlabel("成功比 1-FAR", fontsize=sizes["label"])
        ax.set_ylabel("命中率 POD", fontsize=sizes["label"])
        ax.set_title(
            f"性能图（预报时效 {lead:g}h）", fontsize=sizes["suptitle"]
        )
        ax.grid(True, linestyle="--")
        fig.tight_layout()
        self._save(fig, save_path)
        return fig

    def plot_metric_boxplot(
        self,
        model_dfs: Dict[str, pd.DataFrame],
        metric: str = "TS",
        figsize: Tuple[int, int] = (12, 6),
        save_path: Optional[Path] = None,
    ):
        """各模型、各降水等级的**跨时效分布**（箱线图）。

        一个箱体 = 同一模型、同一量级在**全部时效**上的取值组成的样本：
        箱体是四分位距（IQR），箱中横线是中位数，须线延伸到 1.5×IQR 内最远的点，
        空心圆是离群时效。横坐标是降水等级，每个量级下并排各模型一个箱体。

        比率型指标（0–1）用**断轴**：下段 0–0.6（数据都落在这里，刻度每 0.1），
        上段 0.9–1.0 只标出 1 的天花板，中间 0.6–0.9 的空白直接跳过；
        否则 0.5 以上的半张图全是空的，箱体挤在底部看不出四分位差。
        """
        prepared = {name: prepare_ts_dataframe(frame) for name, frame in model_dfs.items()}
        names = list(prepared)
        grades = ordered_grades(list(prepared.values()))
        if not names or not grades:
            raise ValueError("箱线图没有可画的数据（模型列表或降水等级为空）")

        box_width = 0.8 / len(names)
        colors = model_colors(names)
        width = snap_width(figsize[0])
        sizes = font_sizes(width)
        # 比率型指标断轴；BIAS 不截断（可正可负、量纲无上界），仍用单轴。
        ratio_metric = metric in ("TS", "POD", "FAR", "漏报率")
        if ratio_metric:
            # 第 0 行是上段（0.9–1.0 的天花板），第 1 行才是真正的数据段 0–0.6。
            fig, (ax_top, ax) = plt.subplots(
                2, 1, figsize=(width, float(figsize[1])), sharex=True,
                gridspec_kw={"height_ratios": (1, 6)},
            )
            targets = (ax, ax_top)
        else:
            fig, ax = plt.subplots(figsize=(width, float(figsize[1])))
            ax_top = None
            targets = (ax,)

        for grade_index, grade in enumerate(grades):
            for model_index, name in enumerate(names):
                frame = prepared[name]
                subset = frame[frame["grade"].astype(str) == grade]
                stats = box_stats(subset, metric=metric) if len(subset) else {"n": 0}
                position = grade_index + (model_index - len(names) / 2 + 0.5) * box_width
                color = colors[name]
                if stats["n"] == 0:
                    # 该模型没有这个量级（或全无有效值）。**不能把空数组喂给 boxplot**：
                    # 它随 matplotlib 版本时好时坏，还会吐 RuntimeWarning。画个字说明空缺。
                    ax.text(
                        position, 0.02, "n/a", rotation=90,
                        ha="center", va="bottom", fontsize=sizes["annot"],
                        color=NEUTRAL["muted"],
                    )
                    continue
                # 一次调用只画一个箱体：这样每个箱子能单独上色，缺值也能按格跳过。
                # 断轴时上下两段各画一遍，跨断点的箱体会被各自裁切，不会造假。
                for target in targets:
                    target.boxplot(
                        subset[metric].dropna().to_numpy(dtype=float),
                        positions=[position],
                        widths=box_width * 0.85,
                        patch_artist=True,
                        showfliers=True,
                        whis=1.5,
                        manage_ticks=False,  # 否则每次调用都重设刻度定位器，最后一次说了算
                        boxprops=dict(facecolor=color, edgecolor=NEUTRAL["text"],
                                      linewidth=1.0, alpha=0.85),
                        medianprops=dict(color=NEUTRAL["observation"], linewidth=1.6),
                        whiskerprops=dict(color=NEUTRAL["text"], linewidth=1.2),
                        capprops=dict(color=NEUTRAL["text"], linewidth=1.2),
                        flierprops=dict(
                            marker="o", markerfacecolor="none", markeredgecolor=color,
                            markersize=5, markeredgewidth=1.0, linestyle="none",
                        ),
                    )

        ax.set_xticks(range(len(grades)))
        ax.set_xticklabels(grades, fontsize=sizes["tick"])
        ax.set_xlim(-0.5, len(grades) - 0.5)
        ax.tick_params(axis="y", labelsize=sizes["tick"])
        ax.set_xlabel("Rain grade", fontsize=sizes["label"])
        ax.set_ylabel(metric, fontsize=sizes["label"])
        if ratio_metric:
            # 下段 0–0.6（刻度每 0.1），上段只留 0.9–1.0 露出 1 这个天花板；
            # 0.6–0.9 没有数据，直接跳过，避免把箱体压在半张空图底部。
            ax.set_ylim(0, 0.6)
            ax.yaxis.set_major_locator(MultipleLocator(0.1))
            ax_top.set_ylim(0.9, 1.0)
            ax_top.set_yticks([1.0])
            ax_top.tick_params(axis="y", labelsize=sizes["tick"])
            # 上段的 x 刻度线和标签都去掉：它只是天花板，不承载数据。
            ax_top.tick_params(axis="x", bottom=False, labelbottom=False)
            ax_top.set_title(f"{metric} across leads", fontsize=sizes["suptitle"])
            # 两条内边界不画，靠断轴斜线表示跳变
            ax.spines["top"].set_visible(False)
            ax_top.spines["bottom"].set_visible(False)
            mark = 0.012
            for target, edge in ((ax, 1.0), (ax_top, 0.0)):
                props = dict(
                    transform=target.transAxes, color=NEUTRAL["text"],
                    clip_on=False, linewidth=1.2,
                )
                target.plot(
                    (-mark, mark), (edge - mark, edge + mark), **props
                )
                target.plot(
                    (1 - mark, 1 + mark), (edge - mark, edge + mark), **props
                )
        else:
            ax.set_title(f"{metric} across leads", fontsize=sizes["suptitle"])
            if metric == "BIAS":
                ax.axhline(1.0, color=NEUTRAL["reference"], linestyle="--",
                           linewidth=1, alpha=0.8)
        if len(names) > 1:
            # 手工拼图例：直接 ax.legend() 会给每次 boxplot 调用各留一个条目
            ax.legend(
                handles=[
                    Patch(facecolor=colors[name],
                          edgecolor=NEUTRAL["text"], label=name)
                    for name in names
                ],
                loc="upper right",
                fontsize=sizes["legend"], **LEGEND_FRAME,
            )
        fig.tight_layout()
        if ratio_metric:
            # tight_layout 会按子图内容重算间距，断轴缝隙最后再钉一次。
            fig.subplots_adjust(hspace=0.06)
        self._save(fig, save_path)
        return fig

    # ------------------------------------------------------------------ #
    # 多模型对比的几张图（报告 4.1–4.5 与 §5 用）
    # ------------------------------------------------------------------ #
    @staticmethod
    def _at_lead(frame: pd.DataFrame, lead: float) -> pd.DataFrame:
        """某个时效的行；表里有 ``window_h`` 时只留同一个累积窗口。"""
        subset = frame[pd.to_numeric(frame["lead_h"], errors="coerce") == lead]
        if "window_h" in subset.columns and subset["window_h"].notna().any():
            window = subset["window_h"].dropna().iloc[0]
            subset = subset[subset["window_h"] == window]
        return subset

    def plot_metrics_at_lead(
        self,
        model_dfs: Dict[str, pd.DataFrame],
        lead_h: float,
        metrics: Sequence[str] = ("TS", "POD", "FAR", "BIAS"),
        figsize: Tuple[float, float] = (12, 7.2),
        save_path: Optional[Path] = None,
    ):
        """固定时效上的分等级指标对比：一个指标一个子图、每格一组柱。

        柱按模型着色（与箱线图、差值图同一份配色），横轴是降水等级。
        ``TS`` / ``POD`` / ``FAR`` 锁 0–1，``BIAS`` 自适应并画 ``=1`` 虚线——
        BIAS 永远不截断，截断就是在藏高报/漏报。

        宽度取 12 in（与其他多模型图同一档），字号随之落在 17/20 pt；
        子图标题左对齐并带 ``(a)(b)…`` 面板号，每个子图各自标 y 轴。
        """
        prepared = {name: prepare_ts_dataframe(frame) for name, frame in model_dfs.items()}
        names = list(prepared)
        grades = ordered_grades(list(prepared.values()))
        if not names or not grades:
            raise ValueError("分等级柱状图没有可画的数据（模型列表或降水等级为空）")
        available = [
            metric
            for metric in metrics
            if any(metric in frame.columns for frame in prepared.values())
        ]
        if not available:
            raise ValueError(f"结果表里没有 {list(metrics)} 中的任何指标")

        columns = 2
        rows = int(np.ceil(len(available) / columns))
        width = snap_width(figsize[0])
        sizes = font_sizes(width)
        fig, axes = plt.subplots(
            rows, columns, figsize=(width, float(figsize[1])), squeeze=False
        )
        axes = axes.ravel()
        colors = model_colors(names)
        bar_width = 0.8 / len(names)
        positions = np.arange(len(grades), dtype=float)

        for index, metric in enumerate(available):
            ax = axes[index]
            for model_index, name in enumerate(names):
                frame = prepared[name]
                if metric not in frame.columns:
                    continue
                subset = self._at_lead(frame, float(lead_h))
                series = subset.groupby("grade", observed=True)[metric].mean()
                values = [
                    float(series[grade]) if grade in series.index else np.nan
                    for grade in grades
                ]
                ax.bar(
                    positions + (model_index - (len(names) - 1) / 2) * bar_width,
                    values,
                    width=bar_width * 0.9,
                    color=colors[name],
                    edgecolor="white",
                    linewidth=0.6,
                    label=name,
                )
            ax.set_xticks(positions)
            ax.set_xticklabels(grades, fontsize=sizes["tick"])
            ax.tick_params(axis="y", labelsize=sizes["tick"])
            # 子图标题比轴标签低一档：总标题已经写了时效，面板行只是索引；
            # 与轴标签同号（都是 20 pt）时四行标题会抢走视线。
            ax.set_title(
                f"({chr(ord('a') + index)}) {metric} at {lead_h:g}h",
                loc="left", fontsize=round(sizes["title"] * 0.85),
            )
            ax.set_xlabel("Rain grade", fontsize=sizes["label"])
            ax.set_ylabel(metric, fontsize=sizes["label"])
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            if metric in ("TS", "POD", "FAR", "漏报率"):
                ax.set_ylim(0, 1)
            elif metric == "BIAS":
                ax.axhline(
                    1.0, color=NEUTRAL["reference"], linestyle="--",
                    linewidth=1.0, alpha=0.8,
                )
        for ax in axes[len(available):]:
            ax.axis("off")
        handles, labels = axes[0].get_legend_handles_labels()
        if len(names) > 1 and handles:
            fig.legend(
                handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.005),
                ncol=min(len(labels), 5),
                fontsize=sizes["legend"], **LEGEND_FRAME,
            )
        fig.suptitle(
            f"Metrics by rain grade at {lead_h:g}h lead",
            fontsize=sizes["suptitle"],
        )
        # 底行子图自己带 x 轴名，图例再占一条：底部留 17%，否则图例会压到
        # "Rain grade" 上（8% 只够图例、不够最后一行 x 标签）。
        bottom = 0.17 if (len(names) > 1 and handles) else 0.0
        fig.tight_layout(rect=(0, bottom, 1, 0.95))
        self._save(fig, save_path)
        return fig

    def plot_metric_vs_lead(
        self,
        model_dfs: Dict[str, pd.DataFrame],
        metric: str = "TS",
        figsize: Tuple[float, float] = (12, 7.2),
        save_path: Optional[Path] = None,
        zoom_y: bool = False,
    ):
        """**一个指标**随时效变化：每个降水等级一个子图，每个模型一条**实线**。

        报告里 TS / POD / FAR / BIAS 各出一张（不再把三个误差指标挤进一张矩阵图）。
        图内文字一律英文（交付件要能直接进英文文档）。子图标题左对齐、带
        ``(a)``–``(f)`` 面板号；每个子图都标 x 轴（最多 6 个等距采样时效，
        见 :func:`lead_ticks`）和 ``Lead time (h)``，y 轴各自标指标名，不靠总标题。``BIAS`` 画 ``=1``
        虚线，``TS`` / ``POD`` / ``FAR`` 锁 0–1（``zoom_y=True`` 时 TS 按各子图
        数据范围自适应，便于分辨模型间差异）。
        """
        prepared = {name: prepare_ts_dataframe(frame) for name, frame in model_dfs.items()}
        names = list(prepared)
        if not names:
            raise ValueError("指标随时效图没有可画的模型")
        if not any(metric in frame.columns for frame in prepared.values()):
            raise ValueError(f"结果表里没有 {metric} 列")
        grades = ordered_grades(list(prepared.values()))
        if not grades:
            raise ValueError("指标随时效图没有可画的数据（降水等级为空）")

        columns = min(len(grades), 3)
        rows = int(np.ceil(len(grades) / columns))
        width = snap_width(figsize[0])
        sizes = font_sizes(width)
        fig, axes = plt.subplots(
            rows, columns, figsize=(width, float(figsize[1])),
            sharex=True, sharey=not zoom_y, squeeze=False,
        )
        axes = axes.ravel()
        colors = model_colors(names)

        for index, grade in enumerate(grades):
            ax = axes[index]
            for name in names:
                frame = prepared[name]
                if metric not in frame.columns:
                    continue
                subset = frame[frame["grade"].astype(str) == grade].sort_values("lead_h")
                values = pd.to_numeric(subset[metric], errors="coerce")
                if subset.empty or not values.notna().any():
                    continue
                ax.plot(
                    subset["lead_h"], values,
                    marker="o", markersize=4, markevery=2, linewidth=1.8,
                    linestyle="-", color=colors[name], label=name,
                )
            ax.set_title(
                f"({chr(ord('a') + index)}) {grade} mm",
                loc="left", fontsize=sizes["title"],
            )
            ax.set_ylabel(metric, fontsize=sizes["label"])
            ax.tick_params(labelsize=sizes["tick"])
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            if metric == "BIAS":
                ax.axhline(
                    1.0, color=NEUTRAL["reference"], linestyle="--",
                    linewidth=1.0, alpha=0.8,
                )
            elif metric in ("TS", "POD", "FAR", "漏报率"):
                ax.set_ylim(0, 1)
        for ax in axes[len(grades):]:
            ax.axis("off")

        # 子图多时 x 刻度会挤成 "100200300"：抽稀到最多 5 个（原来是 3 个，
        # 中间段衰减看不出来）；刻度只落在真实采样时效上。
        all_leads = sorted(
            {float(value) for frame in prepared.values() for value in pd.unique(frame["lead_h"])}
        )
        ticks = lead_ticks(all_leads)
        for ax in axes[: len(grades)]:
            ax.set_xticks(ticks)
            ax.set_xticklabels([f"{value:g}" for value in ticks])
            # sharex 的默认只在最底行显示刻度字；这里每格都打开，
            # 再各自写一次 x 轴名——单看任一子图都能读出时效轴。
            ax.tick_params(axis="x", labelsize=sizes["tick"], labelbottom=True)
            ax.set_xlabel("Lead time (h)", fontsize=sizes["label"])

        handles, labels = axes[0].get_legend_handles_labels()
        if not handles:  # 第一个子图可能全空，图例从任意子图捞
            for ax in axes.ravel():
                handles, labels = ax.get_legend_handles_labels()
                if handles:
                    break
        # 5 个等级排 3×2 时右下角剩一格空白：图例直接放进去，不占图的底部，
        # 也不用在坐标轴外再留一条横带。格子满了才退回底部居中。
        empty = list(axes[len(grades):])
        if handles and empty:
            empty[-1].legend(
                handles, labels, loc="center",
                fontsize=sizes["legend"], **LEGEND_FRAME,
            )
        elif handles:
            fig.legend(
                handles, labels, loc="lower center",
                bbox_to_anchor=(0.5, 0.02),
                ncol=len(labels), fontsize=sizes["legend"], **LEGEND_FRAME,
            )
        bottom = 0.11 if (handles and not empty) else 0.03
        fig.tight_layout(rect=(0, bottom, 1, 1))
        self._save(fig, save_path)
        return fig

    def plot_performance_diagram_multi(
        self,
        model_dfs: Dict[str, pd.DataFrame],
        lead_h: Optional[float] = None,
        figsize: Tuple[float, float] = (9, 9),
        save_path: Optional[Path] = None,
    ):
        """多模型性能图：每个模型一条线，按阈值由小到大连接。

        背景仍是 TS 等值线与 BIAS 参考线；点越靠右上技巧越好。同一条线上
        点的顺序是阈值从小到大——但 TS 不一定随阈值单调，所以连线的方向只
        表示**阈值次序**，不代表模型表现随时间变化。
        """
        prepared = {name: prepare_ts_dataframe(frame) for name, frame in model_dfs.items()}
        names = list(prepared)
        if not names:
            raise ValueError("性能图没有可画的模型")
        first = prepared[names[0]]
        leads = available_leads(first)
        lead = float(lead_h) if lead_h is not None else leads[0]

        width = snap_width(figsize[0])
        # aspect="equal" 让坐标区是个正方形，保存时两侧留白被 tight bbox 裁掉，
        # 成图实际宽度≈figsize[1]。字号按这个宽度算，插进报告（统一按 12 in
        # 显示）才和别的图一样大，不会凭空放大 27%。
        sizes = font_sizes(float(figsize[1]))
        fig, ax = plt.subplots(figsize=(width, float(figsize[1])))
        grid = np.linspace(0.001, 1, 200)
        sr, pod = np.meshgrid(grid, grid)
        with np.errstate(divide="ignore", invalid="ignore"):
            ts = 1.0 / (1.0 / pod + 1.0 / sr - 1.0)
        contour = ax.contour(
            sr, pod, ts,
            levels=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
            colors=NEUTRAL["reference"], linewidths=1, linestyles="--", alpha=0.8,
        )
        ax.clabel(contour, inline=True, fontsize=sizes["annot"], fmt="TS=%.1f")
        for bias in (0.5, 1.0, 1.5, 2.0):
            ax.plot(grid, grid * bias, color=NEUTRAL["reference"], linestyle=":",
                    linewidth=1, alpha=0.8)
            # 标签放在这条线离开坐标框的那个边上：>1 的线从顶边出去，
            # 统一写在 (1,1) 会让 1.5 和 2.0 两个标签叠成一团。
            if bias >= 1.0:
                ax.text(
                    1.0 / bias, 1.0, f"BIAS={bias:g}", ha="center", va="top",
                    fontsize=sizes["annot"], color=NEUTRAL["muted"],
                )
            else:
                ax.text(
                    1.0, bias, f"BIAS={bias:g}", ha="right", va="bottom",
                    fontsize=sizes["annot"], color=NEUTRAL["muted"],
                )

        colors = model_colors(names)
        markers = ("o", "s", "^", "D")
        for model_index, name in enumerate(names):
            frame = prepared[name]
            subset = self._at_lead(frame, lead)
            subset = subset[subset["POD"].notna() & subset["FAR"].notna()]
            if subset.empty:
                continue
            if "threshold_mm" in subset.columns:
                subset = subset.sort_values("threshold_mm")
            ax.plot(
                1.0 - pd.to_numeric(subset["FAR"], errors="coerce"),
                pd.to_numeric(subset["POD"], errors="coerce"),
                color=colors[name], linewidth=1.6,
                linestyle="-",
                marker=markers[model_index % len(markers)],
                markersize=7, markeredgecolor="white", markeredgewidth=0.8,
                label=name, zorder=5,
            )

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        ax.tick_params(labelsize=sizes["tick"])
        ax.set_xlabel("Success ratio 1−FAR", fontsize=sizes["label"])
        ax.set_ylabel("POD", fontsize=sizes["label"])
        # 不写图内标题：时效在报告正文/图注里已经说明，标题只占顶部空间。
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            # 曲线都在左侧中部，右下角是整张图最空的一块，图例放这里不压数据。
            ax.legend(
                handles, labels, loc="lower right",
                fontsize=sizes["legend"], **LEGEND_FRAME,
            )
        fig.tight_layout()
        self._save(fig, save_path)
        return fig

    def plot_lead_dots(
        self,
        model_dfs: Dict[str, pd.DataFrame],
        figsize: Tuple[float, float] = (7.6, 12.6),
        save_path: Optional[Path] = None,
        *,
        benchmark: Optional[float] = 0.1,
    ):
        """分时效综合 TS 点图：行=时效、点=模型，点越靠右越好。

        综合 TS 是各降水等级在同一时效上的等权平均。同一时效的模型点上下错开，
        数值接近时不会互相盖住；每行从左边轴起画一条淡色短横线，当棒棒糖的柄。

        ``benchmark`` 画一条竖直参考线（默认 0.1）：报告 4.1 用「首次跌到 0.1
        以下」描述尾段技巧，点落在线的左侧即达到该提醒条件。设为 None 不画。
        图高取 12.6 in：4 个模型的点各自带三位小数标签，行距不够时尾段
        （312–360h）会叠成一团。
        """
        prepared = {name: prepare_ts_dataframe(frame) for name, frame in model_dfs.items()}
        table = pd.DataFrame(
            {
                name: frame.groupby("lead_h")["TS"].mean()
                for name, frame in prepared.items()
            }
        ).sort_index()
        table = table.dropna(how="all")
        if table.empty:
            raise ValueError("分时效点图没有可画的数据（各时效都没有有效 TS）")

        names = list(table.columns)
        labels = [f"{float(value):g}" for value in table.index]
        colors = model_colors(names)
        width = snap_width(figsize[0])
        sizes = font_sizes(width)
        fig, ax = plt.subplots(figsize=(width, float(figsize[1])))

        positions = np.arange(len(labels), dtype=float)
        offsets = (
            np.linspace(-0.38, 0.38, len(names)) if len(names) > 1
            else np.zeros(1)
        )
        values = table.to_numpy(dtype=float)
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            raise ValueError("分时效点图没有可画的数据（各时效都没有有效 TS）")
        low = float(finite.min())
        high = float(finite.max())
        pad = max((high - low) * 0.12, 0.02)

        for column, name in enumerate(names):
            ys = positions + offsets[column]
            xs = table[name].to_numpy(dtype=float)
            for x, y in zip(xs, ys):
                if not np.isfinite(x):
                    continue
                # 淡色横线从左边轴起画到点：没有参考虚线后，它当棒棒糖的柄，
                # 横排之间的长短差一眼可比。
                ax.plot(
                    [low - pad, x], [y, y], color=colors[name],
                    linewidth=1.6, alpha=0.35, solid_capstyle="round", zorder=2,
                )
            ax.scatter(
                xs, ys, s=64, color=colors[name], edgecolor="white",
                linewidth=1.1, zorder=3, label=name,
            )
            for x, y in zip(xs, ys):
                if not np.isfinite(x):
                    continue
                # 点落在右半区时标在左边，否则贴到画布右缘会被截掉。
                ax.annotate(
                    f"{x:.3f}", (x, y),
                    xytext=(5 if x < 0.5 * (low + high) else -5, 0),
                    textcoords="offset points",
                    ha="left" if x < 0.5 * (low + high) else "right", va="center",
                    fontsize=sizes["annot"], color=NEUTRAL["text"], zorder=4,
                )

        ax.set_yticks(positions)
        ax.set_yticklabels([f"{label}h" for label in labels], fontsize=sizes["tick"])
        # 上下各留 0.65 行：同一时效的模型点要上下错开 ±0.38，只留 0.5 行时
        # 首末两行的三位小数标签会顶出坐标区（360h 那行的标签掉到轴下面）。
        ax.set_ylim(len(labels) - 0.35, -0.65)
        ax.set_xlim(
            min(low - pad, benchmark - 0.5 * pad) if benchmark is not None else low - pad,
            max(high + pad, benchmark + 0.5 * pad) if benchmark is not None else high + pad,
        )
        if benchmark is not None:
            ax.axvline(
                benchmark, color=NEUTRAL["reference"], linestyle="--",
                linewidth=1, alpha=0.8, zorder=1,
            )
            ax.text(
                benchmark, -0.5, f" {benchmark:g}", ha="left", va="center",
                fontsize=sizes["annot"], color=NEUTRAL["muted"], zorder=4,
            )
        ax.tick_params(axis="x", labelsize=sizes["tick"])
        # 每个时效都留一段短刻度线，24h/48h 这类首个时效不会被看成"没画"。
        ax.tick_params(axis="y", length=5, width=0.9, labelsize=sizes["tick"])
        ax.set_xlabel("Composite TS", fontsize=sizes["label"])
        ax.set_ylabel("Lead time (h)", fontsize=sizes["label"])
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        if len(names) > 1:
            handles, labels_ = ax.get_legend_handles_labels()
            # 长时效那几行的点都挤在左侧，右下角整块是空的，图例放这里。
            ax.legend(
                handles, labels_, loc="lower right",
                fontsize=sizes["legend"], **LEGEND_FRAME,
            )
        fig.tight_layout(rect=(0, 0, 1, 1))
        self._save(fig, save_path)
        return fig

    def create_report(
        self,
        df: pd.DataFrame,
        output_dir: Path,
        model_name: str = "model",
        lead_h: Optional[float] = None,
        baseline_df: Optional[pd.DataFrame] = None,
        baseline_name: Optional[str] = None,
        baselines: Optional[Dict[str, pd.DataFrame]] = None,
        sources: Optional[Dict[str, str]] = None,
        change_description: Optional[str] = None,
        region_labels: Optional[Dict[str, str]] = None,
        probability: Optional[Dict[str, pd.DataFrame]] = None,
        members: Optional[int] = None,
        coverage: Optional[Dict[str, int]] = None,
        ensemble: bool = False,
    ) -> Dict[str, Path]:
        """出图 + 写 Markdown 报告，返回 {名称: 路径}。

        图与 ``ts_report.build_ts_report`` 里的图号一一对应：本函数出
        ``metrics_at_lead`` / ``ts_vs_lead`` / ``pod_vs_lead`` / ``far_vs_lead`` /
        ``bias_vs_lead`` / ``TS_boxplot`` / ``performance_diagram`` /
        ``aroc_vs_lead`` / ``bss_vs_lead``，总体热力图与分时效点图由报告模块
        连同正文一起出。``probability`` / ``members`` / ``coverage`` /
        ``ensemble`` 是集合产物专属的，原样透传。
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        data = prepare_ts_dataframe(df)
        # 分纬度带：图和正文一律只看全球行，带行单独进报告的「附 L」（见
        # ``split_regions``）。对比表同样只留全球行——各带平均再平均不等于全球
        # 平均，把带行当全球行拿去比是另一种口径，不是同一件事。
        data, region_frames = split_regions(data)
        # 对比表统一在这里规范化一次：差值图、箱线图、报告表格读的是同一份，
        # spec 有问题的对比表也会在这里就报出清晰的缺列错误。
        baselines = {
            name: split_regions(prepare_ts_dataframe(frame))[0]
            for name, frame in (baselines or {}).items()
        }
        lead = float(lead_h) if lead_h is not None else available_leads(data)[0]

        print(f"\n生成 {model_name} 评测报告（基准时效 {lead:g}h，输出 {output_dir}）")
        artifacts: Dict[str, Path] = {}

        # 箱线图的模型集合与报告里所有对比表**必须是同一个 dict**，
        # 否则"表里的数字和图上的线对不上"会从后门溜回来。
        box_models = collect_box_models(
            model_name,
            data,
            baseline_df=baseline_df,
            baseline_name=baseline_name,
            baselines=baselines,
        )

        path = output_dir / "metrics_at_lead.svg"
        self.plot_metrics_at_lead(box_models, lead_h=lead, save_path=path)
        plt.close("all")
        artifacts["metrics_at_lead"] = path

        # TS / POD / FAR / BIAS 各一张，报告第 4 章各占一节。TS 用自适应纵轴：
        # 几个模型水平接近时，锁 0–1 会把差异压成一条带；绝对水平由报告里的表负责。
        for metric in ("TS", "POD", "FAR", "BIAS"):
            if not any(metric in frame.columns for frame in box_models.values()):
                continue
            key = f"{metric.lower()}_vs_lead"
            path = output_dir / f"{key}.svg"
            self.plot_metric_vs_lead(
                box_models, metric=metric, zoom_y=(metric == "TS"), save_path=path,
            )
            plt.close("all")
            artifacts[key] = path

        box_path = output_dir / "ts_boxplot.svg"
        self.plot_metric_boxplot(box_models, metric="TS", save_path=box_path)
        plt.close("all")
        artifacts["TS_boxplot"] = box_path

        performance_models = {
            name: frame
            for name, frame in box_models.items()
            if {"POD", "FAR"}.issubset(frame.columns)
        }
        if (
            performance_models
            and {"POD", "FAR"}.issubset(data.columns)
            and bool(data[["POD", "FAR"]].notna().any().all())
        ):
            perf_path = output_dir / "performance_diagram.svg"
            self.plot_performance_diagram_multi(
                performance_models, lead_h=lead, save_path=perf_path
            )
            plt.close("all")
            artifacts["performance_diagram"] = perf_path

        # 概率评分的两个指标各出一张图（AROC / BSS 量纲差得远，拼一起会把
        # BSS 压扁）。概率表自己的 6h 窗口与 TS 的累积窗口不是同一个口径，
        # 由报告正文和图注写明，这里不再进标题。
        probability = {
            str(name): loaded
            for name, frame in (probability or {}).items()
            if (loaded := load_probability(frame)) is not None
        }
        for metric in ("AROC", "BSS"):
            if not any(metric in frame.columns for frame in probability.values()):
                continue
            key = f"{metric.lower()}_vs_lead"
            prob_path = output_dir / f"{key}.svg"
            self.plot_probability_vs_lead(
                probability, metric=metric, save_path=prob_path
            )
            plt.close("all")
            artifacts[key] = prob_path

        report_path = output_dir / "REPORT.md"
        build_ts_report(
            data,
            report_path,
            model_name=model_name,
            baseline_df=baseline_df,
            baseline_name=baseline_name,
            baselines=baselines,
            sources=sources,
            change_description=change_description,
            artifacts=artifacts,
            lead_h=lead,
            box_models=box_models,
            region_frames=region_frames,
            region_labels=region_labels,
            probability=probability,
            members=members,
            coverage=coverage,
            ensemble=ensemble,
        )
        artifacts["report"] = report_path
        # 别在这两行用 ✓ / ⚠：Windows 控制台默认 GBK，编码不了这两个字符。报告和图
        # 这时**已经写完了**，却会在最后一行抛 UnicodeEncodeError、进程非 0 退出。
        print(f"报告完成：{report_path}（{len(artifacts) - 1} 张图 + 1 份报告）")
        if self.font is None:
            print("未找到中文字体，图中的中文可能显示为方块；请安装 Noto Sans CJK 或 SimHei")
        return artifacts

    def plot_multi_model_vs_lead(
        self,
        model_dfs: Dict[str, pd.DataFrame],
        metric: str = "TS",
        mode: str = "delta",
        reference: Optional[str] = None,
        figsize: Tuple[int, int] = (16, 7.5),
        save_path: Optional[Path] = None,
        legend_title: str = "Model",
        suptitle: Optional[str] = None,
        zoom_y: bool = False,
    ):
        """各模型指标随时效变化的对比（每个降水等级一个子图）。

        Args:
            mode: ``"delta"``（默认）画相对主模型的**差值**——几个模型水平接近时
                绝对曲线会重合成一团，差值才看得出谁强谁弱；``"absolute"`` 画绝对值。
            reference: 主模型名（差值基准）；默认取 ``model_dfs`` 的第一个。
            legend_title / suptitle: 图例标题与总标题。分纬度带那张图借这个函数画
                （每个带一条线，与"每个模型一条线"结构完全一样），不改标题就会
                写成「各模型」却列着带名。
            zoom_y: ``True`` 时各子图**各自**按数据范围取纵轴（不再锁 0–1），
                用于模型水平接近、需要看清差异的绝对曲线图；比值型指标的绝对
                水平看报告里的表。
        """
        if mode not in ("delta", "absolute"):
            raise ValueError(f"未知 mode: {mode}（可选 delta/absolute）")
        prepared = {name: prepare_ts_dataframe(frame) for name, frame in model_dfs.items()}
        names = list(prepared)
        reference = reference or (names[0] if names else None)
        if mode == "delta" and reference not in prepared:
            raise ValueError(f"差值基准 '{reference}' 不在模型列表 {names} 里")
        grades = ordered_grades(list(prepared.values()))

        columns = min(len(grades), 3)
        rows = int(np.ceil(len(grades) / columns)) if grades else 1
        width = snap_width(figsize[0])
        sizes = font_sizes(width)
        fig, axes = plt.subplots(
            rows, columns, figsize=(width, float(figsize[1])),
            sharex=True, sharey=not zoom_y,
        )
        axes = np.atleast_1d(axes).ravel()
        colors = model_colors(names)

        def _curve(frame: pd.DataFrame, grade: str) -> pd.DataFrame:
            subset = frame[frame["grade"].astype(str) == grade].sort_values("lead_h")
            return subset if metric in subset.columns else subset.iloc[0:0]

        for index, grade in enumerate(grades):
            ax = axes[index]
            base = _curve(prepared[reference], grade) if reference else None
            if mode == "absolute":
                for name, frame in prepared.items():
                    subset = _curve(frame, grade)
                    if subset.empty:
                        continue
                    ax.plot(
                        subset["lead_h"], subset[metric],
                        marker="o", markersize=4, markevery=2, linewidth=1.8,
                        linestyle="-",
                        color=colors[name], label=name,
                    )
            else:
                ax.axhline(0.0, color=NEUTRAL["reference"], linewidth=1, alpha=0.8)
                for name, frame in prepared.items():
                    if name == reference:
                        continue
                    subset = _curve(frame, grade)
                    if subset.empty or base is None or base.empty:
                        continue
                    merged = subset.merge(
                        base[["lead_h", metric]], on="lead_h", suffixes=("", "_base")
                    )
                    if merged.empty:
                        continue
                    ax.plot(
                        merged["lead_h"], merged[metric] - merged[f"{metric}_base"],
                        marker="o", markersize=4, markevery=2, linewidth=1.8,
                        linestyle="-",
                        color=colors[name],
                        label=f"{name} − {reference}",
                    )
            ax.set_title(f"{grade} mm", fontsize=sizes["title"])
            ax.tick_params(labelsize=sizes["tick"])
            if mode == "absolute" and not zoom_y and metric in ("TS", "POD", "FAR", "漏报率"):
                ax.set_ylim(0, 1)
        for ax in axes[len(grades):]:
            ax.axis("off")
        axes[0].set_ylabel(
            metric if mode == "absolute" else f"Δ{metric} vs {reference}",
            fontsize=sizes["label"],
        )
        for ax in axes[:len(grades)]:
            ax.set_xlabel("Lead time (h)", fontsize=sizes["label"])
        handles, labels = axes[0].get_legend_handles_labels()
        empty = list(axes[len(grades):])
        if handles and empty:
            # 右下角的空格子优先放图例；格子满了才回到图底。
            empty[-1].legend(
                handles, labels, loc="center", title=legend_title,
                fontsize=sizes["legend"], frameon=True,
            )
        elif handles:
            fig.legend(
                handles, labels, loc="lower center", ncol=min(len(labels), 3),
                frameon=True, title=legend_title, fontsize=sizes["legend"],
            )
        fig.suptitle(
            suptitle
            or (
                f"{metric} difference vs {reference} by lead time (>0 = better)"
                if mode == "delta"
                else f"{metric} vs lead time"
            ),
            fontsize=sizes["suptitle"],
        )
        # 图例在底部时占两行（标题 + 模型），留 12%；进了空格子就只留边距。
        outside_legend = handles and not empty
        fig.tight_layout(rect=(0, 0.12, 1, 0.96) if outside_legend else (0, 0.02, 1, 0.96))
        self._save(fig, save_path)
        return fig

    def plot_probability_vs_lead(
        self,
        model_dfs: Dict[str, pd.DataFrame],
        *,
        metric: str = "AROC",
        figsize: Tuple[int, int] = (12, 9.8),
        save_path: Optional[Path] = None,
    ):
        """**一个**概率指标（``AROC`` 或 ``BSS``）随时效变化——2×2 面板，一格一个阈值。

        吃的是概率宽表（``diagnostics/probability_wide.csv``），**不是** TS 宽表：
        列的集合完全不同，所以走 ``prepare_ts_dataframe`` 会在第一步就报缺列。
        这里只按 ``region`` 空值筛全球行——与 ``split_regions`` 对"全球行"的定义
        一致，免得带行混进来把曲线拉平。

        AROC / BSS 拆成两张图单独出：两者量纲差得远（BSS 能到 −1），拼成上下
        两排时 BSS 那排被压得很扁。面板排成 2×2 而不是一横排：一横排四个面板
        每个只有 ~2.6 in 宽，曲线挤成一条带；2×2 里每格接近 5 in，横轴和纵轴
        都拉得开，阈值间差异才看得出。

        AROC 纵轴从**各模型各阈值的共同最低值稍下一点**起（动态下界，不会裁数据），
        刻度每 0.1 一格；BSS 一律自适应——它是"相对气候基准的技巧"，负多少都有可能，
        固定下界就是在藏坏消息。

        所有模型一律实线，靠颜色和圆点区分；虚线/点划线在 2×2 小面板里容易看花。

        无技巧线（AROC=0.5 / BSS=0）不再画进图里，改由报告正文说明；图里只留
        曲线、坐标轴和底部图例。

        Args:
            model_dfs: ``{模型名: 概率宽表}``，主模型在前。颜色按模型名取
                （见 :func:`style.model_colors`），与箱线图、差值图同一份配色。
            metric: ``"AROC"`` 或 ``"BSS"``。
        """
        if metric not in ("AROC", "BSS"):
            raise ValueError(f"未知概率指标: {metric}（可选 AROC/BSS）")

        prepared: Dict[str, pd.DataFrame] = {}
        for name, frame in model_dfs.items():
            subset = frame
            if "region" in frame.columns:
                region = frame["region"]
                subset = frame[region.isna() | (region.astype(str).str.strip() == "")]
            prepared[name] = subset

        grades = ordered_grades(list(prepared.values()))
        if not grades:
            return None
        width = snap_width(figsize[0])
        sizes = font_sizes(width)
        # 2×2：四个阈值一格一个，横轴比一横排时长了近一倍；
        # 只有一档时退化成 1×1，不留空格。
        ncols = 2 if len(grades) > 1 else 1
        nrows = int(np.ceil(len(grades) / ncols))
        fig, axes = plt.subplots(
            nrows, ncols, figsize=(width, float(figsize[1])),
            sharex=True, squeeze=False,
        )
        panels = list(axes.ravel())
        for ax in panels[len(grades):]:
            # 只关坐标轴装饰，不整体隐藏：空面板还能用来放图例。
            ax.axis("off")
        colors = model_colors(list(prepared))

        plotted_values: List[pd.Series] = []
        for column, grade in enumerate(grades):
            ax = panels[column]
            row, col = divmod(column, ncols)
            for name, frame in prepared.items():
                if metric not in frame.columns:
                    continue
                subset = frame[frame["grade"].astype(str) == grade].sort_values("lead_h")
                values = pd.to_numeric(subset[metric], errors="coerce")
                if subset.empty or not values.notna().any():
                    continue
                plotted_values.append(values)
                ax.plot(
                    subset["lead_h"], values,
                    marker="o", markersize=4, markevery=2, linewidth=1.8,
                    linestyle="-",
                    color=colors[name], label=name,
                )
            ax.set_title(
                f"({chr(ord('a') + column)}) {grade} mm",
                loc="left", fontsize=sizes["title"],
            )
            # 轴名只放在左列 / 末行：2×2 里每格都写一遍是重复信息。
            if row == nrows - 1:
                ax.set_xlabel("Lead time (h)", fontsize=sizes["label"])
            if col == 0:
                ax.set_ylabel(metric, fontsize=sizes["label"])
            ax.tick_params(labelsize=sizes["tick"])
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)

        if metric == "AROC":
            # 下界跟着数据走：从所有模型、所有阈值里的最低值再往下让 0.02，
            # 取两位小数向下取整（例如最低 0.534 → 0.51），再留 0.1 的刻度。
            # 0.25/0.5 这类固定下界在数据整体较高时会把曲线压成一条平带。
            data_min = min(
                (float(series.min()) for series in plotted_values if len(series)),
                default=None,
            )
            if data_min is None:
                lower = 0.25
            else:
                lower = max(0.0, float(np.floor((data_min - 0.02) * 100) / 100))
            for ax in panels[: len(grades)]:
                ax.set_ylim(lower, 1.0)
                ax.yaxis.set_major_locator(MultipleLocator(0.1))

        # 时效刻度只落在真实采样点上；2×2 每格接近 5 in 宽，
        # 放得下 5 个三位数标签（一横排时只放得下 4 个）。
        all_leads = sorted(
            {
                float(value)
                for frame in prepared.values()
                for value in pd.to_numeric(frame["lead_h"], errors="coerce").dropna()
            }
        )
        ticks = lead_ticks(all_leads, max_ticks=5)
        for column in range(len(grades)):
            ax = panels[column]
            row = column // ncols
            ax.set_xticks(ticks)
            ax.set_xticklabels([f"{value:g}" for value in ticks])
            ax.tick_params(
                axis="x", labelbottom=(row == nrows - 1), labelsize=sizes["tick"],
            )

        handles, labels = panels[0].get_legend_handles_labels()
        if not handles:  # 第一个子图可能全空时图例得从别处捞
            for ax in panels[: len(grades)]:
                handles, labels = ax.get_legend_handles_labels()
                if handles:
                    break
        empty = list(panels[len(grades):])
        if handles and empty:
            # 右下角空格子优先放图例；四个阈值刚好填满 2×2 时才回到图底。
            empty[-1].legend(
                handles, labels, loc="center",
                fontsize=sizes["legend"], **LEGEND_FRAME,
            )
        elif handles:
            fig.legend(
                handles, labels, loc="lower center",
                bbox_to_anchor=(0.5, 0.01),
                ncol=min(len(labels), 4),
                fontsize=sizes["legend"], **LEGEND_FRAME,
            )
        # 图例在格子里时底部只留 x 轴名；在格子里则不占额外高度。
        bottom = 0.12 if (handles and not empty) else 0.03
        # h_pad / w_pad 都比默认宽，让面板之间留出真正的空白。
        fig.tight_layout(rect=(0, bottom, 1, 1), h_pad=3.0, w_pad=2.0)
        self._save(fig, save_path)
        return fig
