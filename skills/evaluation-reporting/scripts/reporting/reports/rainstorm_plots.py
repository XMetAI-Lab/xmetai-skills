# -*- coding: utf-8 -*-
"""暴雨过程个例检验的出图器（多模型 × 多过程 × 双口径）。

只画图，统计量与正文在 :mod:`reporting.reports.rainstorm_report`；
图用的是它算好的同一批统计量。四类图都多模型并排：过程总览分组条形（两种口径
各一张）、单过程逐日 24h TS 折线（按等级分面）、单过程累积 TS 条形、
A/B 口径散点（带 ``y = x`` 参考线，看名次会不会翻转）。

不出降水空间分布图——产物只有站点列联表的聚合统计，没有逐站/逐格降水场。
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np

from reporting.core.style import (
    NEUTRAL,
    apply_style,
    font_sizes,
    model_colors,
    snap_width,
)

#: 折线/条形里"没有有效取值"的图元直接跳过，而不是画成 0——
#: TS 的 NaN 是无有效配对或无事件，与"技巧为 0"是两回事（SKILL.md 硬性规则 4）。
_MISSING = "—"


class RainstormPlotter:
    """暴雨过程个例检验的绘图器。

    Args:
        style: 可选 matplotlib 样式名；默认不套样式（样式会覆盖中文字体设置）。
    """

    def __init__(self, style: Optional[str] = None):
        import matplotlib.pyplot as plt

        self._plt = plt
        self.font = apply_style()
        if style:
            plt.style.use(style)
            self.font = apply_style()
        plt.rcParams["figure.dpi"] = 150
        plt.rcParams["savefig.dpi"] = 150
        plt.rcParams["savefig.bbox"] = "tight"

    # ------------------------------------------------------------------ 私有

    def _save(self, fig, save_path: Optional[Path]) -> None:
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path)
            print(f"已保存: {save_path}")
        self._plt.close(fig)

    # ------------------------------------------------------------------ 图

    def plot_process_overview(
        self,
        codes: Sequence[str],
        values: Dict[str, Dict[str, float]],
        *,
        threshold: float,
        save_path: Optional[Path] = None,
        order: Optional[Sequence[str]] = None,
        metric_label: str = "过程累积",
    ) -> None:
        """43 个过程的分组条形，按 ``order``（默认第一个模型）降序。

        Args:
            codes: 过程编号，决定横轴顺序。
            values: ``{模型名: {编号: TS}}``；缺的过程画成空位（不补 0）。
            threshold: 图题里的阈值，只影响文字。
            order: 排序依据的模型名；缺省取 ``values`` 的第一个模型。
            metric_label: 纵轴与图题里的口径名，如 ``过程累积`` / ``逐日 24h 汇总``；
                **只影响文字**，数值由调用方决定用哪个口径。
        """
        plt = self._plt
        names = list(values)
        ranked_by = order or (names[0] if names else None)
        ordered = sorted(
            codes,
            key=lambda code: (
                values.get(ranked_by, {}).get(code) is None,
                -(values.get(ranked_by, {}).get(code) or 0.0),
            ),
        )
        colors = model_colors(names)
        positions = np.arange(len(ordered), dtype=float)
        width = 0.8 / max(len(names), 1)

        width = snap_width(max(10.0, 0.34 * len(ordered)))
        sizes = font_sizes(width)
        fig, ax = plt.subplots(figsize=(width, 5.4))
        for index, name in enumerate(names):
            series = [values.get(name, {}).get(code) for code in ordered]
            xs = [x for x, value in zip(positions, series) if value is not None]
            ys = [value for value in series if value is not None]
            ax.bar(
                np.array(xs) + (index - (len(names) - 1) / 2) * width,
                ys,
                width=width,
                label=name,
                color=colors[name],
            )
        ax.set_xticks(positions)
        ax.set_xticklabels(ordered, rotation=90, fontsize=sizes["tick"])
        ax.set_ylabel(
            f"{metric_label} TS（≥{threshold:g}mm）", fontsize=sizes["label"]
        )
        ax.tick_params(axis="y", labelsize=sizes["tick"])
        ax.set_title(
            f"各过程{metric_label} TS（≥{threshold:g}mm，按 {ranked_by} 降序）",
            fontsize=sizes["suptitle"],
        )
        ax.set_ylim(bottom=0)
        ax.grid(axis="y")
        ax.legend(
            fontsize=sizes["legend"], ncols=min(len(names), 4)
        )
        fig.tight_layout()
        self._save(fig, save_path)

    def plot_daily_ts(
        self,
        days: Sequence[str],
        series: Dict[str, Dict[str, Dict[float, Optional[float]]]],
        thresholds: Sequence[float],
        *,
        code: str,
        span: str,
        save_path: Optional[Path] = None,
    ) -> None:
        """单个过程的逐日 24h TS 折线，一个阈值一个子图。

        Args:
            days: 起报日，按顺序决定横轴。
            series: ``{模型名: {起报日: {阈值: TS}}}``。
            thresholds: 要分面的阈值（升序）。
        """
        plt = self._plt
        names = list(series)
        colors = model_colors(names)
        short = [_short_day(day) for day in days]

        width = snap_width(4.2 * max(len(thresholds), 1))
        sizes = font_sizes(width)
        fig, axes = plt.subplots(
            1, max(len(thresholds), 1), figsize=(width, 3.8),
            squeeze=False, sharey=True,
        )
        for column, threshold in enumerate(thresholds):
            ax = axes[0][column]
            for name in names:
                values = [
                    series.get(name, {}).get(day, {}).get(threshold) for day in days
                ]
                xs = [x for x, value in zip(short, values) if value is not None]
                ys = [value for value in values if value is not None]
                if xs:
                    ax.plot(xs, ys, marker="o", markersize=4, label=name, color=colors[name])
            ax.set_title(f"≥{threshold:g}mm", fontsize=sizes["title"])
            ax.grid()
            ax.set_ylim(bottom=0)
            ax.tick_params(labelsize=sizes["tick"])
        axes[0][0].set_ylabel("TS", fontsize=sizes["label"])
        axes[0][0].legend(fontsize=sizes["legend"])
        fig.suptitle(
            f"{code}（{span}）逐日 24h TS（08–08 北京时，day-1）",
            fontsize=sizes["suptitle"],
        )
        fig.tight_layout()
        self._save(fig, save_path)

    def plot_threshold_bars(
        self,
        thresholds: Sequence[float],
        values: Dict[str, Dict[float, Optional[float]]],
        *,
        code: str,
        span: str,
        save_path: Optional[Path] = None,
    ) -> None:
        """单个过程的累积 TS 按阈值分组的条形，每个模型一组。"""
        plt = self._plt
        names = list(values)
        colors = model_colors(names)
        labels = [f"≥{threshold:g}" for threshold in thresholds]
        positions = np.arange(len(thresholds), dtype=float)
        width = 0.8 / max(len(names), 1)

        width = snap_width(max(6.0, 1.3 * len(thresholds)))
        sizes = font_sizes(width)
        fig, ax = plt.subplots(figsize=(width, 4.6))
        for index, name in enumerate(names):
            series = [values.get(name, {}).get(threshold) for threshold in thresholds]
            xs = [x for x, value in zip(positions, series) if value is not None]
            ys = [value for value in series if value is not None]
            ax.bar(
                np.array(xs) + (index - (len(names) - 1) / 2) * width,
                ys,
                width=width,
                label=name,
                color=colors[name],
            )
        ax.set_xticks(positions)
        ax.set_xticklabels(labels, fontsize=sizes["tick"])
        ax.tick_params(axis="y", labelsize=sizes["tick"])
        ax.set_xlabel("降水等级（mm）", fontsize=sizes["label"])
        ax.set_ylabel("过程累积 TS", fontsize=sizes["label"])
        ax.set_title(
            f"{code}（{span}）过程累积 TS 随降水等级的变化",
            fontsize=sizes["suptitle"],
        )
        ax.set_ylim(bottom=0)
        ax.grid(axis="y")
        ax.legend(fontsize=sizes["legend"], ncols=min(len(names), 4))
        fig.tight_layout()
        self._save(fig, save_path)

    def plot_caliber_scatter(
        self,
        points: Dict[str, Sequence[tuple]],
        *,
        threshold: float,
        save_path: Optional[Path] = None,
    ) -> None:
        """逐过程散点：横轴口径 A 起报日汇总、纵轴口径 B 累积，带 ``y = x`` 参考线。

        落在这条线**上方**的过程，说明它的累积 TS 高于按逐日平均外推的水平；
        偏离越大，两个口径给出的"这个过程好不好预报"差得越远。点都是有效
        配对（两个口径都有数）——缺一样的那个过程整点不画。

        Args:
            points: ``{模型名: [(口径A 汇总 TS, 口径B 累积 TS), …]}``。
        """
        plt = self._plt
        names = list(points)
        colors = model_colors(names)

        width = snap_width(5.6)
        sizes = font_sizes(width)
        fig, ax = plt.subplots(figsize=(width, 5.2))
        for name in names:
            pairs = [(x, y) for x, y in points.get(name, []) if x is not None and y is not None]
            if not pairs:
                continue
            xs = [pair[0] for pair in pairs]
            ys = [pair[1] for pair in pairs]
            ax.scatter(xs, ys, s=26, label=f"{name}（{len(pairs)} 个过程）",
                       color=colors[name], alpha=0.8, edgecolors="none")
        ax.plot([0, 1], [0, 1], linestyle="--", linewidth=0.9,
                color=NEUTRAL["reference"])
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel(
            f"口径 A：逐日 24h 汇总 TS（≥{threshold:g}mm）",
            fontsize=sizes["label"],
        )
        ax.set_ylabel(
            f"口径 B：过程累积 TS（≥{threshold:g}mm）",
            fontsize=sizes["label"],
        )
        ax.tick_params(labelsize=sizes["tick"])
        ax.set_title(
            "两种口径的逐过程对照（虚线为 y = x）", fontsize=sizes["suptitle"]
        )
        ax.grid()
        ax.legend(fontsize=sizes["legend"])
        fig.tight_layout()
        self._save(fig, save_path)


def _short_day(day: str) -> str:
    """``20250311`` → ``0311``，折线横轴用。不是八位数字就原样返回。"""
    text = str(day)
    return text[4:] if len(text) == 8 and text.isdigit() else text
