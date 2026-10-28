#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""确定性多模型对比出图（RMSE / ACC / FA / 纬向谱）。

与 ``field_plots`` / ``precipitation_plots`` 的分工：那两份画的是**单模型**的产物，
这一份画的是**同一指标、多个模型**叠在一起的对比图——
单成员报告的 RMSE / ACC / FA 和集合报告正文 §4.1–§4.5 的指标曲线都由它出。

配色默认走 :func:`visualization.style.model_colors`：同一个模型在任何图、任何
命令行次序下都是同一个颜色。调用方仍可传 ``palette="Dark2"`` 临时改用
ColorBrewer 离散色板，或用 ``color_map`` 按名字覆盖单个模型。

约定：``series_by_model`` 是 ``{模型名: DataFrame}``，DataFrame 的
index 是预报时效（h），columns 是变量名——和逐日 ``rmse_<date>_det.csv`` 的
布局一致（那边 index 就是 lead_h）。
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from matplotlib.colors import to_rgba
from matplotlib.patches import ConnectionPatch, Rectangle
from matplotlib.ticker import MaxNLocator, MultipleLocator

from reporting.core.common import fmt_number
from reporting.core.style import (
    CMAP_SKILL,
    LEGEND_FRAME,
    NEUTRAL,
    ZOOM_HIGHLIGHT,
    apply_style,
    font_sizes,
    model_colors,
    snap_width,
)


def brewer_colors(palette: str, names: Sequence[str]) -> Dict[str, str]:
    """ColorBrewer 离散色板（``Set1`` / ``Dark2`` …）按模型次序取色。

    论文图用 ``Dark2``：8 色都足够深，黑白打印也能区分六条线；想要更鲜艳可以用
    ``Set1``，但它含亮黄，浅色背景上最后一条线会发虚。名字按出现次序逐个对应，
    超过色板长度时循环——同一批图传同样的 ``names`` 次序，模型颜色就一致。

    matplotlib 3.5 里 ``plt.colormaps`` 还是函数、不能下标取，所以两种写法都兜住。
    """
    names = [str(name) for name in names]
    try:
        cmap = plt.colormaps[palette]           # matplotlib >= 3.6 的注册表
    except (AttributeError, TypeError, KeyError):
        cmap = plt.get_cmap(palette)            # 3.5 的旧接口 / 未知名字的清晰报错
    colors = list(getattr(cmap, "colors", None) or [])
    if not colors:                              # 连续色板：等间隔采样
        colors = [
            cmap(index / max(len(names) - 1, 1)) for index in range(len(names))
        ]
    return {
        name: colors[index % len(colors)] for index, name in enumerate(names)
    }


def _numeric_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """把非数值列/行清掉，剩下纯 float 的 lead × 变量表。"""
    data = frame.copy()
    if not isinstance(data.index, pd.Index):
        raise ValueError("series_by_model 的值得是 DataFrame（index=lead_h，columns=变量）")
    data.index = pd.to_numeric(data.index, errors="coerce")
    data = data[data.index.notna()]
    data = data.apply(pd.to_numeric, errors="coerce")
    return data.sort_index()


def _draw_zoom_insets(
    axis,
    frames: Mapping[str, pd.DataFrame],
    variable: str,
    colors: Mapping[str, str],
    line_styles: Mapping[str, object],
    windows: Sequence[Tuple[float, float]],
    positions: Sequence[Sequence[float]],
    *,
    linewidth: float,
    tick_labelsize: float = 7.0,
    tick_steps: Optional[Mapping[str, float]] = None,
    x_tick_step: Optional[float] = None,
) -> None:
    """圈出前期/后期窗口，并在空角落放放大后的 inset。

    窗口范围用极浅灰色高亮框圈出（灰色实线描边 1.6 pt，底色压在曲线下方）；inset
    只画放大后的曲线，不画轴标签、标题和网格。纵轴自动取 6 档以内、档数不多时
    数字全标（细分到 1.0/1.1/1.2 这种量级），超过 8 档才隔一个标；横轴
    ``x_tick_step`` 给了才画刻度，数字隔一个标一个，None 时横轴不画刻度。
    刻度放在 inset 的左或右侧，避开主图的轴标签/曲线。两条细淡虚线把 inset
    靠长方形那条边的两个角连到长方形靠 inset 那条边的两个角，形成「放大镜」
    关系。窗口里没有有限值就跳过，不画空框。
    """
    for index, (start, end) in enumerate(windows):
        if not math.isfinite(start) or not math.isfinite(end) or end <= start:
            continue
        series_by_name: Dict[str, pd.Series] = {}
        low, high = math.inf, -math.inf
        for name, frame in frames.items():
            if variable not in frame.columns:
                continue
            series = frame[variable].dropna()
            series = series[np.isfinite(series)].sort_index()
            if series.empty:
                continue
            # 窗口可能比两个 lead 的间距还窄（如 6–11h，而数据只有 6/12h）：
            # inset 仍画整条曲线、靠 xlim 裁到窗口，纵轴范围则用窗口两端的
            # 线性插值补上——只按窗口内的实测点算，框会缩成一条缝、曲线被裁没。
            if series.index.max() < start or series.index.min() > end:
                continue
            inside = series[(series.index >= start) & (series.index <= end)]
            edges = np.interp(
                [start, end], series.index.to_numpy(dtype=float),
                series.to_numpy(dtype=float),
            )
            values = np.concatenate([inside.to_numpy(dtype=float), edges])
            values = values[np.isfinite(values)]
            if values.size == 0:
                continue
            series_by_name[name] = series
            low = min(low, float(values.min()))
            high = max(high, float(values.max()))
        if not series_by_name or not math.isfinite(low) or not math.isfinite(high):
            continue
        span = high - low
        pad = span * 0.08 if span > 0 else max(abs(high) * 0.02, 0.05)
        box_low, box_high = low - pad, high + pad

        # 完整长方形，不拆角；灰色实线描边 + 极浅灰底，标出被放大的区域。
        # 中性色不会和模型线（尤其 S2B 的橙线）抢注意力。底色压在曲线下方
        # （zorder 1.5），只做高亮，不遮住模型颜色。
        axis.add_patch(
            Rectangle(
                (start, box_low), end - start, box_high - box_low,
                facecolor=to_rgba(ZOOM_HIGHLIGHT["face"], 0.10),
                edgecolor=to_rgba(ZOOM_HIGHLIGHT["edge"], 0.85),
                linewidth=1.6, linestyle="-", zorder=1.5,
            )
        )

        bounds = (
            tuple(positions[index])
            if index < len(positions)
            else (0.02, 0.62, 0.36, 0.35)
        )
        inset = axis.inset_axes(bounds)
        inset.set_zorder(7)
        for name, series in series_by_name.items():
            inset.plot(
                series.index, series.values,
                color=colors[name],
                linestyle=line_styles.get(name, "-"),
                linewidth=linewidth,
            )
        inset.set_xlim(start, end)
        inset.set_ylim(box_low, box_high)
        inset.set_facecolor("white")
        inset.patch.set_alpha(0.95)
        if x_tick_step is not None:
            # 横轴刻度间距做小（如 6 h），起到「放大」的读数效果；窗口内 4 档
            # 以内全标，再多才隔一个标，避免小图里数字挤成一团。**压在窗口右边缘
            # 的那一档**只留刻度线、不写数字——它和右下角的纵轴标签挤在同一个角上。
            inset.xaxis.set_major_locator(MultipleLocator(float(x_tick_step)))
            # get_xticks 会带出窗口外的刻度（如 0），先裁到放大窗口内。
            ticks = [
                value for value in inset.get_xticks()
                if start <= value <= end
            ]
            label_every = 1 if len(ticks) <= 4 else 2
            edge_label = bool(ticks) and abs(ticks[-1] - end) <= 1e-9
            inset.set_xticks(ticks)
            inset.set_xticklabels(
                [
                    f"{value:g}"
                    if index % label_every == 0
                    and not (edge_label and index == len(ticks) - 1)
                    else ""
                    for index, value in enumerate(ticks)
                ],
                fontsize=tick_labelsize,
            )
            inset.tick_params(
                axis="x", which="major", direction="out", length=2.5,
                width=0.8, color=NEUTRAL["text"], labelcolor=NEUTRAL["text"],
                pad=1.5,
            )
        else:
            inset.set_xticks([])
        # 只标纵轴刻度：inset 在左半边时放右侧，在右半边时放左侧，避开主图坐标轴。
        # z500 这类跨度大的变量用 tick_steps 固定间隔（如 10），其余自动取
        # 6 档以内。档数给到 6（而不是 4）是为了让 u850 这类变量落在
        # 0.9/1.0/1.1 的细分上，放大图的「分度值」才看得出比主图小一档；
        # 刻度只取 1/2/2.5/5 这些整齐倍数，避免出现 0.15 这种读不出的间隔。
        tick_step = (tick_steps or {}).get(variable)
        if tick_step is not None:
            inset.yaxis.set_major_locator(MultipleLocator(float(tick_step)))
        else:
            inset.yaxis.set_major_locator(
                MaxNLocator(nbins=6, steps=[1, 2, 5, 10])
            )
        inset.ticklabel_format(axis="y", style="plain", useOffset=False)
        # 3 档以内全标，再多就隔一个标一个——小图里数字不挤。锚点从**最上面**
        # 那档往下数，且最底下一档只留刻度线、不写数字（它会和横轴数字撞在
        # 同一个角上）。
        y_ticks = [
            value for value in inset.get_yticks()
            if box_low - 1e-12 <= value <= box_high + 1e-12
        ]
        label_every = 1 if len(y_ticks) <= 3 else 2
        top = len(y_ticks) - 1
        inset.set_yticks(y_ticks)
        inset.set_yticklabels(
            [
                f"{value:g}"
                if (top - index) % label_every == 0 and index != 0
                else ""
                for index, value in enumerate(y_ticks)
            ],
            fontsize=tick_labelsize,
        )
        if bounds[0] + bounds[2] / 2.0 < 0.5:
            inset.yaxis.set_ticks_position("right")
        else:
            inset.yaxis.set_ticks_position("left")
        inset.tick_params(
            axis="y", which="major", labelsize=tick_labelsize,
            direction="out", length=2.5, width=0.8,
            color=NEUTRAL["text"], labelcolor=NEUTRAL["text"], pad=2.0,
        )
        inset.grid(False)
        for spine in inset.spines.values():
            spine.set_edgecolor(NEUTRAL["spine"])
            spine.set_linewidth(1.0)

        # 两条引导线，从 inset 靠长方形那条边的两个角连到长方形靠 inset 那条边
        # 的两个角，形成放大镜效果。注意 xyB 用的是 inset.transAxes，锚点必须写
        # inset 自己的 0–1 坐标；写父图的 bounds 会让端点落进 inset 内部、被盖住。
        box_center_axes_y = axis.transAxes.inverted().transform(
            axis.transData.transform(
                ((start + end) / 2.0, (box_low + box_high) / 2.0)
            )
        )[1]
        inset_center_y = bounds[1] + bounds[3] / 2.0
        if inset_center_y >= box_center_axes_y:
            inset_anchors = ((0.0, 0.0), (1.0, 0.0))   # 下边两角
            box_anchors = ((start, box_high), (end, box_high))
        else:
            inset_anchors = ((0.0, 1.0), (1.0, 1.0))   # 上边两角
            box_anchors = ((start, box_low), (end, box_low))
        for box_anchor, inset_anchor in zip(box_anchors, inset_anchors):
            connector = ConnectionPatch(
                xyA=box_anchor, coordsA=axis.transData,
                xyB=inset_anchor, coordsB=inset.transAxes,
                color=NEUTRAL["grid"], linestyle="--", linewidth=1.1, alpha=0.9,
                zorder=5, clip_on=False, shrinkA=0, shrinkB=0,
            )
            axis.figure.add_artist(connector)


def _is_number(value) -> bool:
    """能不能当有限浮点数用（``None`` / ``NaN`` / 非数值串都算不能）。

    雷达图拿它判「这根轴这个模型有没有值」——缺值要整根剔除，
    不能当 0 画（那会把模型画成某一项能力为零，是另一回事）。
    """
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def radar_scores(
    raw: Mapping[str, Mapping[str, float]],
    axes: Sequence[str],
    higher_is_better: Mapping[str, bool],
) -> Dict[str, Dict[str, float]]:
    """把各模型的**原始**综合评分折算成雷达图用的**批内 min-max** 得分。

    每根轴上 ``1 = 本批最好``、``0 = 本批最差``，方向由 ``higher_is_better``
    定（缺省按"越低越好"，与 RMSE / 偏差这类指标的惯例一致）。

    这是**相对分**，不是绝对能力分：两个模型时必然是 1 和 0。报告里必须写明。

    两个退化情形：

    - 某根轴全批打平（极差为 0）→ 一律给 1.0。除零会炸，而且"大家一样"
      本来就该画在满格上，画成 0 会让多边形凭空凹进去；
    - 某根轴只要有模型缺值 → **整根剔除**（不出现在返回值的键里）。
      雷达图没有断点，少画一个顶点会把多边形拉歪。

    抽成模块级函数是**故意的**：绘图（:meth:`DetPlotter.plot_capability_radar`）
    与正文的判读句都要用这两个退化规则，各写一份迟早会漂。
    """
    names = [str(name) for name in raw]
    kept = [
        str(axis) for axis in axes
        if all(_is_number(raw.get(name, {}).get(axis)) for name in names)
    ]
    scores: Dict[str, Dict[str, float]] = {name: {} for name in names}
    for axis_name in kept:
        values = {name: float(raw[name][axis_name]) for name in names}
        low, high = min(values.values()), max(values.values())
        span = high - low
        for name, value in values.items():
            if span <= 0:
                scores[name][axis_name] = 1.0
            elif higher_is_better.get(axis_name, False):
                scores[name][axis_name] = (value - low) / span
            else:
                scores[name][axis_name] = (high - value) / span
    return scores


def plot_overall_heatmap(
    names: Sequence[str],
    columns: Sequence[Tuple[str, Mapping[str, float], bool, int]],
) -> plt.Figure:
    """总体指标热力图：行是模型、列是指标，格内数字是原始值。

    ``columns`` 每项是 ``(列名, 各模型取值, 是否越大越好, 小数位)``；颜色是
    **列内**批内归一化的相对表现（1 = 本批最优、0 = 最差），量纲不同，
    只在同一列内可比。缺值的指标整列不画。行序由调用方排好。
    """
    labels = [label for label, *_ in columns]
    direction = {label: higher for label, _, higher, _ in columns}
    raw = {
        name: {label: float(series.get(name, np.nan)) for label, series, _, _ in columns}
        for name in names
    }
    scores = radar_scores(raw, labels, direction)
    kept = [label for label in labels if label in scores[names[0]]]
    if not kept:
        raise ValueError("总体指标一个都没有，画不了热力图")
    matrix = np.array(
        [[scores[name][label] for label in kept] for name in names], dtype=float
    )

    apply_style()
    width = snap_width(7.2)
    sizes = font_sizes(width)
    figure, axis = plt.subplots(
        figsize=(width, max(3.2, 0.55 * len(names) + 2.0))
    )
    cmap = CMAP_SKILL.copy()
    cmap.set_bad(NEUTRAL["panel"])
    image = axis.imshow(
        np.ma.masked_invalid(matrix), aspect="auto", cmap=cmap,
        vmin=0.0, vmax=1.0, origin="upper", interpolation="nearest",
    )
    digits_by_label = {label: digits for label, _, _, digits in columns}
    for row, name in enumerate(names):
        for column, label in enumerate(kept):
            value = matrix[row, column]
            if not np.isfinite(value):
                continue
            # 按格子实际颜色选字色：深蓝/深红上白字，浅蓝/灰/浅红上黑字。
            red, green, blue = cmap(value)[:3]
            luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
            axis.text(
                column, row, fmt_number(raw[name][label], digits_by_label[label]),
                ha="center", va="center", fontsize=sizes["annot"],
                color="white" if luminance < 0.5 else NEUTRAL["text"],
            )
    axis.set_xticks(range(len(kept)))
    axis.set_xticklabels(
        kept, rotation=45, ha="right", rotation_mode="anchor",
        fontsize=sizes["tick"],
    )
    axis.set_yticks(range(len(names)))
    axis.set_yticklabels(names, fontsize=sizes["tick"])
    # 白色细线分格；热力图不要外框和刻度短线（数据本身就是格子）。
    axis.set_axisbelow(False)
    axis.set_xticks(np.arange(-0.5, len(kept)), minor=True)
    axis.set_yticks(np.arange(-0.5, len(names)), minor=True)
    axis.grid(which="minor", color="white", linewidth=1.0, zorder=3)
    axis.tick_params(which="both", length=0)
    for spine in axis.spines.values():
        spine.set_visible(False)

    bar = figure.colorbar(
        image, ax=axis, fraction=0.035, pad=0.02,
        ticks=[0.0, 0.25, 0.5, 0.75, 1.0],
    )
    bar.ax.tick_params(
        labelsize=sizes["annot"], length=2.5, width=0.8,
        color=NEUTRAL["text"], labelcolor=NEUTRAL["text"],
    )
    bar.outline.set_edgecolor(NEUTRAL["spine"])
    bar.outline.set_linewidth(0.6)
    figure.tight_layout()
    return figure


class DetPlotter:
    """多模型确定性指标对比图的绘图器。"""

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
    def _save(self, fig, save_path: Optional[Path], dpi: Optional[int] = None) -> None:
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=dpi)
            print(f"已保存: {save_path}")

    # ------------------------------------------------------------------ #
    def plot_model_panels(
        self,
        series_by_model: Mapping[str, pd.DataFrame],
        variables: Sequence[str],
        *,
        ylabel: str,
        ref_line: Optional[float] = None,
        ncols: int = 4,
        suptitle: Optional[str] = None,
        save_path: Optional[Path] = None,
        panel_size: Tuple[float, float] = (3.4, 2.9),
        linewidth: float = 1.6,
        markersize: float = 3.0,
        markevery: Optional[int] = None,
        zoom_y: bool = False,
        tick_labelsize: Optional[float] = None,
        label_fontsize: Optional[float] = None,
        title_fontsize: Optional[float] = None,
        legend_fontsize: Optional[float] = None,
        palette: Optional[str] = None,
        color_map: Optional[Mapping[str, str]] = None,
        line_styles: Optional[Mapping[str, object]] = None,
        grid: bool = True,
        horizontal_grid: bool = False,
        minimal_spines: bool = False,
        legend_box: bool = False,
        legend_ncols: Optional[int] = None,
        legend_cell: int = 0,
        center_last: bool = False,
        xlabel: str = "预报时效 (h)",
        units: Optional[Mapping[str, str]] = None,
        marker: Optional[str] = "o",
        panel_letters: bool = False,
        ref_line_label: Optional[str] = "参考线",
        zoom_windows: Optional[Sequence[Tuple[float, float]]] = None,
        zoom_positions: Optional[Sequence[Sequence[float]]] = None,
        zoom_linewidth: float = 1.0,
        zoom_tick_labelsize: Optional[float] = None,
        zoom_tick_steps: Optional[Mapping[str, float]] = None,
        zoom_x_tick_step: Optional[float] = None,
        dpi: Optional[int] = None,
    ):
        """一个变量一个子图、一个模型一条线。RMSE / ACC / FA 三张图共用。

        ``series_by_model`` 是 ``{模型名: DataFrame}``（index=lead_h、columns=变量）。
        字号默认按图宽查 :func:`style.font_sizes`，颜色默认走
        :func:`style.model_colors`，``palette`` / ``color_map`` / ``line_styles`` /
        各 ``*_fontsize`` 只给特殊图覆盖。``zoom_*`` 控制局部放大 inset，
        ``legend_*`` 控制图例位置与列数，``units`` 给纵轴标签补单位。

        Raises:
            ValueError: 没有可画的变量，或所有模型在该变量上都是空的。
        """
        variables = [str(v) for v in variables]
        if not variables:
            raise ValueError("variables 是空的，没东西可画")

        frames = {str(name): _numeric_frame(frame) for name, frame in series_by_model.items()}
        fallback_colors = (
            brewer_colors(palette, list(frames))
            if palette else model_colors(list(frames))
        )
        colors = {
            name: (color_map or {}).get(name, fallback_colors[name])
            for name in frames
        }
        resolved_styles = {
            name: (line_styles or {}).get(name, "-") for name in frames
        }

        ncols = max(1, min(int(ncols), len(variables)))
        nrows = int(np.ceil(len(variables) / ncols))
        width = snap_width(float(panel_size[0]) * ncols)
        sizes = font_sizes(width)
        tick_labelsize = sizes["tick"] if tick_labelsize is None else tick_labelsize
        label_fontsize = sizes["label"] if label_fontsize is None else label_fontsize
        title_fontsize = sizes["title"] if title_fontsize is None else title_fontsize
        legend_fontsize = sizes["legend"] if legend_fontsize is None else legend_fontsize
        zoom_tick_labelsize = (
            sizes["annot"] if zoom_tick_labelsize is None else zoom_tick_labelsize
        )
        figure, axes = plt.subplots(
            nrows, ncols,
            figsize=(width, float(panel_size[1]) * nrows),
            squeeze=False,
        )
        axes = axes.ravel()

        for index, variable in enumerate(variables):
            axis = axes[index]
            plotted = 0
            plotted_min, plotted_max = np.inf, -np.inf
            for name, frame in frames.items():
                if variable not in frame.columns:
                    continue
                series = frame[variable].dropna()
                series = series[np.isfinite(series)]
                if series.empty:
                    continue
                axis.plot(
                    series.index,
                    series.values,
                    color=colors[name],
                    linestyle=resolved_styles[name],
                    linewidth=linewidth,
                    marker=marker,
                    markersize=markersize,
                    markevery=markevery,
                    label=name,
                )
                plotted += 1
                plotted_min = min(plotted_min, float(series.min()))
                plotted_max = max(plotted_max, float(series.max()))
            if plotted == 0:
                raise ValueError(f"所有模型都没有变量 {variable!r} 的数据")
            if zoom_y:
                low, high = plotted_min, plotted_max
                if ref_line is not None:
                    low, high = min(low, ref_line), max(high, ref_line)
                if math.isfinite(low) and math.isfinite(high):
                    span = high - low
                    pad = span * 0.04 if span > 0 else max(abs(high) * 0.02, 0.05)
                    axis.set_ylim(low - pad, high + pad)
            if ref_line is not None:
                axis.axhline(ref_line, color=NEUTRAL["reference"], linestyle="--", linewidth=1.0, alpha=0.8)
            if panel_letters:
                # 面板字母与变量名同一字号、同一字重，整行加粗，不会一大一小。
                letter = chr(ord("a") + index)
                # 面板字母与变量名同一字号：按图宽缩放的 sizes["panel"] 在
                # 12 in 宽的报告图里会跳到 20 pt，比相邻子图的轴标签还大。
                axis.set_title(
                    rf"$\mathbf{{({letter})}}$ {variable}",
                    fontsize=title_fontsize,
                    fontweight="bold", loc="left",
                )
            else:
                axis.set_title(
                    variable, fontsize=title_fontsize, fontweight="bold",
                )
            axis.set_xlabel(xlabel, fontsize=label_fontsize)
            # 无量纲指标（Spread/RMSE、FA ratio）传的是 "-"，不写进轴标签，
            # 否则会渲染成 "Spread / RMSE (-)"。
            unit = units.get(variable) if units else None
            unit_text = "" if unit in (None, "", "-") else f" ({unit})"
            axis.set_ylabel(f"{ylabel}{unit_text}", fontsize=label_fontsize)
            if grid:
                axis.set_axisbelow(True)
                if horizontal_grid:
                    axis.grid(
                        True, axis="y", color=NEUTRAL["grid"], linestyle="--",
                        linewidth=0.5, alpha=0.7,
                    )
                    axis.grid(False, axis="x")
                else:
                    axis.grid(linewidth=0.6)
            else:
                axis.grid(False)
            if minimal_spines:
                for side in ("top", "right"):
                    axis.spines[side].set_visible(False)
            axis.tick_params(labelsize=tick_labelsize)
            if zoom_windows:
                _draw_zoom_insets(
                    axis, frames, variable, colors, resolved_styles,
                    zoom_windows,
                    zoom_positions or (
                        (0.02, 0.62, 0.36, 0.35),   # 前期放大：左上
                        (0.54, 0.02, 0.44, 0.36),   # 后期放大：右下
                    ),
                    linewidth=zoom_linewidth,
                    tick_labelsize=zoom_tick_labelsize,
                    tick_steps=zoom_tick_steps,
                    x_tick_step=zoom_x_tick_step,
                )

        # 图例优先放空出来的子图格（5 面板两列排 → (d) 正下方/ (e) 右侧），
        # 没有空格才回退到整张图底部。图例统一用 LEGEND_FRAME（只描灰色细边），
        # 即使压到曲线也能看清底下是什么。``center_last``（2-2-1）例外：最后
        # 一张子图会横跨两列居中，右上角的空格也会被它压住，图例只能退回图底。
        handles, labels = axes[0].get_legend_handles_labels()
        if ref_line is not None and ref_line_label is not None:
            handles.append(plt.Line2D([], [], color=NEUTRAL["reference"], linestyle="--", linewidth=1.0))
            labels.append(f"{ref_line_label} = {ref_line:g}")
        unused = list(axes[len(variables):])
        # ``legend_box=True`` 是**强制图底**：即使还有空格也不用，留给后续局部放大图。
        legend_axis = (
            unused[legend_cell]
            if handles and unused and not center_last and not legend_box
            else None
        )
        for axis in unused:
            axis.set_visible(axis is legend_axis)
        if legend_axis is not None:
            legend_axis.set_axis_off()   # 只借位置，不画刻度/边框
        if handles:
            if legend_ncols is None:
                # 空格里放两列（六模型 2×3）；底部图例维持每行三项。
                legend_ncols = (
                    2 if legend_axis is not None
                    else (min(3, len(labels)) if legend_box else min(len(labels), 6))
                )
            if legend_ncols > 1 and len(labels) > legend_ncols:
                # matplotlib 图例按列优先填充：直接传 [A,B,C,D,E,F] 会显示成
                # A D / B E / C F。重排成列优先的入参，视觉上才是逐行 A B / C D。
                rows = math.ceil(len(labels) / legend_ncols)
                order = [
                    row * legend_ncols + column
                    for column in range(legend_ncols)
                    for row in range(rows)
                    if row * legend_ncols + column < len(labels)
                ]
                handles = [handles[index] for index in order]
                labels = [labels[index] for index in order]
            legend_kwargs = dict(
                loc="center" if legend_axis is not None else "lower center",
                ncols=legend_ncols,
                fontsize=legend_fontsize,
                handlelength=2.0,
                columnspacing=1.6,
                labelspacing=0.6,
                borderaxespad=0.2,
                **LEGEND_FRAME,
            )
            legend = (
                legend_axis.legend(handles, labels, **legend_kwargs)
                if legend_axis is not None
                else figure.legend(handles, labels, **legend_kwargs)
            )
            legend.get_frame().set_linewidth(0.8)
        if suptitle:
            figure.suptitle(suptitle, fontsize=sizes["suptitle"])
        # 底部预留按图例行数与图高换算，别让两行/三行图例压到 x 轴标签上。
        # 只留「图例高度 + 0.2 in 余量」，不再额外垫到 12%，否则五面板图底下会空一大块。
        bottom = 0.05
        if legend_box and handles and legend_axis is None:
            rows = math.ceil(len(labels) / legend_ncols)
            reserved = rows * (legend_fontsize * 1.45) / 72.0 + 0.2
            bottom = min(
                0.4,
                max(0.07, reserved / (float(panel_size[1]) * nrows)),
            )
        figure.tight_layout(
            rect=(0, bottom, 1, 0.96 if suptitle else 1.0)
        )
        # 2-2-1：奇数个变量按两列排时，把最后一张子图从左侧移到整行中间。
        # 只有一张子图时（如 CRPS）同样居中，否则整张图会偏在左边。
        # 必须在 tight_layout 之后改，否则会被重新算掉。
        if center_last and len(variables) % ncols == 1:
            last = axes[len(variables) - 1]
            position = last.get_position()
            if nrows == 1:
                target_x0 = (1.0 - position.width) / 2
            else:
                left = axes[0].get_position().x0
                right = axes[ncols - 1].get_position().x1
                target_x0 = (left + right) / 2 - position.width / 2
            last.set_position([
                target_x0,
                position.y0,
                position.width,
                position.height,
            ])
        self._save(figure, save_path, dpi=dpi)
        return figure

    # ------------------------------------------------------------------ #
    def plot_group_bars(
        self,
        table: pd.DataFrame,
        *,
        ylabel: str,
        ref_line: Optional[float] = None,
        suptitle: Optional[str] = None,
        save_path: Optional[Path] = None,
    ):
        """一行一个分组、组里一个模型一根柱。

        「分组」是纬度带这类**同一指标、不同统计范围**的切法（分带综合相对 RMSE 就是
        这个形状）。和 :meth:`plot_model_panels` 的分工：那个是「一个分组一张子图、
        组内画随时效的曲线」，这个是「所有分组并排、组内比大小」——分带只有几个，
        并排放才看得出带间差距。

        Args:
            table: index=分组名，columns=模型名。
            ylabel: 纵轴标签。
            ref_line: 参考线（相对指标画 100）。
            suptitle: 总标题。
            save_path: PNG 落点。

        Raises:
            ValueError: 表是空的。
        """
        data = table.apply(pd.to_numeric, errors="coerce")
        if data.empty or not len(data.columns):
            raise ValueError("分组对比表是空的，没东西可画")

        names = [str(column) for column in data.columns]
        groups = [str(index) for index in data.index]
        colors = model_colors(names)

        positions = np.arange(len(groups), dtype=float)
        bar_width = 0.8 / max(len(names), 1)
        width = snap_width(max(6.0, 1.8 * len(groups)))
        sizes = font_sizes(width)
        figure, axis = plt.subplots(figsize=(width, 4.6))
        for offset, name in enumerate(names):
            axis.bar(
                positions + offset * bar_width - 0.4 + bar_width / 2,
                data[name].values,
                width=bar_width,
                color=colors[name],
                label=name,
            )
        if ref_line is not None:
            axis.axhline(ref_line, color=NEUTRAL["reference"], linestyle="--", linewidth=1.0,
                         label=f"参考线 = {ref_line:g}")
        axis.set_xticks(positions)
        axis.set_xticklabels(groups, fontsize=sizes["tick"])
        axis.set_ylabel(ylabel, fontsize=sizes["label"])
        axis.tick_params(axis="y", labelsize=sizes["tick"])
        axis.grid(axis="y", linewidth=0.6)
        legend = axis.legend(fontsize=sizes["legend"], **LEGEND_FRAME)
        legend.get_frame().set_linewidth(0.8)
        if suptitle:
            axis.set_title(suptitle, fontsize=sizes["suptitle"])
        figure.tight_layout()
        self._save(figure, save_path)
        return figure

    # ------------------------------------------------------------------ #
    def plot_segment_dots(
        self,
        table: pd.DataFrame,
        *,
        xlabel: str = "Relative RMSE Index",
        ylabel: str = "Lead-time band",
        ref_line: float = 100.0,
        save_path: Optional[Path] = None,
    ):
        """分时效段点图（dot plot / 棒棒糖图）：行=时效段，点=模型。

        每个时效段一行，行内各模型按固定顺序上下错开，避免数值接近时圆点
        互相盖住；每个点有一根从参考线伸出的细杆，读者只需看谁最靠左。
        比堆叠柱状图省墨，也不需要给每个模型单开一条曲线。

        Args:
            table: index=时效段名（按显示顺序从上到下），columns=模型名。
            xlabel / ylabel: 轴标签（报告图统一英文）。
            ref_line: 参考线位置（相对指标画 100）；只画虚线，不进图例。
            save_path: SVG 落点。

        Raises:
            ValueError: 表是空的。
        """
        data = table.apply(pd.to_numeric, errors="coerce")
        if data.empty or not len(data.columns):
            raise ValueError("分时效段表是空的，没东西可画")

        names = [str(column) for column in data.columns]
        bands = [str(index) for index in data.index]
        colors = model_colors(names)
        width = snap_width(7.2)
        sizes = font_sizes(width)
        figure, axis = plt.subplots(figsize=(width, 3.9))

        positions = np.arange(len(bands), dtype=float)
        offsets = (
            np.linspace(-0.18, 0.18, len(names)) if len(names) > 1
            else np.zeros(1)
        )
        values = data.to_numpy(dtype=float)
        finite = values[np.isfinite(values)]
        span = float(finite.max() - finite.min()) if finite.size else 1.0
        pad = max(span * 0.16, 1.2)

        for column, name in enumerate(names):
            ys = positions + offsets[column]
            xs = data[name].to_numpy(dtype=float)
            for x, y in zip(xs, ys):
                if not np.isfinite(x):
                    continue
                axis.plot(
                    [ref_line, x], [y, y], color=colors[name],
                    linewidth=1.8, alpha=0.35, solid_capstyle="round", zorder=2,
                )
            axis.scatter(
                xs, ys, s=72, color=colors[name], edgecolor="white",
                linewidth=1.2, zorder=3, label=name,
            )
            # 数值标在点的外侧：低于参考线向左、高于向右，避免压住杆。
            for x, y in zip(xs, ys):
                if not np.isfinite(x):
                    continue
                axis.annotate(
                    f"{x:.2f}", (x, y),
                    xytext=(4 if x >= ref_line else -4, 0),
                    textcoords="offset points",
                    ha="left" if x >= ref_line else "right",
                    va="center", fontsize=sizes["annot"] + 1,
                    color=NEUTRAL["text"], zorder=4,
                )

        axis.axvline(
            ref_line, color=NEUTRAL["reference"], linestyle="--",
            linewidth=1.0, zorder=1,
        )
        axis.set_yticks(positions)
        axis.set_yticklabels(bands, fontsize=sizes["tick"])
        axis.set_ylim(len(bands) - 0.5, -0.5)
        axis.set_xlim(float(finite.min() - pad), float(finite.max() + pad))
        axis.set_xlabel(xlabel, fontsize=sizes["label"])
        axis.set_ylabel(ylabel, fontsize=sizes["label"])
        axis.tick_params(axis="x", labelsize=sizes["tick"])
        axis.tick_params(axis="y", length=0)
        for side in ("top", "right"):
            axis.spines[side].set_visible(False)

        handles, labels = axis.get_legend_handles_labels()
        legend = figure.legend(
            handles, labels,
            loc="lower center", bbox_to_anchor=(0.5, 0.005),
            ncols=len(handles), fontsize=sizes["legend"], frameon=False,
            handlelength=1.6, columnspacing=1.6, handletextpad=0.5,
        )
        legend.get_frame().set_linewidth(0.8)
        figure.tight_layout(rect=(0, 0.10, 1, 1))
        self._save(figure, save_path)
        return figure

    # ------------------------------------------------------------------ #
    def plot_model_spectrum(
        self,
        spectra_by_model: Mapping[str, pd.DataFrame],
        variable: str,
        *,
        suptitle: Optional[str] = None,
        save_path: Optional[Path] = None,
    ):
        """双对数坐标：多模型的预测谱 + 一条共享的观测谱。

        报告图统一用英文轴标签；``suptitle=None`` 时**不画标题**（图号与说明
        交给报告正文的图注），只有传了才在顶部写一行。

        Args:
            spectra_by_model: ``{模型名: DataFrame}``，index=波数 k，
                columns 含 ``pred`` 与 ``obs``。``obs`` 是所有模型共用的实况谱，
                只画第一条非空的（同一个变量、同一批日期，观测谱本就该一致）。
            variable: 变量名；``suptitle=None`` 时不再用它拼默认标题。
            suptitle: 总标题；None = 不画。
            save_path: PNG 落点。

        Raises:
            ValueError: 没有任何模型的谱数据，或所有模型都只有 k=0。
        """
        width = snap_width(9)
        sizes = font_sizes(width)
        figure, axis = plt.subplots(figsize=(width, 6))
        colors = model_colors(list(spectra_by_model))
        observation = None
        plotted = 0

        for name, raw in spectra_by_model.items():
            frame = _numeric_frame(raw)
            if "pred" not in frame.columns:
                raise ValueError(f"{name} 的谱表没有 pred 列；现有列: {list(frame.columns)}")
            curve = frame["pred"].dropna()
            curve = curve[curve.index > 0]  # k=0 的功率恒为 0，log 轴画不了
            curve = curve[curve > 0]
            if curve.empty:
                continue
            axis.plot(curve.index, curve.values, color=colors[name], linewidth=1.6, label=name)
            plotted += 1
            if observation is None and "obs" in frame.columns:
                reference = frame["obs"].dropna()
                reference = reference[(reference.index > 0) & (reference > 0)]
                if not reference.empty:
                    observation = reference

        if plotted == 0:
            raise ValueError(f"没有任何模型给出 {variable} 的可画谱线（k>0 且功率>0）")

        if observation is not None:
            axis.plot(
                observation.index, observation.values,
                color=NEUTRAL["observation"], linestyle="--", linewidth=1.4,
                label="Observation",
            )

        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_xlabel("Zonal wavenumber k", fontsize=sizes["label"])
        axis.set_ylabel("Power (log scale)", fontsize=sizes["label"])
        axis.tick_params(labelsize=sizes["tick"])
        axis.grid(which="both", linewidth=0.6)
        axis.legend(loc="best", fontsize=sizes["legend"])
        if suptitle:
            axis.set_title(suptitle, fontsize=sizes["suptitle"])
        figure.tight_layout()
        self._save(figure, save_path)
        return figure

    # ------------------------------------------------------------------ #
    def plot_capability_radar(
        self,
        raw: Mapping[str, Mapping[str, float]],
        axes: Sequence[str],
        higher_is_better: Mapping[str, bool],
        *,
        suptitle: Optional[str] = None,
        save_path: Optional[Path] = None,
    ) -> Dict[str, Dict[str, float]]:
        """多模型能力雷达图；返回各模型的**归一化**得分。

        ``raw`` 是 ``{模型名: {轴名: 原始值}}``，``axes`` 给轴序。

        归一化是**批内 min-max**：每个模型在每根轴上得 ``1 = 本批最好``、
        ``0 = 本批最差``，方向由 ``higher_is_better`` 定（缺省按"越低越好"，
        与 RMSE / 偏差这类指标的惯例一致）。这是**相对分**——两个模型时
        必然是 1 和 0，不代表绝对能力，报告里必须把这句话写出来。

        两个退化情形（全批打平给满格、缺值整根剔除）见 :func:`radar_scores`，
        归一化只有那一处实现。
        """
        names = [str(name) for name in raw]
        scores = radar_scores(raw, axes, higher_is_better)
        # 归一化后还留在结果里的，就是真正画出来的轴（缺值的已被剔掉）
        kept = [str(axis) for axis in axes if str(axis) in scores[names[0]]] if names else []

        if not kept:
            return scores

        count = len(kept)
        angles = [index / count * 2 * math.pi for index in range(count)]
        closed = angles + angles[:1]
        colors = model_colors(names)

        width = snap_width(7.2)
        sizes = font_sizes(width)
        figure = plt.figure(figsize=(width, 7.2))
        axis = figure.add_subplot(111, polar=True)
        axis.set_theta_offset(math.pi / 2)      # 第一根轴摆在正上方
        axis.set_theta_direction(-1)            # 顺时针排，读起来跟表同序
        axis.set_xticks(angles)
        axis.set_xticklabels(kept, fontsize=sizes["label"])
        # 轴名往外挪一点：极坐标默认贴着外圈画，五六个字的中文标签会压到圆周上
        axis.tick_params(axis="x", pad=10)
        axis.set_ylim(0.0, 1.0)
        axis.set_yticks([0.25, 0.5, 0.75, 1.0])
        axis.set_yticklabels(["0.25", "0.50", "0.75", "1.00"], fontsize=sizes["annot"],
                             color=NEUTRAL["muted"])
        axis.grid(linewidth=0.6)

        for name in names:
            ring = [scores[name][each] for each in kept]
            axis.plot(closed, ring + ring[:1], color=colors[name], linewidth=1.8,
                      label=name)
            axis.fill(closed, ring + ring[:1], color=colors[name], alpha=0.12)

        legend = axis.legend(
            loc="upper right", bbox_to_anchor=(1.22, 1.08),
            fontsize=sizes["legend"], **LEGEND_FRAME,
        )
        legend.get_frame().set_linewidth(0.8)
        if suptitle:
            # 顶部那根轴的名字就压在 suptitle 正下方，不留出这段空隙会叠在一起
            figure.subplots_adjust(top=0.86)
            figure.suptitle(suptitle, fontsize=sizes["suptitle"])
        self._save(figure, save_path)
        # 返回**得分**而不是 figure：调用方要靠它写图注与判读句，
        # 图本身落盘就够了（其余几个 plot_* 返回 figure 但调用方一律不用）。
        return scores
