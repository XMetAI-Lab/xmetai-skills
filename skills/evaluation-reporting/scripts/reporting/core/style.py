# -*- coding: utf-8 -*-
"""全仓唯一的配色与中文字体入口。

四类色池，各管一种编码任务，不要混用：

* **模型身份**：:data:`MODEL_PALETTE` / :data:`MODEL_COLOR_MAP` /
  :func:`model_colors`。已知模型永远拿同一个颜色，不随命令行里
  ``--model`` 的先后变；未知模型按名字排序取剩下的槽位。
* **降水等级**（有序）：:data:`GRADE_COLORS` / :func:`grade_colors`。
  等级是"从弱到强"的序，用一条连续色带，而不是彩虹式分类色。
* **连续量**：:data:`CMAP_SEQ`（单向）；:data:`CMAP_DIV`（以 0 为中心）。
* **中性色**：:data:`NEUTRAL`。观测、基准、参考线、网格、边框、文字都从这里取，
  图上不再出现零散的 ``"gray"`` / ``"#333333"``。

色相骨架直接取 NPG（Nature Publishing Group）的常用色系：饱和度和明度都比
降饱和的莫兰迪色高一档，屏幕上更醒目；相邻两色的色相拉得够开，灰度打印下
仍可分辨，又不会像 Set1 那样在浅底上互相抢。
"""

from __future__ import annotations

import re
from typing import Dict, Iterable, Optional, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# --------------------------------------------------------------------- 模型色

#: 模型色池：NPG（Nature Publishing Group）配色。前 8 色用于线条/散点/箱体，
#: 后 2 色对比度低，只做面积填充。整体比旧版莫兰迪色亮一档，但相邻色相仍拉得开。
MODEL_PALETTE = (
    "#E64B35",  # 1 砖红 NPG red
    "#4DBBD5",  # 2 青蓝 cyan
    "#00A087",  # 3 蓝绿 teal
    "#3C5488",  # 4 藏蓝 navy
    "#F39B7F",  # 5 鲑粉 salmon
    "#8491B4",  # 6 雾蓝 blue-gray
    "#DC0000",  # 7 亮红 crimson
    "#7E6148",  # 8 棕 brown
    "#91D1C2",  # 9 薄荷 mint（填充）
    "#B09C85",  # 10 卡其 tan（填充）
)

MODEL_LINE_COLORS = MODEL_PALETTE[:8]
MODEL_FILL_COLORS = MODEL_PALETTE[8:]

#: 已知模型 → 固定颜色。加新模型在这里加一行，不要靠调用顺序碰运气。
MODEL_COLOR_MAP: Dict[str, str] = {
    "FGVP": "#E64B35",
    "FuXi": "#4DBBD5",
    "AIFS": "#3C5488",
    "FGVP-SPA": "#00A087",
    "FGVP-SPW": "#8491B4",
    "S2B": "#F39B7F",
    "FengQing": "#91D1C2",
    "Pangu": "#7E6148",
}

#: 对照/旧版不是一个"模型"，而是同一个模型的另一种状态——统一走中性灰，
#: 让读图的人一眼看出它是对照组，而不是又冒出一个新模型。
_CONTROL_TOKENS = frozenset({"ctrl", "control", "old"})

# --------------------------------------------------------------------- 中性色

NEUTRAL: Dict[str, str] = {
    "observation": "#2B2B2B",  # 观测/真值，深灰虚线
    "baseline": "#8A8A8A",     # 基准/对照
    "reference": "#8A8A8A",    # 参考线，虚线/点线
    "grid": "#D9D9D9",         # 网格，0.6pt
    # 坐标轴的一切（边框、刻度线、刻度数字、轴标题）统一纯黑；灰边框在
    # 大量文字并排时会让整张图发虚，也让打印稿显得没对齐。
    "spine": "#000000",        # 边框，1.0pt
    "text": "#000000",         # 正文/标题/轴标签
    "muted": "#000000",        # 刻度数字（不再弱化成灰色）
    "panel": "#F2F2F2",        # 图例底、浅色衬底
}

#: 局部放大框：极浅中性灰底 + 灰色虚线描边。不用彩色高亮——琥珀/橙色太
#: 抢眼，而且容易和 S2B 的橙线混淆。
ZOOM_HIGHLIGHT: Dict[str, str] = {
    "face": "#BFBFBF",
    "edge": "#7F7F7F",
}

#: 图例边框：**只描边、不填底**。填一层浅灰底看起来像一块灰色方块/阴影，
#: 把图压得发闷；留一道 0.8 pt 的灰边就够把图例和曲线分开。所有图例统一走
#: 这一套，别再各处手写。
LEGEND_FRAME: Dict[str, object] = {
    "frameon": True,
    "facecolor": "none",
    "edgecolor": "#B8B8B8",
    "framealpha": 1.0,
}

# --------------------------------------------------------------------- 等级色

#: 降水等级 → 颜色。**有序**：越强越深，不是分类色循环。
GRADE_COLORS: Dict[str, str] = {
    "≥0.1": "#D8CFC0",
    "≥4": "#C4B39C",
    "≥10": "#A98F7A",
    "≥13": "#8A7B72",
    "≥25": "#6E7F7B",
    "≥50": "#567673",
    "≥100": "#3F6565",
    "≥250": "#2A5052",
}

GRADE_RAMP = tuple(GRADE_COLORS.values())

# --------------------------------------------------------------------- 连续色带

#: 单向连续量（RMSE、倍数、谱密度等）：米白 → 砂 → 褐 → 墨绿。
CMAP_SEQ = LinearSegmentedColormap.from_list(
    "xmetai_seq", ["#F4F1EA", "#D8CFC0", "#A98F7A", "#6E7F7B", "#2A5052"]
)

#: 发散量（差值场、正负距平）：墨蓝 ← 米白 → 陶土红，0 落在中点的米白上。
#: LUT 取 257 档（奇数），0.5 才精确落在中间那格上。
CMAP_DIV = LinearSegmentedColormap.from_list(
    "xmetai_div", ["#3D5A80", "#F4F1EA", "#B85C4E"], N=257
)

#: 相对表现（热力图专用）：深蓝 0 = 最差 → 浅蓝 0.25 → 中灰 0.50 → 浅红
#: 0.75 → 深红 1 = 最优。比默认 ``RdBu_r`` 更亮、层次更清楚；灰中点保证
#: “不好不坏”一眼可辨。
CMAP_SKILL = LinearSegmentedColormap.from_list(
    "xmetai_skill",
    ["#3B4CC0", "#8DB0FE", "#DDDDDD", "#F4987A", "#B40426"],
)

# --------------------------------------------------------------------- 排版

#: Nature 双栏宽度（183 mm）。所有字号都以这个宽度为基准定义。
BASE_WIDTH = 7.2

#: 任何文字都不小于 5 pt——Nature 排版指南的下限，再小印刷就糊了。
MIN_PT = 5.0

#: 文字角色 → 183 mm 双栏下的字号（pt）。绘图代码只引用角色，不写字面量。
#: 2026-10-20 按用户反馈二次上调：屏幕报告图要按 12/10/10 这组绝对字号看。
TYPE_SCALE: Dict[str, float] = {
    "tick": 10.0,      # 坐标轴刻度
    "label": 12.0,     # 轴标签
    "title": 12.0,     # 子图标题 / 变量名（加粗）
    "legend": 10.0,    # 图例
    "annot": 8.0,      # 图内注释、格内标数、inset/色标刻度
    "panel": 12.0,     # 面板字母 (a)(b)，粗体
    "suptitle": 12.0,  # 总标题
}

#: 图宽档位（英寸）：Nature 单栏 89 mm、双栏 183 mm，其余是屏幕报告档。
FIGURE_WIDTHS: Dict[str, float] = {
    "single": 3.5,
    "onehalf": 4.7,
    "double": 7.2,
    "wide": 12.0,
    "full": 16.0,
}

# --------------------------------------------------------------------- 字体

_CJK_FONTS = (
    "Noto Sans CJK SC",
    "Source Han Sans SC",
    "Source Han Sans CN",
    "WenQuanYi Zen Hei",
    "WenQuanYi Micro Hei",
    "Microsoft YaHei",
    "SimHei",
    "PingFang SC",
    "Heiti TC",
    "Arial Unicode MS",
)


def setup_chinese_font() -> Optional[str]:
    """选择可用的中文字体；返回选中的字体名（没有则 None）。"""
    available = {font.name for font in fm.fontManager.ttflist}
    for name in _CJK_FONTS:
        if name in available:
            plt.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
            plt.rcParams["font.family"] = "sans-serif"
            plt.rcParams["axes.unicode_minus"] = False
            return name
    plt.rcParams["axes.unicode_minus"] = False
    return None


def apply_style() -> Optional[str]:
    """套上中性色、中文字体与 183 mm 基准字号，返回选中的字体名。

    字号只是**兜底**：没显式写字号的小字（色标标题、极少数漏标的注释）按
    双栏基准走；正常绘图都应先 ``sizes = font_sizes(figsize[0])`` 再按角色传。
    必须在 ``plt.style.use(...)`` **之后**调用，否则样式表会把颜色盖回去。
    """
    font = setup_chinese_font()
    plt.rcParams.update(
        {
            "axes.edgecolor": NEUTRAL["spine"],
            "axes.linewidth": 1.0,
            "axes.labelcolor": NEUTRAL["text"],
            "axes.axisbelow": True,
            "xtick.color": NEUTRAL["muted"],
            "ytick.color": NEUTRAL["muted"],
            "xtick.major.width": 1.0,
            "ytick.major.width": 1.0,
            "text.color": NEUTRAL["text"],
            "grid.color": NEUTRAL["grid"],
            "grid.linewidth": 0.6,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "font.size": TYPE_SCALE["label"],
            "axes.titlesize": TYPE_SCALE["title"],
            "axes.labelsize": TYPE_SCALE["label"],
            "xtick.labelsize": TYPE_SCALE["tick"],
            "ytick.labelsize": TYPE_SCALE["tick"],
            "legend.fontsize": TYPE_SCALE["legend"],
            "figure.titlesize": TYPE_SCALE["suptitle"],
        }
    )
    return font


def font_sizes(width_in: float) -> Dict[str, float]:
    """按图宽返回各文字角色的字号（pt），宽图字号等比放大。

    ``width_in <= 183 mm`` 时直接用 :data:`TYPE_SCALE`；超过双栏宽度的屏幕
    报告图按 ``宽 / 7.2`` 放大，缩到 183 mm 印刷时视觉字号仍与 Nature 一致。
    结果四舍五入到整数 pt，并卡在 :data:`MIN_PT` 之上。

    Args:
        width_in: 图的物理宽度（英寸），传 ``figsize[0]`` 或
            ``fig.get_size_inches()[0]``。

    Returns:
        ``{"tick"/"label"/"title"/"legend"/"annot"/"panel"/"suptitle": pt}``。
    """
    try:
        width = float(width_in)
    except (TypeError, ValueError):
        width = BASE_WIDTH
    if not (width > 0):
        width = BASE_WIDTH
    scale = max(1.0, width / BASE_WIDTH)
    return {
        role: max(MIN_PT, float(round(size * scale)))
        for role, size in TYPE_SCALE.items()
    }


def snap_width(width_in: float) -> float:
    """把图宽归到 :data:`FIGURE_WIDTHS` 的档位：取不小于当前值的最小档。

    宽度**只升不降**，现有版式不会因为归一化被挤窄；超过 16 in 的动态宽度
    （类别多到一屏放不下）保持原值，字号继续按 :func:`font_sizes` 放大。
    """
    try:
        width = float(width_in)
    except (TypeError, ValueError):
        return FIGURE_WIDTHS["double"]
    for candidate in FIGURE_WIDTHS.values():
        if width <= candidate + 1e-9:
            return candidate
    return width


# --------------------------------------------------------------------- 取色函数


def _is_control(name: str) -> bool:
    """``FGVP-ctrl`` / ``FGVP_old`` 这类对照名（按 ``-``/``_`` 切词再比）。"""
    return any(token in _CONTROL_TOKENS for token in re.split(r"[-_\s]+", name.lower()))


def model_colors(names: Iterable[str]) -> Dict[str, str]:
    """``{模型名: 颜色}``，**与传入顺序无关**。

    已知模型直接查 :data:`MODEL_COLOR_MAP`；对照/旧版统一走中性灰；剩下的
    未知模型**按名字排序**依次填当前图里没被已知模型占用的色槽。所以交换
    ``--model`` 的先后不会换色，加一个模型也只会让未知的那几个顺延。

    模型的"主/次"不要靠颜色区分——要突出主模型就加粗它的线宽。
    """
    ordered = list(dict.fromkeys(str(name) for name in names))
    colors: Dict[str, str] = {}
    used = set()
    unknown = []

    for name in ordered:
        if _is_control(name):
            colors[name] = NEUTRAL["baseline"]
        elif name in MODEL_COLOR_MAP:
            colors[name] = MODEL_COLOR_MAP[name]
            used.add(MODEL_COLOR_MAP[name])
        else:
            unknown.append(name)

    available = [color for color in MODEL_LINE_COLORS if color not in used]
    pool = available or list(MODEL_LINE_COLORS)
    for index, name in enumerate(sorted(unknown)):
        colors[name] = pool[index % len(pool)]
    return {name: colors[name] for name in ordered}


def grade_colors(grades: Sequence[str]) -> Dict[str, str]:
    """``{等级标签: 颜色}``；不认识的等级按出现次序沿有序色带顺延。"""
    ordered = [str(grade) for grade in grades]
    return {
        grade: GRADE_COLORS.get(grade, GRADE_RAMP[index % len(GRADE_RAMP)])
        for index, grade in enumerate(ordered)
    }
