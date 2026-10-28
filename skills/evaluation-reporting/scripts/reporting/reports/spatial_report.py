#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""空间分布：月平均宽表 CSV → 模型 / ERA5 / 偏差（/ 离散度）图与 §4.6 正文。

宽表由 ``skills/evaluation-reporting/scripts/extract_spatial_distribution*.py`` 产出::

    lat, lon,
    {var}_{YYYYMM}_h{lead},
    {var}_spread_{YYYYMM}_h{lead},      # 只有集合版有
    {var}_era5_{YYYYMM}_h{lead}

网格是规则经纬网格，``lat`` 北 → 南、``lon`` 0–360 升序、行主序。确定性 CSV
是集合 CSV 的子集（没有 spread 列），所以一个 loader 吃两种。

**离散度口径**：先对每个成员求月平均场，再对成员求标准差（``ddof=0``）。
不是「逐时次集合 std 再月平均」——那一行与上面的月平均场不是同一层级。

本模块不 import ``det_report``（它由两份报告 import，避免循环）。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

from reporting.core.common import bullet_lines, table_lines

# ---------------------------------------------------------------- 契约
COLUMN_RE = re.compile(
    r"^(?P<var>[A-Za-z][A-Za-z0-9]*?)(?:_(?P<kind>spread|era5))?"
    r"_(?P<tag>\d{6})_h(?P<lead>\d{3})$"
)
VARIABLE_ORDER = ("z500", "ws850")
PANEL_STYLE = {
    "z500": ("gpm", "viridis"),
    "ws850": ("m/s", "YlGnBu"),
}
SPREAD_CMAP = "YlOrRd"
GRAVITY = 9.80665
Z500_PHYS_THRESHOLD = 1e4

SPATIAL_ABSENT = (
    "**本批未出。** 未提供 `--spatial-csv`"
    "（空间分布图需要独立的月平均宽表 CSV）。"
)


@dataclass(frozen=True)
class SpatialData:
    """一张宽表拆出来的全部场。``fields`` 键 = (变量, 月份, 时效, 种类)。"""

    lat: np.ndarray                 # 北 → 南
    lon: np.ndarray                 # 0–360 升序
    fields: Dict[Tuple[str, str, int, str], np.ndarray]
    variables: Tuple[str, ...]
    months: Tuple[Tuple[str, str], ...]   # (202507, 2025-07)
    leads: Tuple[int, ...]
    has_spread: bool

    @property
    def figure_count(self) -> int:
        """一变量 × 一月一张图。"""
        return len(self.variables) * len(self.months)


@dataclass(frozen=True)
class SpatialPayload:
    """报告渲染需要的全部：数据 + 图例名 + 图文件映射。"""

    data: SpatialData
    model: str
    mode: str                       # "det" / "ens"
    figures: Mapping[str, str]      # spatial_{var}_{tag} → 相对文件名


def spatial_figure_key(variable: str, tag: str) -> str:
    return f"spatial_{variable}_{tag}"


def _month_label(tag: str) -> str:
    return f"{tag[:4]}-{tag[4:]}"


# ---------------------------------------------------------------- 读表


def load_spatial_csv(path: Path) -> SpatialData:
    """读宽表并做完整性校验；缺列、网格错序都直接报错，不静默补。"""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"空间分布 CSV 不存在：{path}")
    frame = pd.read_csv(path)
    for name in ("lat", "lon"):
        if name not in frame.columns:
            raise ValueError(f"空间分布 CSV 缺少 {name!r} 列：{path}")

    lat = np.sort(frame["lat"].unique())[::-1]
    lon = np.sort(frame["lon"].unique())
    if len(frame) != lat.size * lon.size:
        raise ValueError(
            f"空间分布 CSV 不是完整规则网格：{len(frame)} 行，"
            f"应为 {lat.size}×{lon.size}={lat.size * lon.size}"
        )
    if not (
        np.allclose(frame["lat"].to_numpy(dtype=float), np.repeat(lat, lon.size))
        and np.allclose(frame["lon"].to_numpy(dtype=float), np.tile(lon, lat.size))
    ):
        raise ValueError(
            "空间分布 CSV 的行序必须是 lat 北→南、lon 0–360 升序的行主序网格"
        )

    columns: Dict[Tuple[str, str, int, str], str] = {}
    unknown = []
    for column in frame.columns:
        if column in ("lat", "lon"):
            continue
        matched = COLUMN_RE.match(str(column))
        if matched is None:
            unknown.append(str(column))
            continue
        columns[(
            matched.group("var"),
            matched.group("tag"),
            int(matched.group("lead")),
            matched.group("kind") or "model",
        )] = str(column)
    if unknown:
        raise ValueError(f"空间分布 CSV 有无法识别的列：{unknown}")

    combos = sorted({(var, tag, lead) for var, tag, lead, _kind in columns})
    missing = [
        f"{var}_{tag}_h{lead:03d}({kind})"
        for var, tag, lead in combos
        for kind in ("model", "era5")
        if (var, tag, lead, kind) not in columns
    ]
    if missing:
        raise ValueError(f"空间分布 CSV 缺少列：{missing}")

    has_spread = any(kind == "spread" for *_rest, kind in columns)
    if has_spread:
        missing = [
            f"{var}_spread_{tag}_h{lead:03d}"
            for var, tag, lead in combos
            if (var, tag, lead, "spread") not in columns
        ]
        if missing:
            raise ValueError(f"空间分布 CSV 的 spread 列不完整，缺：{missing}")

    present = sorted({var for var, _tag, _lead in combos})
    variables = tuple([v for v in VARIABLE_ORDER if v in present]
                      + [v for v in present if v not in VARIABLE_ORDER])
    months = tuple((tag, _month_label(tag)) for tag in sorted({t for _v, t, _l in combos}))
    leads = tuple(sorted({lead for _v, _t, lead in combos}))

    fields = {
        key: np.asarray(frame[column], dtype=float).reshape(lat.size, lon.size)
        for key, column in columns.items()
    }
    if "z500" in variables:
        stock = np.nanmedian(np.abs(fields[("z500", months[0][0], leads[0], "model")]))
        if stock > Z500_PHYS_THRESHOLD:
            for key in list(fields):
                if key[0] == "z500":
                    fields[key] = fields[key] / GRAVITY

    return SpatialData(
        lat=lat, lon=lon, fields=fields, variables=variables,
        months=months, leads=leads, has_spread=has_spread,
    )


# ---------------------------------------------------------------- 出图


def _range(values: np.ndarray, variable: str) -> Tuple[float, float]:
    if variable == "z500":
        low = float(np.nanmin(values))
        high = float(np.nanmax(values))
        return np.floor(low / 100.0) * 100.0, np.ceil(high / 100.0) * 100.0
    # 风速强偏，色标顶端取 p99.5，否则 99% 的场压成一片浅色
    return 0.0, float(np.ceil(np.nanpercentile(values, 99.5)))


def _symmetric_limit(values: np.ndarray, variable: str) -> float:
    step = 5.0 if variable == "z500" else 1.0
    high = float(np.nanpercentile(np.abs(values), 99.5))
    return max(step, float(np.ceil(high / step) * step))


def _upper_limit(values: np.ndarray, variable: str) -> float:
    step = 5.0 if variable == "z500" else 1.0
    high = float(np.nanpercentile(values, 99.5))
    return max(step, float(np.ceil(high / step) * step))


def _safe_name(model: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", model.lower()).strip("_") or "model"


def _rows(data: SpatialData, model: str, mode: str) -> List[Tuple[str, str]]:
    if mode == "auto":
        mode = "ens" if data.has_spread else "det"
    if mode not in ("det", "ens"):
        raise ValueError(f"mode 只能是 auto / det / ens，收到 {mode!r}")
    if mode == "ens" and not data.has_spread:
        raise ValueError(
            "集合空间分布 CSV 缺少 spread 列；请用 "
            "extract_spatial_distribution_ens.py 重新抽取"
        )
    if mode == "ens":
        return [
            ("model", f"{model} ensemble mean"),
            ("era5", "ERA5"),
            ("diff", f"{model} − ERA5"),
            ("spread", "Spread"),
        ]
    return [("model", model), ("era5", "ERA5"), ("diff", f"{model} − ERA5")]


def _draw_month(data, lat, lon, variable, unit, cmap, tag, label, rows,
                low, high, diff_limit, spread_limit, out_path):
    figure, axes = plt.subplots(
        len(rows), len(data.leads), figsize=(15.0, 9.6 if len(rows) == 3 else 12.6),
        sharex=True, sharey=True, squeeze=False,
    )
    lon_wrap = np.append(lon, lon[0] + 360.0)
    absolute_norm = Normalize(vmin=low, vmax=high)
    diff_norm = Normalize(vmin=-diff_limit, vmax=diff_limit)
    spread_norm = Normalize(vmin=0.0, vmax=spread_limit)
    for row, (kind, row_label) in enumerate(rows):
        for column, lead in enumerate(data.leads):
            axis = axes[row, column]
            if kind == "diff":
                field = (data.fields[(variable, tag, lead, "model")]
                         - data.fields[(variable, tag, lead, "era5")])
            else:
                field = data.fields[(variable, tag, lead, kind)]
            field_wrap = np.concatenate([field, field[:, :1]], axis=1)
            if kind == "diff":
                norm, colors = diff_norm, "RdBu_r"
            elif kind == "spread":
                norm, colors = spread_norm, SPREAD_CMAP
            else:
                norm, colors = absolute_norm, cmap
            # rasterized：单张 SVG 从 100 MB 压到 ~1 MB
            axis.pcolormesh(
                lon_wrap, lat, field_wrap, cmap=colors, norm=norm,
                shading="auto", rasterized=True,
            )
            axis.set_xlim(0.0, 360.0)
            axis.set_ylim(-90.0, 90.0)
            axis.set_xticks([0, 90, 180, 270, 360])
            axis.set_yticks([-90, -45, 0, 45, 90])
            axis.tick_params(labelsize=11, length=2.5)
            for spine in axis.spines.values():
                spine.set_linewidth(0.6)
            if row == 0:
                axis.set_title(f"({chr(ord('a') + column)}) +{lead} h", fontsize=14)
            if column == 0:
                axis.text(-0.07, 0.5, row_label, transform=axis.transAxes,
                          rotation=90, ha="right", va="center", fontsize=14)

    if len(rows) == 3:
        figure.subplots_adjust(left=0.07, right=0.985, top=0.925, bottom=0.13,
                               wspace=0.06, hspace=0.08)
        bar_axis = figure.add_axes([0.16, 0.058, 0.50, 0.020])
        column_width = (0.985 - 0.07) / (len(data.leads) + (len(data.leads) - 1) * 0.06)
        column_center = 0.07 + (
            len(data.leads) - 1 + (len(data.leads) - 1) * 0.06 + 0.5
        ) * column_width
        diff_axis = figure.add_axes([column_center - 0.08, 0.058, 0.16, 0.020])
        spread_axis = None
    else:
        figure.subplots_adjust(left=0.07, right=0.985, top=0.945, bottom=0.145,
                               wspace=0.06, hspace=0.08)
        bar_axis = figure.add_axes([0.075, 0.055, 0.25, 0.018])
        diff_axis = figure.add_axes([0.375, 0.055, 0.25, 0.018])
        spread_axis = figure.add_axes([0.675, 0.055, 0.25, 0.018])

    bar = figure.colorbar(
        ScalarMappable(norm=absolute_norm, cmap=cmap), cax=bar_axis,
        orientation="horizontal",
        extend="max" if variable == "ws850" else "neither",
    )
    bar.ax.tick_params(labelsize=11)
    bar.set_label(f"{variable} / {unit}", fontsize=12)
    diff_bar = figure.colorbar(
        ScalarMappable(norm=diff_norm, cmap="RdBu_r"), cax=diff_axis,
        orientation="horizontal",
    )
    diff_bar.ax.tick_params(labelsize=11)
    diff_bar.set_label(f"{rows[2][1]} / {unit}", fontsize=12)
    if spread_axis is not None:
        spread_bar = figure.colorbar(
            ScalarMappable(norm=spread_norm, cmap=SPREAD_CMAP), cax=spread_axis,
            orientation="horizontal",
        )
        spread_bar.ax.tick_params(labelsize=11)
        spread_bar.set_label(f"Spread / {unit}", fontsize=12)

    figure.suptitle(f"{variable} {label} global", fontsize=15, y=0.975)
    figure.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def render_spatial(data: SpatialData, out_dir: Path, *, model: str,
                   mode: str = "auto") -> Dict[str, Path]:
    """按 CSV 里发现的变量 × 月份逐张出 SVG，返回 图键 → 路径。"""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = _rows(data, model, mode)
    with_spread = len(rows) == 4
    paths: Dict[str, Path] = {}
    for variable in data.variables:
        unit, cmap = PANEL_STYLE.get(variable, ("", "viridis"))
        absolute = np.concatenate([
            data.fields[(variable, tag, lead, kind)].ravel()
            for tag, _label in data.months
            for lead in data.leads
            for kind in ("model", "era5")
        ])
        diffs = np.concatenate([
            (data.fields[(variable, tag, lead, "model")]
             - data.fields[(variable, tag, lead, "era5")]).ravel()
            for tag, _label in data.months
            for lead in data.leads
        ])
        low, high = _range(absolute, variable)
        diff_limit = _symmetric_limit(diffs, variable)
        if with_spread:
            spread_limit = _upper_limit(np.concatenate([
                data.fields[(variable, tag, lead, "spread")].ravel()
                for tag, _label in data.months
                for lead in data.leads
            ]), variable)
        else:
            spread_limit = 1.0
        for tag, label in data.months:
            path = out_dir / f"{_safe_name(model)}_{variable}_{tag}_vs_era5.svg"
            _draw_month(
                data, data.lat, data.lon, variable, unit, cmap,
                tag, label, rows, low, high, diff_limit, spread_limit, path,
            )
            paths[spatial_figure_key(variable, tag)] = path
    return paths


# ---------------------------------------------------------------- §4.6 正文


def _area_mean(field: np.ndarray, lat: np.ndarray,
               low: float = -90.0, high: float = 90.0) -> float:
    mask = (lat >= low) & (lat <= high)
    weights = np.cos(np.deg2rad(lat[mask]))[:, None] * np.ones((1, field.shape[1]))
    weights = np.where(np.isfinite(field[mask]), weights, 0.0)
    total = float(weights.sum())
    if total == 0.0:
        return float("nan")
    return float(np.nansum(np.where(np.isfinite(field[mask]), field[mask], 0.0) * weights)
                 / total)


def _text(value: float, signed: bool = True) -> str:
    if not np.isfinite(value):
        return "—"
    return f"{value:+.2f}" if signed else f"{value:.2f}"


def _bias_table(data: SpatialData) -> List[str]:
    header = ["**变量**", "**月份**"] + [f"**+{lead} h**" for lead in data.leads]
    header.append("**偏差最大时效**")
    rows = []
    for variable in data.variables:
        unit = PANEL_STYLE.get(variable, ("", ""))[0]
        for tag, label in data.months:
            values = [
                _area_mean(
                    data.fields[(variable, tag, lead, "model")]
                    - data.fields[(variable, tag, lead, "era5")],
                    data.lat,
                )
                for lead in data.leads
            ]
            if np.isfinite(values).any():
                worst = data.leads[int(np.nanargmax(np.abs(values)))]
                worst_cell = f"+{worst} h"
            else:
                worst_cell = "—"
            rows.append([f"{variable} ({unit})", label]
                        + [_text(value) for value in values] + [worst_cell])
    return table_lines(header, rows)


def _notes(data: SpatialData, model: str) -> List[str]:
    items: List[str] = []
    first, last = data.leads[0], data.leads[-1]
    for variable in data.variables:
        unit = PANEL_STYLE.get(variable, ("", ""))[0]
        pieces = []
        for tag, label in data.months:
            early = _area_mean(
                data.fields[(variable, tag, first, "model")]
                - data.fields[(variable, tag, first, "era5")], data.lat)
            late = _area_mean(
                data.fields[(variable, tag, last, "model")]
                - data.fields[(variable, tag, last, "era5")], data.lat)
            values = [
                _area_mean(
                    data.fields[(variable, tag, lead, "model")]
                    - data.fields[(variable, tag, lead, "era5")], data.lat)
                for lead in data.leads
            ]
            worst = data.leads[int(np.nanargmax(np.abs(values)))]
            pieces.append(
                f"{label} 从 +{first} h 的 {_text(early)} {unit} 到 +{last} h 的 "
                f"{_text(late)} {unit}（绝对偏差最大在 +{worst} h）"
            )
        items.append(f"{variable} 的全球面积加权平均偏差：" + "；".join(pieces) + "。")
        north = _area_mean(
            data.fields[(variable, data.months[0][0], last, "model")]
            - data.fields[(variable, data.months[0][0], last, "era5")],
            data.lat, 20.0, 60.0)
        south = _area_mean(
            data.fields[(variable, data.months[0][0], last, "model")]
            - data.fields[(variable, data.months[0][0], last, "era5")],
            data.lat, -60.0, -20.0)
        items.append(
            f"{variable} 在 +{last} h、{data.months[0][1]} 的中纬度带平均偏差："
            f"20–60°N {_text(north)} {unit}，20–60°S {_text(south)} {unit}。"
        )
    if data.has_spread:
        pieces = []
        for variable in data.variables:
            unit = PANEL_STYLE.get(variable, ("", ""))[0]
            early = float(np.nanmean([
                _area_mean(data.fields[(variable, tag, first, "spread")], data.lat)
                for tag, _label in data.months
            ]))
            late = float(np.nanmean([
                _area_mean(data.fields[(variable, tag, last, "spread")], data.lat)
                for tag, _label in data.months
            ]))
            trend = "增至" if late >= early else "变为"
            pieces.append(
                f"{variable} {_text(early, signed=False)} → {_text(late, signed=False)} {unit}"
                f"（{trend}）"
            )
        items.append(
            f"{model} 的全球平均离散度（成员月平均场之间的标准差）"
            f"从 +{first} h 到 +{last} h：" + "，".join(pieces) + "。"
        )
    items.append(
        "读图只看 20° 以上的成片结构：单月只有约 30 个起报样本，"
        "单格点的红蓝斑点多为噪声；+24 h 偏差接近 0 也不等于「完美」，"
        "随机误差不在这张月平均偏差图上体现。"
    )
    return bullet_lines(items)


def build_spatial_section(payload: Optional[SpatialPayload], *,
                          heading: str, figure_start: int) -> List[str]:
    """生成 §4.6 的 Markdown 行；``payload`` 为 None 时写「本批未出」。"""
    lines = [heading, ""]
    if payload is None:
        lines += [SPATIAL_ABSENT,
                  "因此本节无表、无图，保留节位以便将来补数后直接填入。"]
        return lines

    data = payload.data
    months = "、".join(label for _tag, label in data.months)
    leads = "、".join(f"+{lead} h" for lead in data.leads)
    rows_text = (
        "四行 = 集合均值 / ERA5 / 偏差 / 离散度（离散度单独色标）"
        if payload.mode == "ens"
        else "三行 = 模型 / ERA5 / 偏差"
    )
    lines += [
        f"本节给出 {months} 的月平均空间分布：按起报月内全部起报日、同一时效求平均，"
        f"ERA5 取同一批有效时刻（起报日 00Z + 时效）再平均，因此差值行是"
        f"**平均误差场（偏差）**，不是两个气候态相减。{rows_text}，"
        f"列为 {leads}。风速先逐时次合成 `sqrt(u²+v²)` 再平均；z500 单位为 gpm。",
    ]
    if payload.mode == "ens":
        lines += [
            "离散度是**成员月平均场之间的标准差**（ddof=0）——先对每个成员求月平均，"
            "再对成员求 std，不是逐时次集合 std 的月平均。",
        ]
    lines += [
        "",
        "> 这里的月份、时效与变量来自独立的空间分布 CSV，可能覆盖与正文其余各节"
        "不同的样本；不要把两者的数字互相代入或排名。",
        "",
        "**分变量全球面积加权平均偏差**",
        "",
    ]
    lines += _bias_table(data)
    lines.append("")

    expected = [spatial_figure_key(variable, tag)
                for variable in data.variables for tag, _label in data.months]
    missing = [key for key in expected if key not in payload.figures]
    if missing:
        raise ValueError(f"§4.6 缺图：{missing}（render_spatial 没有出全）")

    pairs = [(variable, tag)
             for variable in data.variables for tag, _label in data.months]
    for offset, (variable, tag) in enumerate(pairs):
        label = dict(data.months)[tag]
        number = figure_start + offset
        if payload.mode == "ens":
            caption = (
                f"图 {number}：{variable} {label} 的月平均场（{payload.model} 集合均值 / "
                f"ERA5 / 偏差 / 离散度），列为 {leads}；偏差为平均误差场，"
                "离散度为成员月平均场之间的标准差（ddof=0）。"
            )
        else:
            caption = (
                f"图 {number}：{variable} {label} 的月平均场（{payload.model} / ERA5 / "
                f"偏差），列为 {leads}；偏差为平均误差场。"
            )
        lines += [
            f"![{spatial_figure_key(variable, tag)}]({payload.figures[spatial_figure_key(variable, tag)]})",
            "",
            caption,
            "",
        ]

    lines += _notes(data, payload.model)
    return lines
