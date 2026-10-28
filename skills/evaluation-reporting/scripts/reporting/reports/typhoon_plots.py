"""Figures from the same paired summary used to render the report."""
from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Dict, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

#: 图内文字一律英文：模型显示名（报告正文用中文）在这里换成英文图例。
_ENGLISH_NAMES = {"伏羲": "FuXi", "风清": "FengQing", "通云": "Tongyun"}
_TITLES = {
    "track": "Track error",
    "along_track": "Along-track error",
    "cross_track": "Cross-track error",
    "minimum_pressure": "Minimum pressure error",
    "maximum_wind": "Maximum wind error",
}
#: 分量/强度在报告与图中的固定顺序。
_ORDER = ("along_track", "cross_track", "maximum_wind", "minimum_pressure")
_PALETTE = ("#e76f2e", "#2878b5", "#2a9d5b", "#b23a48", "#7a5195")


def _color(index: int):
    return _PALETTE[index] if index < len(_PALETTE) else plt.get_cmap("tab10")(index % 10)


def _shade(axis, rows, presentation) -> None:
    """背景带：浅灰 = 探索性时效，灰 = 个例参考时效（与报告口径一致）。"""
    leads = np.asarray([row["lead_h"] for row in rows])
    for group, shade in (("exploratory_leads", "#eeeeee"),
                         ("reference_leads", "#dddddd")):
        selected = set(presentation[group])
        for index, lead in enumerate(leads):
            if lead in selected:
                left = (leads[index - 1] + lead) / 2 if index else lead - 6
                right = (lead + leads[index + 1]) / 2 if index + 1 < len(leads) else lead + 6
                axis.axvspan(left, right, color=shade, alpha=0.6, zorder=0)


def _draw(axis, metric: str, info: dict, names, presentation,
          *, value_key: str, interval_key: str, suffix: str) -> None:
    rows = info["by_lead"]
    leads = np.asarray([row["lead_h"] for row in rows])
    _shade(axis, rows, presentation)
    for index, name in enumerate(names):
        values = [row["models"][name] for row in rows]
        axis.plot(
            leads,
            [np.nan if value[value_key] is None else value[value_key] for value in values],
            marker="o", label=_ENGLISH_NAMES.get(name, name), color=_color(index),
        )
        axis.fill_between(
            leads,
            [np.nan if value[interval_key][0] is None else value[interval_key][0]
             for value in values],
            [np.nan if value[interval_key][1] is None else value[interval_key][1]
             for value in values],
            color=_color(index), alpha=0.12,
        )
    axis.set_ylabel(f"{_TITLES[metric]} ({info['unit']})")
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.set_xlabel("Forecast lead time (h)")
    axis.set_xticks(leads)
    axis.set_xticklabels([
        f"{row['lead_h']:g}\nN={row['n_case']}\nstorms={row['n_storm']}"
        + (" *" if row["small_sample"] else "") for row in rows
    ], fontsize=8)
    if suffix == "me":
        axis.axhline(0, color="gray", linestyle="--", linewidth=1)
    else:
        axis.set_ylim(bottom=0)
    axis.legend(ncol=min(4, len(names)), frameon=False, loc="upper left")


def _number(value: object) -> Optional[float]:
    try:
        result = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _read_case(path: Path, max_lead: float) -> Tuple[list, Dict[str, tuple]]:
    """读一场次的预报轨迹与实况点（实况按有效时间去重）。"""
    forecast, observed = [], {}
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            lead = _number(row.get("lead_h"))
            if lead is None or lead < 0 or lead > max_lead + 1e-9:
                continue
            lat, lon = _number(row.get("fcst_lat")), _number(row.get("fcst_lon"))
            if lat is not None and lon is not None and -90 <= lat <= 90 and -180 <= lon <= 360:
                forecast.append((lead, lon, lat))
            obs_lat, obs_lon = _number(row.get("obs_lat")), _number(row.get("obs_lon"))
            valid = (row.get("valid_bjt") or "").strip()
            if valid and obs_lat is not None and obs_lon is not None and (
                -90 <= obs_lat <= 90 and -180 <= obs_lon <= 360
            ):
                observed.setdefault(valid, (obs_lon, obs_lat))
    forecast.sort()
    return forecast, observed


def _storm_catalog(summary: dict, max_lead: float) -> Dict[str, dict]:
    catalog: Dict[str, dict] = {}
    for source in summary["sources"]:
        for file in source["files"]:
            meta = file["metadata"]
            key = f"{str(meta.get('init_utc', ''))[:4]}:{meta.get('tcid')}"
            entry = catalog.setdefault(key, {
                "tcid": str(meta.get("tcid", "")), "name": meta.get("tcname") or "",
                "obs": {}, "models": {},
            })
            forecast, observed = _read_case(Path(file["csv"]), max_lead)
            if forecast:
                entry["models"].setdefault(source["model"], []).append(forecast)
            for valid, point in observed.items():
                entry["obs"].setdefault(valid, point)
    return dict(sorted(
        catalog.items(),
        key=lambda item: (int(item[1]["tcid"]) if item[1]["tcid"].isdigit() else 10**9,
                          item[1]["tcid"]),
    ))


def _wrap(points, reference: float):
    return [(((lon - reference + 180.0) % 360.0) + reference - 180.0, lat)
            for lon, lat in points]


def _padded(values, minimum: float = 1.5):
    lower, upper = min(values), max(values)
    padding = max((upper - lower) * 0.08, minimum)
    return lower - padding, upper + padding


def _grid(limits):
    span = limits[1] - limits[0]
    step = next((value for value in (0.5, 1, 2, 5, 10, 20, 30, 45, 60)
                 if value >= span / 5), 60)
    first = math.ceil(limits[0] / step) * step
    return [first + index * step for index in range(int(span / step) + 1)]


def _plot_storm(key: str, entry: dict, names, out_dir: Path) -> Optional[str]:
    from mpl_toolkits.basemap import Basemap

    obs_items = sorted(entry["obs"].items())
    if len(obs_items) < 2 or not entry["models"]:
        return None
    obs_points = [point for _, point in obs_items]
    reference = sum(lon for lon, _ in obs_points) / len(obs_points)
    obs_points = _wrap(obs_points, reference)
    tracks = {
        name: [_wrap([(lon, lat) for _, lon, lat in track], reference)
               for track in entry["models"].get(name, []) if track]
        for name in names
    }
    all_points = list(obs_points) + [point for name in names for track in tracks[name]
                                     for point in track]
    lon_limits = _padded([lon for lon, _ in all_points])
    lat_limits = (max(_padded([lat for _, lat in all_points])[0], -90.0),
                  min(_padded([lat for _, lat in all_points])[1], 90.0))
    mean_lat = sum(lat for _, lat in obs_points) / len(obs_points)
    map_aspect = 1 / max(math.cos(math.radians(mean_lat)), 0.25)
    # 面板高度按地图真实长宽比算，而不是给固定高度再让 aspect 缩：
    # 东西长、南北窄的台风（如 NONGFA）固定高度会留下大片空白。
    panel_width = 6.4
    axes_width = panel_width - 0.7
    data_ratio = ((lat_limits[1] - lat_limits[0]) * map_aspect
                  / (lon_limits[1] - lon_limits[0]))
    panel_height = min(max(axes_width * data_ratio + 0.55, 2.3), 6.5)
    columns = 2 if len(names) <= 4 else 3
    rows = math.ceil(len(names) / columns)
    figure, axes = plt.subplots(
        rows, columns, figsize=(panel_width * columns, panel_height * rows + 1.9),
        squeeze=False, sharex=True, sharey=True,
    )
    flattened = list(axes.flat)
    obs_lons = [lon for lon, _ in obs_points]
    obs_lats = [lat for _, lat in obs_points]
    parallels, meridians = _grid(lat_limits), _grid(lon_limits)
    forecast_handles, obs_handles = [], []
    for index, (axis, name) in enumerate(zip(flattened, names)):
        view = Basemap(
            projection="cyl", llcrnrlon=lon_limits[0], urcrnrlon=lon_limits[1],
            llcrnrlat=lat_limits[0], urcrnrlat=lat_limits[1],
            resolution="l", ax=axis,
        )
        view.drawmapboundary(fill_color="#e8f3f7", linewidth=0)
        view.fillcontinents(color="#eeeade", lake_color="#e8f3f7", zorder=0)
        view.drawcoastlines(color="#666b68", linewidth=0.65, zorder=1)
        view.drawcountries(color="#888c89", linewidth=0.45, zorder=1)
        view.drawparallels(
            parallels, labels=[index % columns == 0, 0, 0, 0],
            color="#9aa6a8", dashes=[2, 3], linewidth=0.45, fontsize=11, zorder=2,
        )
        view.drawmeridians(
            meridians, labels=[0, 0, 0, index >= columns * (rows - 1)],
            color="#9aa6a8", dashes=[2, 3], linewidth=0.45, fontsize=11, zorder=2,
        )
        for track_index, track in enumerate(tracks[name]):
            line, = axis.plot(
                [lon for lon, _ in track], [lat for _, lat in track],
                color=_color(index), linewidth=1.15, alpha=0.38, zorder=3,
                label=(f"{_ENGLISH_NAMES.get(name, name)} forecast "
                       f"({len(tracks[name])} initializations)")
                if track_index == 0 else None,
            )
            if track_index == 0:
                forecast_handles.append(line)
        obs_line, = axis.plot(
            obs_lons, obs_lats, color="#202020", linewidth=2.4, marker="o",
            markersize=3.2, markerfacecolor="white", markeredgewidth=0.8,
            label="OBS", zorder=4,
        )
        start = axis.scatter(obs_lons[0], obs_lats[0], marker="s", s=42,
                             color="#202020", zorder=5, label="OBS start")
        end = axis.scatter(obs_lons[-1], obs_lats[-1], marker="X", s=50,
                           color="#202020", zorder=5, label="OBS end")
        if index == 0:
            obs_handles = [obs_line, start, end]
        axis.set_title(_ENGLISH_NAMES.get(name, name), fontsize=15, pad=6)
        axis.set_xlim(*lon_limits)
        axis.set_ylim(*lat_limits)
        axis.set_aspect(map_aspect, adjustable="box")
        for side in ("top", "right"):
            axis.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            axis.spines[side].set_color("#666b68")
            axis.spines[side].set_linewidth(0.8)
            axis.spines[side].set_zorder(10)
    for axis in flattened[len(names):]:
        axis.set_visible(False)
    handle_labels = [
        (handle, f"{_ENGLISH_NAMES.get(name, name)} forecast")
        for handle, name in zip(forecast_handles, names)
    ] + list(zip(obs_handles, ["OBS", "OBS start", "OBS end"]))
    figure.legend(
        [handle for handle, _ in handle_labels],
        [label for _, label in handle_labels],
        loc="lower center", bbox_to_anchor=(0.5, 0.03),
        ncol=min(4, len(handle_labels)), frameon=False, fontsize=11,
    )
    figure.tight_layout(rect=(0, 0.13, 1, 0.98), h_pad=2.5, w_pad=1.0)
    # 地图受 aspect 约束，tight_layout 在行间留不出标题空间；再显式加一次行距。
    figure.subplots_adjust(hspace=0.14)
    safe_name = entry["name"].lower().replace("-", "_").replace(" ", "_")
    relative = Path("track_plots") / f"tc{entry['tcid']}_{safe_name}.svg"
    (out_dir / "track_plots").mkdir(parents=True, exist_ok=True)
    figure.savefig(out_dir / relative, dpi=150)
    plt.close(figure)
    return str(relative).replace("\\", "/")


def plot_typhoon(summary: dict, out_dir: Path) -> Dict[str, str]:
    names = summary["models"]
    metrics = summary["metrics"]
    presentation = summary["presentation"]
    components = [
        key for key in _ORDER
        if key in metrics and any(row["n_case"] for row in metrics[key]["by_lead"])
    ]
    figures: Dict[str, str] = {}

    # 图 1：总体路径误差
    figure, axis = plt.subplots(figsize=(11, 4.2))
    try:
        _draw(axis, "track", metrics["track"], names, presentation,
              value_key="mean", interval_key="ci95", suffix="mean")
        figure.tight_layout()
        figure.savefig(out_dir / "track.svg")
        figures["track"] = "track.svg"
    finally:
        plt.close(figure)

    # 图 2…：分量/强度 MAE，每个指标单独一张，与 §4.2.x 小节一一对应
    for metric in components:
        figure, axis = plt.subplots(figsize=(11, 4.2))
        try:
            _draw(axis, metric, metrics[metric], names, presentation,
                  value_key="mean", interval_key="ci95", suffix="mae")
            figure.tight_layout()
            figure.savefig(out_dir / f"mae_{metric}.svg")
            figures[f"mae_{metric}"] = f"mae_{metric}.svg"
        finally:
            plt.close(figure)

    # 接着：有符号偏差，每个指标单独一张
    for metric in components:
        figure, axis = plt.subplots(figsize=(11, 4.2))
        try:
            _draw(axis, metric, metrics[metric], names, presentation,
                  value_key="me", interval_key="me_ci95", suffix="me")
            figure.tight_layout()
            figure.savefig(out_dir / f"me_{metric}.svg")
            figures[f"me_{metric}"] = f"me_{metric}.svg"
        finally:
            plt.close(figure)

    # 逐台风路径图
    max_lead = max(summary["leads"])
    for key, entry in _storm_catalog(summary, max_lead).items():
        relative = _plot_storm(key, entry, names, out_dir)
        if relative:
            figures[f"storm:{key}"] = relative
    return figures
