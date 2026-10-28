# -*- coding: utf-8 -*-
"""统一结果长表。

全框架只在这里定义一次输出格式：任何 ``MetricResult`` 都被展平成同一组列，
各评测、各可视化模块不再自造列名。

约定：

- 标量进长表（``scores.csv``），诊断量（列联表计数等）进 ``details``；
- ``value`` 与 ``status`` 分离，空值不冒充状态；
- ``aggregation`` / ``weights_id`` 随行落盘，展示层不得自行决定聚合口径；
- 空间场、ROC、功率谱等大对象不进长表，只进 ``diagnostics/``。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

from reporting.sources.contracts import MetricResult

# 长表固定列（顺序即写盘顺序）
SCORE_COLUMNS: List[str] = [
    "run_id",
    "model_id",
    "dataset_id",
    "variable",
    "level",
    "metric",
    "product_kind",
    "aggregation",
    "weights_id",
    "region",
    # 分组指标的组标签（球谐带的 "1_4" 等）；非分组指标为空
    "group",
    "threshold",
    "window_h",
    "lead_h",
    "init_time",
    "valid_time",
    "sample_unit",
    # 台风编号。sample_unit 那一列放的是样本单位**维度名**（"storm"），
    # 不带是哪一号；路径类结果要认人，必须有自己的列。
    "storm",
    "value",
    "unit",
    "status",
    "n_valid",
    "n_requested",
    "metric_version",
    "protocol_id",
]

# 诊断量长表列（列联表计数、阈值明细等）
DETAIL_COLUMNS: List[str] = [
    "run_id",
    "model_id",
    "dataset_id",
    "variable",
    "metric",
    "product_kind",
    "threshold",
    "window_h",
    "lead_h",
    "init_time",
    "valid_time",
    # 分纬度带评估下这一列是必需的：列联表计数（hits/misses/false_alarms）
    # 逐带各不相同，没有它不同带的同名诊断量会在宽表里互相覆盖，且不报错
    "region",
    "group",
    "field",
    "value",
]

# 覆盖率表列
COVERAGE_COLUMNS: List[str] = [
    "run_id",
    "model_id",
    "dataset_id",
    "metric",
    "variable",
    "threshold",
    "window_h",
    "lead_h",
    "n_requested",
    "n_valid",
    "coverage",
    "status",
]

# MetricResult.coordinates 中可以落进长表的坐标字段
_COORD_FIELDS: Sequence[str] = (
    "variable",
    "level",
    "region",
    "threshold",
    "window_h",
    "lead_h",
    "init_time",
    "valid_time",
    "sample_unit",
    "storm",
    "weights_id",
)

# 指标 value 字典里的评分字段 -> 标准指标名
_SCORE_FIELDS: Dict[str, str] = {
    "ts": "ts",
    "ets": "ets",
    "pod": "pod",
    "far": "far",
    "miss_rate": "miss_rate",
    "漏报率": "miss_rate",
    "bias": "frequency_bias",
    "fss": "fss",
    "bs": "bs",
    "bss": "bss",
    "aroc": "aroc",
    "crps": "crps",
    "spread": "spread",
    "rmse": "rmse",
    "mae": "mae",
    "ratio": "spread_error_ratio",
    "activity_ratio": "activity_ratio",
    "fc_activity": "activity_forecast",
    "obs_activity": "activity_observation",
    "fc_obs_bias": "activity_bias",
    "power_ratio": "spectrum_power_ratio",
    # 球谐带功率：group 列保存频带标签，如 "1_4"。
    "spherical_pred": "spherical_band_power_forecast",
    "spherical_obs": "spherical_band_power_observation",
    "spherical_ratio": "spherical_band_power_ratio",
}

DETAIL_CELLS = ("hits", "misses", "false_alarms", "correct_negatives", "n_pairs")


@dataclass
class ResultTables:
    """统一长表集合。

    Attributes:
        scores: 评分长表，列见 ``SCORE_COLUMNS``；``group`` 是分组指标的组标签
            （球谐带 ``1_4`` 等、分类检验的阈值名），非分组指标为空。
        details: 诊断量长表，列见 ``DETAIL_COLUMNS``。
        coverage: 覆盖率表，列见 ``COVERAGE_COLUMNS``。
        curves: 逐点曲线类诊断量（来自 ``MetricResult.curve``），每项形如
            ``{"base": <该结果的长表基准行>, "curve": <指标给的曲线字典>}``。
            曲线**故意不展开成长表**——它们动辄上千点，展开就是几千万行；
            writer 直接从这份原始数组里出派生视图。
    """

    scores: pd.DataFrame = field(default_factory=lambda: pd.DataFrame(columns=SCORE_COLUMNS))
    details: pd.DataFrame = field(default_factory=lambda: pd.DataFrame(columns=DETAIL_COLUMNS))
    coverage: pd.DataFrame = field(default_factory=lambda: pd.DataFrame(columns=COVERAGE_COLUMNS))
    curves: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def score_table(self) -> pd.DataFrame:
        """只含固定列的长表，用于写盘。"""
        return self.scores.reindex(columns=SCORE_COLUMNS)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool)


def _clean(value: Any) -> Any:
    """把值规范成可写进 CSV 的标量；None / NaN / Inf 统一为空。"""
    if value is None:
        return ""
    if isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
        return ""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return str(value)
    return value


def _as_float(value: Any) -> Any:
    if _is_number(value):
        return float(value) if np.isfinite(value) else ""
    return _clean(value)


def _base_row(
    result: MetricResult,
    run_id: str,
    model_id: str,
    dataset_id: str,
    protocol_id: Optional[str],
    defaults: Dict[str, Any],
    coordinates: Dict[str, Any],
) -> Dict[str, Any]:
    row: Dict[str, Any] = {column: "" for column in SCORE_COLUMNS}
    row.update(
        {
            "run_id": run_id,
            "model_id": model_id,
            "dataset_id": dataset_id,
            "metric": result.metric_name,
            "product_kind": result.product_kind or "",
            "aggregation": result.aggregation or "",
            "metric_version": result.metric_version,
            "protocol_id": protocol_id or result.protocol_id or "",
            "status": result.status.value,
            "n_valid": int(result.n_valid),
            "n_requested": int(result.n_requested),
            "unit": result.unit or "",
        }
    )
    for key in _COORD_FIELDS:
        if coordinates.get(key) is not None:
            row[key] = _clean(coordinates[key])
        elif defaults.get(key) is not None:
            row[key] = _clean(defaults[key])
    return row


def _detail_row(base: Dict[str, Any], group: Any, field_name: Any, field_value: Any) -> Dict[str, Any]:
    return {
        "run_id": base.get("run_id", ""),
        "model_id": base.get("model_id", ""),
        "dataset_id": base.get("dataset_id", ""),
        "variable": base.get("variable", ""),
        "metric": base.get("metric", ""),
        "product_kind": base.get("product_kind", ""),
        "threshold": base.get("threshold", ""),
        "window_h": base.get("window_h", ""),
        "lead_h": base.get("lead_h", ""),
        "init_time": base.get("init_time", ""),
        "valid_time": base.get("valid_time", ""),
        "region": base.get("region", ""),
        "group": _clean(group),
        "field": _clean(field_name),
        "value": _clean(field_value),
    }


def _expand_grouped_value(
    result: MetricResult,
    base: Dict[str, Any],
    value: Dict[str, Any],
    detail_rows: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """展开 ``value = {分组名: {字段: 数值}}`` 形式的指标结果（如分类检验）。"""
    rows: List[Dict[str, Any]] = []
    for group_key, group_value in value.items():
        if isinstance(group_value, dict):
            threshold = group_value.get("threshold")
            n_pairs = group_value.get("n_pairs")
            detail_base = dict(base)
            if threshold is not None:
                detail_base["threshold"] = _clean(threshold)
            for field_name, field_value in group_value.items():
                key = str(field_name).strip().lower() if isinstance(field_name, str) else ""
                metric_name = _SCORE_FIELDS.get(key)
                if metric_name and _is_number(field_value):
                    row = dict(base)
                    row["metric"] = metric_name
                    row["value"] = float(field_value)
                    row["group"] = _clean(group_key)
                    if threshold is not None:
                        row["threshold"] = _clean(threshold)
                    if n_pairs is not None:
                        row["n_valid"] = int(n_pairs)
                    rows.append(row)
                else:
                    detail_rows.append(
                        _detail_row(detail_base, group_key, field_name, field_value)
                    )
        elif _is_number(group_value):
            row = dict(base)
            key = str(group_key).strip().lower()
            row["metric"] = _SCORE_FIELDS.get(key) or f"{result.metric_name}.{group_key}"
            row["value"] = float(group_value)
            row["group"] = ""
            rows.append(row)
        else:
            detail_rows.append(_detail_row(base, group_key, group_key, group_value))
    return rows


def build_tables(
    results: Iterable[MetricResult],
    *,
    run_id: str = "",
    model_id: str = "",
    dataset_id: str = "",
    protocol_id: Optional[str] = None,
    defaults: Optional[Dict[str, Any]] = None,
) -> ResultTables:
    """把一批 ``MetricResult`` 展平成统一长表。

    Args:
        results: 指标结果集合。
        run_id: 运行标识，通常是配置名。
        model_id: 预报来源标识（通常取预报数据源 source_id）。
        dataset_id: 观测/参考来源标识。
        protocol_id: 验证协议标识；未提供时取结果自带的 protocol_id。
        defaults: 结果自身没有携带坐标时的兜底值（如 variable / window_h）。

    Returns:
        ResultTables
    """
    defaults = dict(defaults or {})
    score_rows: List[Dict[str, Any]] = []
    detail_rows: List[Dict[str, Any]] = []
    coverage_rows: List[Dict[str, Any]] = []
    curve_rows: List[Dict[str, Any]] = []

    for result in results:
        coordinates = dict(result.coordinates or {})
        base = _base_row(result, run_id, model_id, dataset_id, protocol_id, defaults, coordinates)

        n_requested = int(result.n_requested)
        coverage_rows.append(
            {
                "run_id": base["run_id"],
                "model_id": base["model_id"],
                "dataset_id": base["dataset_id"],
                "metric": base["metric"],
                "variable": base["variable"],
                "threshold": base["threshold"],
                "window_h": base["window_h"],
                "lead_h": base["lead_h"],
                "n_requested": n_requested,
                "n_valid": int(result.n_valid),
                "coverage": (result.n_valid / n_requested) if n_requested else "",
                "status": result.status.value,
            }
        )

        value = result.value
        if isinstance(value, dict):
            score_rows.extend(_expand_grouped_value(result, base, value, detail_rows))
        else:
            row = dict(base)
            row["value"] = _as_float(value)
            row["group"] = ""
            score_rows.append(row)

        if result.curve is not None:
            curve_rows.append({"base": base, "curve": result.curve})

    scores = pd.DataFrame(score_rows)
    details = pd.DataFrame(detail_rows)
    coverage = pd.DataFrame(coverage_rows)

    score_columns = SCORE_COLUMNS
    if scores.empty:
        scores = pd.DataFrame(columns=score_columns)
    else:
        for column in score_columns:
            if column not in scores.columns:
                scores[column] = ""
    if details.empty:
        details = pd.DataFrame(columns=DETAIL_COLUMNS)
    else:
        for column in DETAIL_COLUMNS:
            if column not in details.columns:
                details[column] = ""
    if coverage.empty:
        coverage = pd.DataFrame(columns=COVERAGE_COLUMNS)
    else:
        for column in COVERAGE_COLUMNS:
            if column not in coverage.columns:
                coverage[column] = ""

    return ResultTables(
        scores=scores[score_columns],
        details=details[DETAIL_COLUMNS],
        coverage=coverage[COVERAGE_COLUMNS],
        curves=curve_rows,
    )


# 分类检验宽表列（兼容既有降水绘图脚本的列名）
WIDE_COLUMNS: List[str] = [
    "run_id",
    "window_h",
    "lead_h",
    # 分纬度带评估：全球行为空，其余是带名。**必须进透视键**（见
    # ``categorical_wide``），否则各带的同名 (时效, 等级) 会互相覆盖。
    "region",
    "grade",
    "threshold_mm",
    "hits",
    "misses",
    "false_alarms",
    "n_pairs",
    "TS",
    "POD",
    "FAR",
    "漏报率",
    "BIAS",
]

_SCORE_TO_WIDE = {
    "ts": "TS",
    "pod": "POD",
    "far": "FAR",
    "miss_rate": "漏报率",
    "frequency_bias": "BIAS",
}


def categorical_wide(tables: ResultTables) -> pd.DataFrame:
    """把分类检验结果透视成 阈值 x 时效 宽表。

    这是长表的派生视图，供既有降水绘图脚本使用；不改变长表口径。
    """
    scores = tables.scores
    if scores.empty or "product_kind" not in scores.columns:
        return pd.DataFrame(columns=WIDE_COLUMNS)

    categorical = scores[scores["product_kind"] == "categorical"].copy()
    if categorical.empty:
        return pd.DataFrame(columns=WIDE_COLUMNS)

    # region 进键：分纬度带时同一个 (时效, 等级) 在每条带上各有一份，
    # 不进键就是后写的带覆盖前面的带，且一声不响
    keys = ["run_id", "window_h", "lead_h", "region", "group"]
    records: Dict[tuple, Dict[str, Any]] = {}

    def _record_for(row: Any) -> Dict[str, Any]:
        key = tuple(_clean(row[name]) for name in keys)
        entry = records.setdefault(key, {name: _clean(row[name]) for name in keys})
        if _clean(row.get("threshold")) != "":
            entry["threshold_mm"] = _clean(row["threshold"])
        return entry

    # 评分列：长表 -> 宽表列名
    for _, score_row in categorical.iterrows():
        entry = _record_for(score_row)
        wide_name = _SCORE_TO_WIDE.get(str(score_row["metric"]))
        if wide_name:
            entry[wide_name] = _clean(score_row["value"])

    # 列联表计数等诊断量
    details = tables.details
    if not details.empty:
        for _, detail_row in details[details["field"].isin(DETAIL_CELLS)].iterrows():
            entry = _record_for(detail_row)
            entry[str(detail_row["field"])] = _clean(detail_row["value"])

    wide = pd.DataFrame(list(records.values())).rename(columns={"group": "grade"})
    for column in WIDE_COLUMNS:
        if column not in wide.columns:
            wide[column] = ""
    return wide[WIDE_COLUMNS]


#: 概率评分宽表列；综合统计由报告按有效阈值计算，不写综合行。
PROBABILITY_WIDE_COLUMNS: List[str] = [
    "run_id",
    "window_h",
    "lead_h",
    # 同分类宽表：分纬度带时全球行为空、其余是带名，且必须进透视键
    "region",
    "grade",
    "threshold_mm",
    "AROC",
    "BS",
    "BS_ref",
    "BSS",
    "base_rate",
    "n_points",
]

_PROBABILITY_DETAIL_FIELDS = ("BS_ref", "base_rate", "n_points")


def probability_wide(tables: ResultTables) -> pd.DataFrame:
    """把概率评分结果透视成 阈值 x 时效 宽表（AROC/BS/BSS）。

    评分值来自 ``scores.csv``，``BS_ref`` / ``base_rate`` / ``n_points`` 等诊断量
    来自明细表。不生成跨时效的 AVG 综合行，聚合口径由展示层显式声明。
    """
    scores = tables.scores
    if scores.empty or "product_kind" not in scores.columns:
        return pd.DataFrame(columns=PROBABILITY_WIDE_COLUMNS)

    probabilistic = scores[scores["product_kind"] == "probabilistic"]
    if probabilistic.empty:
        return pd.DataFrame(columns=PROBABILITY_WIDE_COLUMNS)

    keys = ["run_id", "window_h", "lead_h", "region", "group"]
    merge_keys = ["run_id", "window_h", "lead_h", "region", "grade"]
    wide = (
        probabilistic.drop_duplicates(subset=keys)[keys + ["threshold"]]
        .rename(columns={"group": "grade", "threshold": "threshold_mm"})
        .copy()
    )

    for metric_name, column in (("aroc", "AROC"), ("bs", "BS"), ("bss", "BSS")):
        values = probabilistic[probabilistic["metric"] == metric_name]
        values = values[keys + ["value"]].rename(
            columns={"group": "grade", "value": column}
        )
        wide = wide.merge(values, on=merge_keys, how="left")

    details = tables.details
    if not details.empty:
        extra = details[details["field"].isin(_PROBABILITY_DETAIL_FIELDS)]
        if not extra.empty:
            pivoted = extra.pivot_table(
                index=keys, columns="field", values="value", aggfunc="first", dropna=False
            ).reset_index()
            pivoted.columns.name = None
            pivoted = pivoted.rename(columns={"group": "grade"})
            wide = wide.merge(pivoted, on=merge_keys, how="left")

    for column in PROBABILITY_WIDE_COLUMNS:
        if column not in wide.columns:
            wide[column] = ""
    return wide[PROBABILITY_WIDE_COLUMNS]
