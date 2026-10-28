"""Validated case curves -> paired statistics, figures and a typhoon report.

Generalizes the paired statistics and storm-block bootstrap in
``typhoon_report/summarize_ens_errors.py`` without its batch-specific paths or conclusions.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import math
import re
import shlex
import shutil
import sys
import tempfile
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

LOG = logging.getLogger(__name__)
METRICS = {
    "track": ("track_err_km", "路径误差", "km"),
    "along_track": ("at_km", "沿轨误差", "km"),
    "cross_track": ("ct_km", "横轨误差", "km"),
    "minimum_pressure": ("pmin_err_hpa", "最低气压误差", "hPa"),
    "maximum_wind": ("wind_err_ms", "最大风速误差", "m/s"),
}
LEGACY_FIELDS = {
    "track_err_km": "track_err_mean_km",
    "at_km": "at_mean_km",
    "ct_km": "ct_mean_km",
    "pmin_err_hpa": "pmin_err_mean_hpa",
}
Key = Tuple[str, str, float]


@dataclass
class TyphoonArchive:
    name: str
    root: Path
    forecast_type: str
    records: Dict[Key, Dict[str, Optional[float]]]
    observations: Dict[Key, Dict[str, float]]
    provenance: List[dict]
    exclusions: dict
    manifest: dict


def _time(value: str, context: str) -> datetime:
    try:
        moment = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (ValueError, TypeError) as error:
        raise ValueError(f"{context}: 无法解析时间 {value!r}") from error
    if moment.tzinfo is not None:
        moment = moment.astimezone(timezone.utc).replace(tzinfo=None)
    return moment


def _file_case(file: dict) -> Tuple[str, str]:
    metadata = file["metadata"]
    init = _time(metadata["init_utc"], file["meta"])
    return f"{init.year}:{metadata['tcid']}", init.isoformat()


def _number(value: object, context: str) -> Optional[float]:
    if value is None or str(value).strip() == "":
        return None
    try:
        result = float(value)
    except (ValueError, TypeError) as error:
        raise ValueError(f"{context}: 非数值 {value!r}") from error
    return result if math.isfinite(result) else None


def _legacy_member_evidence(path: Path, init: datetime, offset: float, meta: dict) -> dict:
    """Read-only audit: prove the stored legacy distances are scheme B, not scheme A."""
    from reporting.sources.track import along_cross_track, great_circle_km

    member_path = path.with_name(path.stem.removesuffix("_ensemble") + "_members.csv")
    if not member_path.is_file():
        raise ValueError(f"{path}: 旧集合核验需要 {member_path.name}")
    groups: Dict[float, List[dict]] = {}
    seen = set()
    with member_path.open(encoding="utf-8-sig", newline="") as stream:
        for index, row in enumerate(csv.DictReader(stream), start=2):
            context = f"{member_path}:{index}"
            lead = _number(row.get("lead_h"), context)
            member = row.get("member", "").strip()
            if lead is None or lead < 0 or not member:
                raise ValueError(f"{context}: 缺少有效 lead_h/member")
            if (lead, member) in seen:
                raise ValueError(f"{context}: 重复成员/时效")
            seen.add((lead, member))
            if _time(row.get("valid_bjt", ""), context) != init + timedelta(
                hours=lead + offset
            ):
                raise ValueError(f"{context}: 成员有效时间冲突")
            groups.setdefault(lead, []).append({
                field: _number(row.get(field), f"{context}:{field}")
                for field in (
                    "fcst_lat", "fcst_lon", "obs_lat", "obs_lon", "obs_pmin_hpa",
                    "obs_vmax_ms", "at_km", "ct_km", "wind_err_ms", "pmin_err_hpa",
                )
            })
    evidence = {}
    initial = meta.get("init_pos")
    previous = tuple(initial) if isinstance(initial, list) and len(initial) == 2 else None
    for lead, rows in sorted(groups.items()):
        observations = {}
        for field in ("obs_lat", "obs_lon", "obs_pmin_hpa", "obs_vmax_ms"):
            values = [row[field] for row in rows if row[field] is not None]
            if values:
                if not np.allclose(values, values[0], rtol=0, atol=1e-5):
                    raise ValueError(f"{member_path}: {lead}h 的成员观测 {field} 冲突")
                observations[field] = values[0]
        positions = [row for row in rows if row["fcst_lat"] is not None
                     and row["fcst_lon"] is not None]
        if any(not (-90 <= row["fcst_lat"] <= 90 and -180 <= row["fcst_lon"] <= 360)
               for row in positions):
            raise ValueError(f"{member_path}: {lead}h 成员坐标越界")
        expected = {}
        position_components = {"along_track": None, "cross_track": None}
        if positions and "obs_lat" in observations and "obs_lon" in observations:
            lons = [row["fcst_lon"] for row in positions]
            if max(lons) - min(lons) > 180:
                raise ValueError(f"{member_path}: 旧算术经度平均跨日界线，不能确认口径")
            mean_lat = float(np.mean([row["fcst_lat"] for row in positions]))
            mean_lon = float(np.mean(lons))
            expected["track"] = float(great_circle_km(
                mean_lat, mean_lon,
                observations["obs_lat"], observations["obs_lon"],
            ))
            if previous is not None:
                at, ct = along_cross_track(
                    *previous, observations["obs_lat"], observations["obs_lon"],
                    mean_lat, mean_lon,
                )
                position_components = {"along_track": at, "cross_track": ct}
        else:
            expected["track"] = None
        for metric, (field, _, _) in METRICS.items():
            if metric != "track":
                values = [row[field] for row in rows if row[field] is not None]
                expected[metric] = float(np.mean(values)) if values else None
        evidence[lead] = {
            "observations": observations, "expected": expected,
            "position_components": position_components,
            "n_valid_members": len(positions) if expected["track"] is not None else 0,
        }
        if "obs_lat" in observations and "obs_lon" in observations:
            previous = (observations["obs_lat"], observations["obs_lon"])
    return {
        "path": str(member_path),
        "sha256": hashlib.sha256(member_path.read_bytes()).hexdigest(),
        "by_lead": evidence,
        "method": "只读核验方案B路径距离；AT/CT核对成员投影平均和平均位置投影，强度核对有符号平均",
        "component_methods": {},
    }


def load_typhoon_archive(
    name: str, root: Path, *, forecast_type: Optional[str] = None,
    verify_legacy_members: bool = False,
) -> TyphoonArchive:
    """Load current writer curves or explicitly documented legacy scheme-B curves."""
    root = Path(root).resolve()
    if not name.strip() or any(char in name for char in "|\n\r"):
        raise ValueError("模型名必须非空，且不能包含 Markdown 表格分隔符或换行")
    if not root.is_dir():
        raise ValueError(f"{root}: 台风产物目录不存在")
    cases_dir = root / "typhoon" if (root / "typhoon").is_dir() else root
    pattern = re.compile(r"tc(?P<storm>\d+)_(?P<init>\d{10})(?:_ensemble)?\.csv$")
    paths = sorted(path for path in cases_dir.glob("tc*.csv") if pattern.fullmatch(path.name))
    if not paths:
        raise ValueError(f"{root}: 没有 tc<编号>_<YYYYMMDDHH>.csv 场次曲线")
    records: Dict[Key, Dict[str, Optional[float]]] = {}
    observations: Dict[Key, Dict[str, float]] = {}
    provenance = []
    counts: Counter = Counter()
    invalid = Counter()
    types = set()
    for path in paths:
        match = pattern.fullmatch(path.name)
        assert match is not None
        stem = path.stem.removesuffix("_ensemble")
        meta_path = path.with_name(stem + "_meta.json")
        if not meta_path.is_file():
            raise ValueError(f"{path}: 缺少 {meta_path.name}，不能猜预报类型或时区")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        kind = meta.get("forecast_type", forecast_type)
        if forecast_type is not None and kind != forecast_type:
            raise ValueError(f"{meta_path}: forecast_type 与显式指定类型冲突")
        if kind not in ("det", "ens"):
            raise ValueError(f"{meta_path}: forecast_type 必须明确为 det 或 ens")
        types.add(kind)
        init = _time(meta.get("init_utc", ""), str(meta_path))
        if init.strftime("%Y%m%d%H") != match["init"]:
            raise ValueError(f"{path}: 文件名与 init_utc 冲突")
        storm = str(meta.get("tcid", ""))
        if storm != match["storm"]:
            raise ValueError(f"{path}: 文件名与 tcid 冲突")
        # Include the initialization year to distinguish short storm IDs across years.
        storm_key = f"{init.year}:{storm}"
        offset = _number(meta.get("tz_shift_h"), str(meta_path))
        if offset is None:
            raise ValueError(f"{meta_path}: 缺少有限 tz_shift_h")
        if meta.get("init_bjt") and _time(meta["init_bjt"], str(meta_path)) != (
            init + timedelta(hours=offset)
        ):
            raise ValueError(f"{meta_path}: init_bjt 与 init_utc/tz_shift_h 冲突")
        with path.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            fields = set(reader.fieldnames or [])
            legacy = "track_err_km" not in fields and "track_err_mean_km" in fields
            mapping = {
                key: LEGACY_FIELDS.get(field, field) if legacy else field
                for key, (field, _, _) in METRICS.items()
            }
            if "lead_h" not in fields or mapping["track"] not in fields:
                raise ValueError(f"{path}: 缺少 lead_h 或路径误差列")
            audit = _legacy_member_evidence(path, init, offset, meta) if (
                legacy and verify_legacy_members
            ) else None
            if kind == "ens":
                aggregation = meta.get("aggregation")
                if legacy:
                    confirmed = aggregation == "B" or (
                        isinstance(aggregation, dict)
                        and mapping["track"] in str(aggregation.get("B", ""))
                    ) or aggregation == (
                        "ensemble mean position error by lead_h (先平均位置再求误差, 方案B)"
                    )
                else:
                    confirmed = isinstance(aggregation, dict) and (
                        "track_err_km" in str(aggregation.get("B", ""))
                    )
                if not confirmed and audit is None:
                    raise ValueError(f"{meta_path}: 无法确认路径列属于集合方案 B")
            member_counts = set()
            valid_members = set()
            for index, row in enumerate(reader, start=2):
                context = f"{path}:{index}"
                lead = _number(row.get("lead_h"), context)
                if lead is None or lead < 0:
                    raise ValueError(f"{context}: lead_h 必须是有限非负数")
                key = (storm_key, init.isoformat(), lead)
                if key in records:
                    raise ValueError(f"{context}: 重复起报/时效键 {key}")
                if audit is not None and lead not in audit["by_lead"]:
                    raise ValueError(f"{context}: 成员明细缺少该时效，无法核验")
                if row.get("valid_bjt"):
                    if _time(row["valid_bjt"], context) != init + timedelta(
                        hours=lead + offset
                    ):
                        raise ValueError(f"{context}: valid_bjt 与 init+lead+tz_shift 冲突")
                elif audit is None:
                    raise ValueError(f"{context}: 缺少 valid_bjt，不能验证有效时间")
                values = {}
                for metric, field in mapping.items():
                    value = _number(row.get(field), f"{context}:{field}")
                    values[metric] = value
                    if value is None:
                        invalid[metric] += 1
                if values["track"] is not None and values["track"] < 0:
                    raise ValueError(f"{context}: 路径距离不能为负")
                if audit is not None:
                    expected = audit["by_lead"][lead]["expected"]
                    for metric, value in values.items():
                        candidates = {"member_signed_mean": expected[metric]}
                        if metric in ("along_track", "cross_track"):
                            candidates["mean_position_projection"] = (
                                audit["by_lead"][lead]["position_components"][metric]
                            )
                        matches = [method for method, candidate in candidates.items()
                                   if candidate is not None and value is not None
                                   and math.isclose(value, candidate, rel_tol=0, abs_tol=0.01)]
                        if value is not None and not matches:
                            raise ValueError(
                                f"{context}: {metric} 与成员明细核验不一致"
                                f"（CSV={value}，期望={candidates}）"
                            )
                        if matches and metric in ("along_track", "cross_track"):
                            audit["component_methods"].setdefault(metric, []).append(matches)
                bad_position = False
                for lat, lon in (("fcst_lat", "fcst_lon"), ("obs_lat", "obs_lon")):
                    if lat in fields and lon in fields:
                        latitude = _number(row.get(lat), context)
                        longitude = _number(row.get(lon), context)
                        if latitude is None or longitude is None or not (
                            -90 <= latitude <= 90 and -180 <= longitude <= 360
                        ):
                            bad_position = True
                if bad_position:
                    counts["invalid_position_rows"] += 1
                    for metric in ("track", "along_track", "cross_track"):
                        values[metric] = None
                observation = {}
                for field in ("obs_lat", "obs_lon", "obs_pmin_hpa", "obs_vmax_ms"):
                    value = _number(row.get(field), f"{context}:{field}")
                    if value is not None:
                        observation[field] = value
                if audit is not None:
                    observation = audit["by_lead"][lead]["observations"]
                observations[key] = observation
                for field, target in (
                    ("n_members", member_counts), ("n_valid_members", valid_members)
                ):
                    value = _number(row.get(field), f"{context}:{field}")
                    if value is not None:
                        if value < 0 or value != int(value):
                            raise ValueError(f"{context}: {field} 必须为非负整数")
                        target.add(int(value))
                if kind == "ens":
                    total = _number(row.get("n_members"), context)
                    valid = _number(row.get("n_valid_members"), context)
                    if total is not None and valid is not None and valid > total:
                        raise ValueError(f"{context}: n_valid_members 大于 n_members")
                    if audit is not None and valid is not None and valid != (
                        audit["by_lead"][lead]["n_valid_members"]
                    ):
                        raise ValueError(f"{context}: 有效成员数与明细冲突")
                records[key] = values
                counts["rows"] += 1
            provenance.append({
                "csv": str(path), "meta": str(meta_path),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "meta_sha256": hashlib.sha256(meta_path.read_bytes()).hexdigest(),
                "metadata": meta, "field_mapping": mapping,
                "member_counts": sorted(member_counts),
                "valid_member_counts": sorted(valid_members),
                "declared_forecast_type": forecast_type,
                "legacy_member_audit": {
                    key: value for key, value in audit.items() if key != "by_lead"
                } if audit is not None else None,
            })
    if len(types) != 1:
        raise ValueError(f"{root}: 同一模型目录混有确定性和集合产物")
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    pipeline = manifest.get("resolved_config", {}).get("pipeline")
    kind = next(iter(types))
    if pipeline in ("weather_typhoon_det", "weather_typhoon_ens"):
        expected = "ens" if pipeline == "weather_typhoon_ens" else "det"
        if kind != expected:
            raise ValueError(f"{root}: manifest pipeline 与场次 forecast_type 冲突")
    return TyphoonArchive(
        name, root, types.pop(), records, observations, provenance,
        {**counts, "missing_or_nonfinite_by_metric": dict(invalid)}, manifest,
    )


def storm_bootstrap(
    values: np.ndarray, storms: Sequence[str], n_bootstrap: int, seed: int
) -> Optional[np.ndarray]:
    """Shared resampling for all model columns, retaining repeated storm draws."""
    labels = np.asarray(storms)
    unique = np.unique(labels)
    if n_bootstrap == 0 or len(unique) < 2:
        return None
    groups = [np.flatnonzero(labels == storm) for storm in unique]
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(n_bootstrap):
        selected = rng.integers(0, len(groups), size=len(groups))
        indices = np.concatenate([groups[index] for index in selected])
        samples.append(values[indices].mean(axis=0))
    return np.asarray(samples)


def _interval(samples: Optional[np.ndarray]) -> List[Optional[float]]:
    if samples is None:
        return [None, None]
    return [float(value) for value in np.percentile(samples, [2.5, 97.5])]


def analyze_typhoon(
    archives: Sequence[TyphoonArchive], *, leads: Optional[Sequence[float]] = None,
    n_bootstrap: int = 2000, seed: int = 20250910,
    fixed_leads: Optional[Sequence[float]] = None, small_cases: int = 10,
    small_storms: int = 3,
) -> dict:
    if not archives or len({item.name for item in archives}) != len(archives):
        raise ValueError("至少需要一个模型，且模型名不能重复")
    if len({item.forecast_type for item in archives}) != 1:
        raise ValueError("确定性与集合报告不能混用；请分别生成")
    if n_bootstrap < 0 or seed < 0 or small_cases < 1 or small_storms < 1:
        raise ValueError("bootstrap 次数/种子须非负，少样本阈值须为正整数")
    available = sorted({key[2] for item in archives for key in item.records})
    selected = sorted(set(leads)) if leads is not None else available
    if not selected or any(not math.isfinite(lead) or lead < 0 for lead in selected):
        raise ValueError("时效须为非空有限非负数列表")
    if fixed_leads is not None and (
        not fixed_leads or not set(fixed_leads).issubset(selected)
    ):
        raise ValueError("固定样本时效必须非空且包含在报告时效中")
    if fixed_leads is not None:
        fixed_leads = sorted({float(lead) for lead in fixed_leads})
    names = [item.name for item in archives]
    all_keys = set.intersection(*(set(item.records) for item in archives))
    for key in all_keys:
        for first, second in combinations(archives, 2):
            a, b = first.observations[key], second.observations[key]
            for field in a.keys() & b.keys():
                if not math.isclose(a[field], b[field], rel_tol=0, abs_tol=1e-5):
                    raise ValueError(f"{key}: {first.name}/{second.name} 观测 {field} 冲突")
    results = {}
    for metric_index, (metric, (_, title, unit)) in enumerate(METRICS.items()):
        by_lead = []
        for lead_index, lead in enumerate(selected):
            keys = sorted(key for key in all_keys if key[2] == lead and all(
                item.records[key][metric] is not None for item in archives
            ))
            values = np.asarray([
                [item.records[key][metric] for item in archives] for key in keys
            ], dtype=float).reshape(len(keys), len(archives))
            storms = [key[0] for key in keys]
            absolute = values if metric == "track" else np.abs(values)
            boot_seed = seed + metric_index * len(selected) + lead_index
            samples = storm_bootstrap(absolute, storms, n_bootstrap, boot_seed)
            signed_samples = (
                storm_bootstrap(values, storms, n_bootstrap, boot_seed)
                if metric != "track" else None
            )
            models = {}
            for index, name in enumerate(names):
                models[name] = {
                    "mean": float(absolute[:, index].mean()) if keys else None,
                    "ci95": _interval(samples[:, index] if samples is not None else None),
                    "me": float(values[:, index].mean()) if keys and metric != "track" else None,
                    "me_ci95": _interval(
                        signed_samples[:, index] if signed_samples is not None else None
                    ),
                }
            differences = []
            for first, second in combinations(range(len(names)), 2):
                differences.append({
                    "first": names[first], "second": names[second],
                    "mean": float((absolute[:, first] - absolute[:, second]).mean())
                    if keys else None,
                    "ci95": _interval(
                        samples[:, first] - samples[:, second] if samples is not None else None
                    ),
                    "me": float((values[:, first] - values[:, second]).mean())
                    if keys and metric != "track" else None,
                    "me_ci95": _interval(
                        signed_samples[:, first] - signed_samples[:, second]
                        if signed_samples is not None else None
                    ),
                })
            by_lead.append({
                "lead_h": lead, "n_case": len(keys), "n_storm": len(set(storms)),
                "small_sample": len(keys) < small_cases or len(set(storms)) < small_storms,
                "paired_keys": [list(key) for key in keys],
                "excluded_by_model": {
                    item.name: sum(key[2] == lead for key in item.records) - len(keys)
                    for item in archives
                },
                "models": models, "differences": differences,
            })
        results[metric] = {
            "title": title, "unit": unit,
            "statistic": "mean_distance" if metric == "track" else "mae",
            "by_lead": by_lead,
        }
    if not any(item["n_case"] for item in results["track"]["by_lead"]):
        raise ValueError("所选时效没有任何有效共同路径样本，不能生成报告")
    fixed = None
    if fixed_leads is not None:
        fixed_cases = {
            key[:2] for key in all_keys if all(
                (*key[:2], lead) in all_keys and all(
                    item.records[(*key[:2], lead)]["track"] is not None for item in archives
                ) for lead in fixed_leads
            )
        }
        fixed = {
            "leads": sorted(set(fixed_leads)), "n_case": len(fixed_cases),
            "n_storm": len({key[0] for key in fixed_cases}),
            "means": {
                item.name: {
                    str(lead): float(np.mean([
                        item.records[(*key, lead)]["track"] for key in sorted(fixed_cases)
                    ])) if fixed_cases else None for lead in sorted(set(fixed_leads))
                } for item in archives
            },
        }
    return {
        "forecast_type": archives[0].forecast_type, "models": names, "leads": selected,
        "weighting": "equal initialization cases, paired separately by metric and lead",
        "difference_direction": "first - second; mean is distance mean or MAE, me is signed",
        "common_file_cases": [
            list(key) for key in sorted(set.intersection(*(
                {_file_case(file) for file in item.provenance} for item in archives
            )))
        ],
        "bootstrap": {"iterations": n_bootstrap, "seed": seed, "unit": "storm", "ci": 0.95},
        "small_sample": {"n_case_below": small_cases, "n_storm_below": small_storms},
        "sources": [{
            "model": item.name, "root": str(item.root),
            "n_case": len({key[:2] for key in item.records}),
            "n_storm": len({key[0] for key in item.records}),
            "init_range": sorted({key[1] for key in item.records})[::max(
                1, len({key[1] for key in item.records}) - 1
            )],
            "exclusions": item.exclusions, "files": item.provenance,
            "failed_chunks": item.manifest.get("failed_chunks", []),
            "statuses": item.manifest.get("statuses", {}),
            "evaluation_config": item.manifest.get("resolved_config", {}),
        } for item in archives],
        "metrics": results, "fixed_sample_track": fixed,
    }


def _format(value: Optional[float]) -> str:
    return "—" if value is None else f"{value:.1f}"


def _ci(values: Sequence[Optional[float]]) -> str:
    return "未计算" if values[0] is None else f"[{_format(values[0])}, {_format(values[1])}]"


def _table(headers: Sequence[str], rows: Sequence[Sequence[object]]) -> str:
    return "\n".join([
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
        *["| " + " | ".join(str(value).replace("|", "\\|").replace("\n", " ")
                           for value in row) + " |" for row in rows],
    ])

def _lead_text(leads: Sequence[float]) -> str:
    return "、".join(f"{lead:g}" for lead in leads) + " h" if leads else "无"


def _focus_difference(row: dict, baseline: str, focus: str) -> dict:
    """Orient precomputed paired differences as baseline - focus without recomputing."""
    difference = next(
        item for item in row["differences"]
        if {item["first"], item["second"]} == {baseline, focus}
    )
    if difference["first"] == baseline:
        return difference
    return {
        **difference, "first": baseline, "second": focus,
        "mean": -difference["mean"] if difference["mean"] is not None else None,
        "ci95": [-value if value is not None else None for value in difference["ci95"][::-1]],
        "me": -difference["me"] if difference["me"] is not None else None,
        "me_ci95": [
            -value if value is not None else None for value in difference["me_ci95"][::-1]
        ],
    }


def _comparison_reading(info: dict, focus: str, baseline: str, primary: Sequence[float]) -> str:
    rows = [row for row in info["by_lead"] if row["lead_h"] in primary and row["n_case"]]
    if not rows:
        return f"{info['title']}：主要时效没有有效共同样本，不能比较。"
    lower, higher, equal, supported_lower, supported_higher, unresolved, no_ci = (
        [], [], [], [], [], [], []
    )
    for row in rows:
        diff = _focus_difference(row, baseline, focus)
        if diff["mean"] > 0:
            lower.append(row["lead_h"])
        elif diff["mean"] < 0:
            higher.append(row["lead_h"])
        else:
            equal.append(row["lead_h"])
        if diff["ci95"][0] is None:
            no_ci.append(row["lead_h"])
        elif diff["ci95"][0] > 0:
            supported_lower.append(row["lead_h"])
        elif diff["ci95"][1] < 0:
            supported_higher.append(row["lead_h"])
        else:
            unresolved.append(row["lead_h"])
    end = rows[-1]
    diff = _focus_difference(end, baseline, focus)
    return (
        f"**{focus} 相对 {baseline} 的{info['title']}**："
        f"均值较低的主要时效为 {_lead_text(lower)}，较高为 {_lead_text(higher)}。"
        + (f"均值相同为 {_lead_text(equal)}。" if equal else "")
        +
        f"配对区间支持主模型误差较低的时效为 {_lead_text(supported_lower)}，"
        f"支持较高的时效为 {_lead_text(supported_higher)}；"
        f"区间包含 0 的时效为 {_lead_text(unresolved)}，未计算区间为 {_lead_text(no_ci)}。"
        f"{end['lead_h']:g} h 时，{focus}={_format(end['models'][focus]['mean'])}、"
        f"{baseline}={_format(end['models'][baseline]['mean'])} {info['unit']}，"
        f"差值（基准−主模型）={_format(diff['mean'])} {_ci(diff['ci95'])} {info['unit']}，"
        f"N_case={end['n_case']}、N_storm={end['n_storm']}。"
        + (f"其中 {_lead_text([row['lead_h'] for row in rows if row['small_sample']])} "
           "为少样本，只作方向性观察。" if any(row["small_sample"] for row in rows) else "")
    )


def _bias_reading(info: dict, focus: str, primary: Sequence[float]) -> str:
    rows = [row for row in info["by_lead"] if row["lead_h"] in primary and row["n_case"]]
    if not rows:
        return f"{info['title']}：无主要时效有效偏差样本。"
    positive, negative = [], []
    for row in rows:
        ci = row["models"][focus]["me_ci95"]
        if ci[0] is not None and ci[0] > 0:
            positive.append(row["lead_h"])
        elif ci[1] is not None and ci[1] < 0:
            negative.append(row["lead_h"])
    row = rows[-1]
    values = row["models"][focus]
    return (
        f"**{focus} 的{info['title']}偏差**：正偏差区间不跨 0 的主要时效为 "
        f"{_lead_text(positive)}，负偏差为 {_lead_text(negative)}。"
        f"{row['lead_h']:g} h ME={_format(values['me'])} {_ci(values['me_ci95'])} "
        f"{info['unit']}，MAE={_format(values['mean'])} {info['unit']}；"
        "净偏差接近 0 不能代替绝对误差小，单模型偏差区间不等于模型间技巧检验。"
    )


def _storm_table(archives: Sequence[TyphoonArchive], summary: dict) -> list:
    """逐台风主要时效平均路径误差；各模型使用完全相同的配对键。"""
    names = list(summary["models"])
    leads = [float(lead) for lead in summary["presentation"]["primary_leads"]]
    labels: Dict[str, str] = {}
    for archive in archives:
        for file in archive.provenance:
            meta = file["metadata"]
            key = f"{str(meta.get('init_utc', ''))[:4]}:{meta.get('tcid')}"
            labels.setdefault(key, str(meta.get("tcname") or ""))
    storms = sorted(
        {key[0] for key in archives[0].records},
        key=lambda storm: (int(storm.split(":")[1]) if storm.split(":")[1].isdigit() else 10**9,
                           storm),
    )
    table = []
    for storm in storms:
        values: Dict[str, list] = {archive.name: [] for archive in archives}
        inits, n_sample = set(), 0
        for init in sorted({key[1] for key in archives[0].records if key[0] == storm}):
            for lead in leads:
                key = (storm, init, lead)
                row = [archive.records.get(key, {}).get("track") for archive in archives]
                if any(value is None for value in row):
                    continue
                for archive, value in zip(archives, row):
                    values[archive.name].append(value)
                inits.add(init)
                n_sample += 1
        if not n_sample:
            continue
        means = {name: float(np.mean(column)) for name, column in values.items() if column}
        best = min(means.values())
        table.append({
            "key": storm,
            "label": f"TC{storm.split(':')[1]} {labels.get(storm, '')}".strip(),
            "n_case": len(inits), "n_sample": n_sample, "means": means,
            "best": "、".join(name for name in names if name in means
                             and math.isclose(means[name], best, rel_tol=0, abs_tol=1e-9)),
        })
    return table


def render_typhoon_report(summary: dict, figures: Mapping[str, str], command: str) -> str:
    """Render the report contract; every quantitative statement comes from summary."""
    ensemble = summary["forecast_type"] == "ens"
    kind = "集合平均" if ensemble else "确定性"
    metrics = summary["metrics"]
    track = metrics["track"]["by_lead"]
    first = next(item for item in track if item["n_case"])
    names = summary["models"]
    presentation = summary["presentation"]
    focus = presentation["primary_model"]
    previous = presentation["previous_model"]
    primary = presentation["primary_leads"]
    discussion = sorted(primary + presentation["exploratory_leads"])
    representative = next(
        (row for row in reversed(track) if row["lead_h"] in primary and row["n_case"]), first
    )
    active = {
        key: info for key, info in metrics.items()
        if any(row["n_case"] for row in info["by_lead"])
    }
    comparison_order = ([previous] if previous else []) + [
        name for name in names if name not in (focus, previous)
    ]
    conclusions = [
        _comparison_reading(info, focus, baseline, primary)
        for baseline in comparison_order for info in active.values()
    ]
    biases = [
        _bias_reading(info, focus, primary) for key, info in active.items() if key != "track"
    ]
    storms = summary.get("storm_table") or []
    wins: Counter = Counter()
    for storm in storms:
        for name in storm["best"].split("、"):
            if name:
                wins[name] += 1
    storm_sentence = (
        "逐台风看，" + "、".join(
            f"{name} 在 {wins[name]} 个台风上平均路径误差最低" for name in names if wins[name]
        ) + "；属个例观察，不参与总体显著性判断。"
    ) if storms and len(names) > 1 and wins else ""

    def horizontal(info: dict, statistic: str = "mean", with_ci: bool = False) -> str:
        rows = [row for row in info["by_lead"]
                if not with_ci or row["lead_h"] in discussion]
        return _table(["模型"] + [f"{row['lead_h']:g} h" for row in rows], [
            [name] + [
                _format(row["models"][name][statistic]) + (
                    " " + _ci(row["models"][name]["me_ci95" if statistic == "me" else "ci95"])
                    if with_ci else ""
                ) for row in rows
            ] for name in names
        ])

    def difference_table(info: dict) -> str:
        rows = [row for row in info["by_lead"] if row["lead_h"] in discussion]
        return _table(["时效"] + [f"{name} − {focus}" for name in comparison_order], [
            [f"{row['lead_h']:g} h"] + [
                _format(_focus_difference(row, baseline, focus)["mean"]) + " "
                + _ci(_focus_difference(row, baseline, focus)["ci95"])
                for baseline in comparison_order
            ] for row in rows
        ])

    narrative = "；".join(
        f"{name} 路径误差 {_format(representative['models'][name]['mean'])} km" for name in names
    )
    common_cases = {
        tuple(key[:2]) for row in track for key in row["paired_keys"]
    }
    lines = [
        f"# {'、'.join(names)} {kind}台风预报误差"
        f"{'对比' if len(names) > 1 else ''}报告", "## 1. 摘要",
        f"以 **{focus}** 为主模型" + (f"，以 **{previous}** 为旧版本比较对象。" if previous else "。"),
        f"主要讨论 {_lead_text(primary)}；探索性时效 {_lead_text(presentation['exploratory_leads'])}；"
        f"个例参考时效 {_lead_text(presentation['reference_leads'])}。"
        + ("讨论范围由用户显式指定，不根据结果优劣挑选。"
           if presentation["primary_leads_source"] == "explicit" else
           "未指定主要时效，按路径样本覆盖阈值选择；若全部为少样本，则仅作方向性观察。"
           "正式分析建议显式指定讨论范围。"),
        f"逐指标逐时效有效共同样本、起报个例等权。文件交集为 "
        f"{len(summary['common_file_cases'])} 个起报、"
        f"{len({key[0] for key in summary['common_file_cases']})} 个台风。"
        f"所选时效的路径有效共同样本合计 "
        f"{len(common_cases)} 个起报、{len({key[0] for key in common_cases})} 个台风"
        "（不同于原始文件交集，具体时效分母见 3.4 节）。",
        f"{representative['lead_h']:g} h，N_case={representative['n_case']}，"
        f"N_storm={representative['n_storm']}；{narrative}。",
        conclusions[0] if conclusions else "单模型报告，不作模型间改进或排名判断。",
        *([storm_sentence] if storm_sentence else []),
        "均值排序不等于可推广的技巧排序；误差与有符号偏差分开呈现。",
        "## 2. 数据与覆盖",
        _table(["模型", "类型", "来源", "可用起报个例", "台风数"], [
            [
                source["model"], kind, source["root"], source["n_case"], source["n_storm"],
            ] for source in summary["sources"]
        ]),
    ]
    if previous:
        findings = []
        for key, info in active.items():
            lower, higher = [], []
            for row in info["by_lead"]:
                if row["lead_h"] not in primary or not row["n_case"]:
                    continue
                ci = _focus_difference(row, previous, focus)["ci95"]
                if ci[0] is not None and ci[0] > 0:
                    lower.append(row["lead_h"])
                elif ci[1] is not None and ci[1] < 0:
                    higher.append(row["lead_h"])
            if lower or higher:
                findings.append(
                    f"{info['title']}：支持主模型误差较低 {_lead_text(lower)}，"
                    f"支持较高 {_lead_text(higher)}"
                )
        lines.insert(lines.index("## 2. 数据与覆盖"),
                     "相对旧版本的主要时效配对信号：" + (
                         "；".join(findings) if findings else
                         "没有区间不跨零的差值证据；区间未计算不能当作无差异证据。"
                     ) + "。未作多重比较校正，不能概括为全时效稳定改进。")
    lines.append(_table(
        ["模型", "起报范围", "缺失/非有限路径记录", "失败块", "成员规模（场次覆盖）"],
        [
            [source["model"], " 至 ".join(source["init_range"]),
             source["exclusions"]["missing_or_nonfinite_by_metric"].get("track", 0),
             len(source["failed_chunks"]) if source["evaluation_config"] or source["statuses"]
             or source["failed_chunks"] else "未知（无运行记录）",
             "；".join(
                 f"{'/'.join(str(value) for value in size) or '未知'} 成员：{count} 场次"
                 for size, count in sorted(Counter(
                     tuple(file["member_counts"]) for file in source["files"]
                 ).items())
             ) if ensemble else "不适用"]
            for source in summary["sources"]
        ],
    ))
    lines.append(
        "成员规模按每场次实际记录的规模集合计数（可能随时效变化），不是有效成员数。"
        "逐文件诊断参数、观测报文来源、状态、成员失败和排除明细保存在 summary.json；"
        "缺少完成记录不能推断所有成员完成。"
    )
    if ensemble:
        common_ids = {tuple(key) for key in summary["common_file_cases"]}
        lines.append(_table(["模型", "文件交集内成员规模与起报数"], [
            [source["model"], "；".join(
                f"{'/'.join(str(value) for value in size) or '未知'} 成员：{count} 起报"
                for size, count in sorted(Counter(
                    tuple(file["member_counts"]) for file in source["files"]
                    if _file_case(file) in common_ids
                ).items())
            )] for source in summary["sources"]
        ]))
    if any(file["legacy_member_audit"] for source in summary["sources"] for file in source["files"]):
        lines.append(
            "历史适配：旧集合汇总已用成员明细只读核验，保留陈旧元数据而不改写。"
            "AT/CT 可能采用成员投影平均或平均位置投影，这一算法差异限制分量比较解释；"
            "具体匹配方法与 0.01 对应单位的精度容差见 summary.json。"
        )
    lines.extend([
        "观测来源及风速标准按原评测配置核实；CSV 的 obs_* 已检查跨模型一致性。"
        "元数据未提供的来源/算法不能由模型名推断；逐文件字段映射与元数据见 summary.json。",
        "## 3. 评测方法与指标",
        "### 3.1 严格配对与样本权重",
        "路径距离读取场次 CSV 已计算的大圆距离（km），不重新诊断、不重算指标。"
        "当前项目算法为 Haversine、R=6371 km；历史数据算法按来源元数据确认。",
        "共同键为 (含年份的台风编号, init_time_utc, lead_h)，每指标每时效独立取有效交集；"
        "同一模型的路径、沿轨、横轨、风速与气压不共享未经校验的样本数。",
        "每个起报个例等权；同一台风的多次起报在均值与 bootstrap 中各自计入。"
        "不对不同样本的模型均值做比较，不跨时效生成未声明权重的综合总分。"
        "文件交集按场次文件计，路径共同样本按 (台风, 起报, 时效) 配对键计；"
        "前者只描述数据覆盖，不能替代后者作分母。",
        "集合主结果采用方案 B（先平均位置后求误差）；方案 A 不混入。"
        "集合强度取成员有符号误差平均的绝对值，不是成员 MAE。" if ensemble
        else "确定性每起报个例每时效贡献一个指标；集合算法不适用。",
        "### 3.2 指标定义",
        _table(["指标", "定义", "单位", "解读方向"], [
            ["路径误差", "预报位置与实况位置的大圆距离，逐起报逐时效平均", "km", "越低越好"],
            ["沿轨误差 MAE", "沿路径方向有符号误差的绝对值平均（AT）", "km", "越低越好"],
            ["横轨误差 MAE", "垂直于路径方向有符号误差的绝对值平均（CT）", "km", "越低越好"],
            ["最大风速误差 MAE", "预报与实况最大风速的有符号误差绝对值平均", "m/s", "越低越好"],
            ["最低气压误差 MAE", "预报与实况最低气压的有符号误差绝对值平均", "hPa", "越低越好"],
            ["有符号偏差 ME", "上述分量与强度误差的有符号平均", "km / m/s / hPa", "接近 0"],
        ]),
        "MAE=mean(abs(e))，ME=mean(e)。当前强度 e=forecast-observation；"
        "AT/CT 正负依赖观测参考方向，不能直接解释为东/西或快/慢。"
        "最低气压正 Bias 表示预报气压偏高、强度偏弱。",
        "### 3.3 不确定性与差值方向",
        f"台风分组 bootstrap：{summary['bootstrap']['iterations']} 次，"
        f"随机种子 {summary['bootstrap']['seed']}，95% 百分位区间。"
        "同一台风的全部配对起报共同重采样，重复抽中重复计入；所有模型共享抽样。"
        "次数为 0 或独立台风不足 2 时不计算区间。",
        f"正文差值为“比较模型 − {focus}”，负值表示比较模型误差更低；"
        "ME 差值不能直接解释为谁更好。区间不跨 0 只是单指标单时效的样本内证据，"
        "未进行多重比较校正，也不通过两条均值曲线的区间是否重叠判断差异。",
        "### 3.4 样本量",
        f"少样本标记：N_case < {summary['small_sample']['n_case_below']} 或 "
        f"N_storm < {summary['small_sample']['n_storm_below']}，不是显著性检验阈值。",
        _table(["指标 / 样本量"] + [f"{lead:g} h" for lead in summary["leads"]], [
            [f"{info['title']} {label}"] + [row[field] for row in info["by_lead"]]
            for info in active.values()
            for label, field in (("起报数", "n_case"), ("台风数", "n_storm"))
        ]),
        "## 4. 实验数据",
        "本章样本分母均见 3.4 节；差值表只列主要时效与探索性时效，"
        "完整模型对与时效见 differences.csv。",
        "### 4.1 总体路径误差",
        f"![路径误差](./{figures['track']})",
        "图 1：路径误差（km）、逐时效严格配对；阴影为可计算的 95% 台风分组区间；"
        "横轴标注各时效的配对起报数与台风数。",
        horizontal(metrics["track"]),
        "单位：km。时效对应的配对起报数与台风数见 3.4 节，均值区间完整值见 summary.json。",
        difference_table(metrics["track"]) if comparison_order else "未提供对比模型。",
        "**小结**：" + (
            _comparison_reading(metrics["track"], focus, comparison_order[0], primary)
            if comparison_order else
            f"{representative['lead_h']:g} h，{narrative}；不作跨模型判断。"
        ),
    ])
    fixed = summary["fixed_sample_track"]
    if fixed is None:
        lines.append("固定完整样本敏感性：未计算；可用 --fixed-leads 指定路径检验窗口。")
    else:
        lines.extend([
            f"固定路径样本：N_case={fixed['n_case']}，N_storm={fixed['n_storm']}。",
            _table(["模型"] + [f"{lead:g} h (km)" for lead in fixed["leads"]], [
                [name] + [_format(fixed["means"][name][str(lead)]) for lead in fixed["leads"]]
                for name in names
            ]),
        ])
        if fixed["n_case"]:
            end = fixed["leads"][-1]
            lines.append(
                f"固定样本 {end:g} h，{focus} 路径误差 "
                f"{_format(fixed['means'][focus][str(end)])} km"
                + (f"，{previous} 为 {_format(fixed['means'][previous][str(end)])} km"
                   if previous else "")
                + "。该比较控制了时效样本更替，不是显著性检验的替代证据。"
            )
    order = ("along_track", "cross_track", "maximum_wind", "minimum_pressure")
    indexed = [(key, metrics[key]) for key in order if key in metrics]
    indexed += [(key, info) for key, info in metrics.items()
                if key not in order and key != "track"]
    available = [(key, info) for key, info in indexed
                 if any(row["n_case"] for row in info["by_lead"])]
    figure_number = 1
    lines.append("### 4.2 分量误差与强度误差")
    if available:
        for index, (key, info) in enumerate(available, start=1):
            lines.append(f"#### 4.2.{index} {info['title']} MAE（{info['unit']}）")
            figure = figures.get(f"mae_{key}")
            if figure:
                figure_number += 1
                lines.extend([
                    f"![{info['title']} MAE](./{figure})",
                    f"图 {figure_number}：{info['title']} MAE 随时间变化；"
                    "阴影为 95% 台风分组区间，横轴标注各时效的配对起报数与台风数。",
                ])
            lines.extend([
                horizontal(info),
                difference_table(info) if comparison_order else "未提供对比模型。",
                "**小结**：" + (
                    _comparison_reading(info, focus, comparison_order[0], primary)
                    if comparison_order else _bias_reading(info, focus, primary)
                ),
            ])
        lines.append(f"#### 4.2.{len(available) + 1} 有符号偏差 ME")
        for key, info in available:
            figure = figures.get(f"me_{key}")
            if figure:
                figure_number += 1
                lines.extend([
                    f"![{info['title']} ME](./{figure})",
                    f"图 {figure_number}：{info['title']}有符号偏差随时间变化；"
                    "零线表示无平均偏差，阴影为 95% 台风分组区间。",
                ])
            lines.extend([
                f"**{info['title']} ME（{info['unit']}）**",
                horizontal(info, "me", with_ci=True), _bias_reading(info, focus, primary),
            ])
        if "minimum_pressure" in metrics:
            lines.append("最低气压正 Bias 表示预报气压偏高、强度偏弱；AT/CT 符号沿用源定义。")
    else:
        lines.append("没有有效共同分量/强度样本，不生成占位图表。")
    for key, info in metrics.items():
        if key != "track" and not any(row["n_case"] for row in info["by_lead"]):
            lines.append(f"不比较{info['title']}：所选时效缺少各模型一致有效字段/样本。")

    lines.append("### 4.3 逐台风解读")
    if storms:
        lines.append(
            "逐台风路径误差取该台风在主要时效内全部严格配对 (起报, 时效) 的平均大圆距离，"
            "各模型使用同一配对键、同一权重。"
            f"起报数低于 {summary['small_sample']['n_case_below']} 或主要时效配对样本少于 10 "
            "的台风只作个例观察。"
        )
        lines.append(_table(
            ["台风", "起报数", "主要时效配对样本"]
            + [f"{name} 平均路径误差（km）" for name in names] + ["最低"],
            [[storm["label"], storm["n_case"], storm["n_sample"]]
             + [_format(storm["means"].get(name)) for name in names]
             + [storm["best"]] for storm in storms],
        ))
        for storm in storms:
            values = [(name, storm["means"][name]) for name in names if name in storm["means"]]
            low = min(values, key=lambda item: item[1])
            high = max(values, key=lambda item: item[1])
            small = (storm["n_case"] < summary["small_sample"]["n_case_below"]
                     or storm["n_sample"] < 10)
            figure = figures.get(f"storm:{storm['key']}")
            if figure:
                figure_number += 1
                lines.extend([
                    f"![{storm['label']}](./{figure})",
                    f"图 {figure_number}：{storm['label']} 各模型的预报路径（细线，逐起报，"
                    f"0 至 {max(summary['leads']):g} h）与实况路径（粗黑线，圆点为逐时位置，"
                    "方块、叉号分别为起止点）。",
                ])
            lines.append(
                f"**{storm['label']}**：主要时效平均路径误差最低为 {low[0]}"
                f"（{_format(low[1])} km），最高为 {high[0]}（{_format(high[1])} km）；"
                f"{storm['n_case']} 个起报、{storm['n_sample']} 个配对样本。"
                + ("该台风只有少量样本，只作个例观察。" if small else "")
            )
    else:
        lines.append("没有有效共同路径样本，不生成逐台风表与路径图。")

    lines.extend([
        "## 5. 结论与限制",
        *conclusions, *biases,
        *([storm_sentence] if storm_sentence else []),
        "结论仅针对本批配对样本，不将均值排序自动解读为显著优势。",
        "同一台风多次起报相关；仅一个台风不能估计台风间不确定性。"
        "长时效小样本、诊断中心定义、成员有效性及观测标准均限制外推。",
        "集合平均误差不能代表集合概率性能；本报告未计算离散度、可靠性、CRPS 或能力雷达图。"
        "成员完成情况缺失时只能注明未知。",
        "## 6. 复现与产物",
        f"输出位置：{summary['output_dir']}",
        f"代码标识：{summary['implementation_sha256']}",
        f"依赖版本：{json.dumps(summary['versions'], ensure_ascii=False)}",
        "原始数据未修改。参数、输入路径、CSV SHA256、字段映射、"
        "元数据、配对键与排除计数均保存在 summary.json。",
        "实际执行命令/程序化调用：", f"```sh\n{command}\n```",
        "[机器可读汇总](./summary.json) · [统计长表](./statistics.csv) · "
        "[配对差值](./differences.csv)"
        + (" · [逐台风路径图](./track_plots/)" if any(
            f"storm:{storm['key']}" in figures for storm in storms) else ""),
    ])
    return "\n\n".join(lines) + "\n"


def build_typhoon_report(
    models: Sequence[Tuple[str, Path]], out_dir: Path, *,
    leads: Optional[Sequence[float]] = None, n_bootstrap: int = 2000,
    seed: int = 20250910, fixed_leads: Optional[Sequence[float]] = None,
    small_cases: int = 10, small_storms: int = 3,
    command: Optional[str] = None, forecast_type: Optional[str] = None,
    verify_legacy_members: bool = False,
    primary_model: Optional[str] = None, previous_model: Optional[str] = None,
    primary_leads: Optional[Sequence[float]] = None,
    exploratory_leads: Optional[Sequence[float]] = None,
) -> Dict[str, Path]:
    out_dir = Path(out_dir).resolve()
    if out_dir.exists() and (not out_dir.is_dir() or any(out_dir.iterdir())):
        raise ValueError(f"{out_dir}: 输出必须为新目录或空目录，不能覆盖已有产物")
    archives = [
        load_typhoon_archive(
            name, path, forecast_type=forecast_type, verify_legacy_members=verify_legacy_members
        ) for name, path in models
    ]
    if any(out_dir == item.root or item.root in out_dir.parents for item in archives):
        raise ValueError("报告输出不能位于原始输入目录内")
    summary = analyze_typhoon(
        archives, leads=leads, n_bootstrap=n_bootstrap, seed=seed,
        fixed_leads=fixed_leads, small_cases=small_cases, small_storms=small_storms,
    )
    names = summary["models"]
    focus = names[0] if primary_model is None else primary_model
    if focus not in names or (previous_model is not None and (
        previous_model not in names or previous_model == focus
    )):
        raise ValueError("主模型/旧版本须属于模型列表，且两者不能相同")
    selected = summary["leads"]
    exploratory = sorted(set(exploratory_leads or []))
    primary = sorted(set(primary_leads)) if primary_leads is not None else [
        row["lead_h"] for row in summary["metrics"]["track"]["by_lead"]
        if row["n_case"] and not row["small_sample"] and row["lead_h"] not in exploratory
    ]
    if not primary and primary_leads is None:
        primary = [row["lead_h"] for row in summary["metrics"]["track"]["by_lead"]
                   if row["n_case"] and row["lead_h"] not in exploratory]
    if not primary or not set(primary + exploratory).issubset(selected) or (
        set(primary) & set(exploratory)
    ):
        raise ValueError("主要/探索时效须包含在报告时效中且互不重叠，主要时效不能为空")
    summary["presentation"] = {
        "primary_model": focus, "previous_model": previous_model,
        "primary_leads": primary, "exploratory_leads": exploratory,
        "reference_leads": [lead for lead in selected if lead not in primary + exploratory],
        "primary_leads_source": "explicit" if primary_leads is not None else "sample_coverage",
    }
    summary["storm_table"] = _storm_table(archives, summary)
    import matplotlib
    import pandas as pd
    from reporting.reports.typhoon_plots import plot_typhoon

    summary["versions"] = {
        "python": sys.version.split()[0], "numpy": np.__version__,
        "pandas": pd.__version__, "matplotlib": matplotlib.__version__,
    }
    summary["implementation_sha256"] = hashlib.sha256(
        Path(__file__).read_bytes()
        + Path(__file__).with_name("typhoon_plots.py").read_bytes()
    ).hexdigest()
    summary["output_dir"] = str(out_dir)
    command = command or (
        f"build_typhoon_report({[(name, str(path)) for name, path in models]!r}, "
        f"{str(out_dir)!r}, leads={leads!r}, n_bootstrap={n_bootstrap}, seed={seed}, "
        f"fixed_leads={fixed_leads!r}, small_cases={small_cases}, small_storms={small_storms}, "
        f"forecast_type={forecast_type!r}, verify_legacy_members={verify_legacy_members!r}, "
        f"primary_model={primary_model!r}, previous_model={previous_model!r}, "
        f"primary_leads={primary_leads!r}, exploratory_leads={exploratory_leads!r})"
    )
    summary["command"] = command
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    # Stage all outputs together so errors do not leave success-shaped partial reports.
    with tempfile.TemporaryDirectory(prefix=".typhoon-report-", dir=out_dir.parent) as tmp:
        staging = Path(tmp)
        figures = plot_typhoon(summary, staging)
        (staging / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        statistics, differences = [], []
        for metric, info in summary["metrics"].items():
            for row in info["by_lead"]:
                common = {
                    "metric": metric, "statistic": info["statistic"],
                    "unit": info["unit"], "lead_h": row["lead_h"],
                    "n_case": row["n_case"], "n_storm": row["n_storm"],
                }
                for name, values in row["models"].items():
                    statistics.append({
                        **common, "model": name, "mean": values["mean"], "me": values["me"],
                        "ci_low": values["ci95"][0], "ci_high": values["ci95"][1],
                        "me_ci_low": values["me_ci95"][0], "me_ci_high": values["me_ci95"][1],
                    })
                for diff in row["differences"]:
                    differences.append({
                        **common, "first": diff["first"], "second": diff["second"],
                        "mean": diff["mean"], "me": diff["me"],
                        "ci_low": diff["ci95"][0], "ci_high": diff["ci95"][1],
                        "me_ci_low": diff["me_ci95"][0], "me_ci_high": diff["me_ci95"][1],
                    })
        pd.DataFrame(statistics).to_csv(staging / "statistics.csv", index=False)
        pd.DataFrame(differences, columns=[
            "metric", "statistic", "unit", "lead_h", "n_case", "n_storm", "first", "second",
            "mean", "me", "ci_low", "ci_high", "me_ci_low", "me_ci_high",
        ]).to_csv(staging / "differences.csv", index=False)
        (staging / "REPORT.md").write_text(
            render_typhoon_report(summary, figures, command), encoding="utf-8"
        )
        if out_dir.exists():
            out_dir.rmdir()
        shutil.move(str(staging), str(out_dir))
    LOG.info("台风报告生成完成：%s", out_dir)
    return {path.stem: path for path in sorted(out_dir.iterdir())}


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="台风场次 CSV -> 配对统计、图表与六章报告")
    parser.add_argument("--model", action="append", required=True, metavar="NAME=目录")
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--leads", type=float, nargs="+", help="小时；默认使用实际数据时效")
    parser.add_argument("--fixed-leads", type=float, nargs="+", help="固定完整路径样本窗口")
    parser.add_argument("--bootstrap", type=int, default=2000, help="重采样次数；0 禁用")
    parser.add_argument("--seed", type=int, default=20250910)
    parser.add_argument("--small-cases", type=int, default=10)
    parser.add_argument("--small-storms", type=int, default=3)
    parser.add_argument("--forecast-type", choices=["det", "ens"],
                        help="显式声明旧元数据缺少的预报类型；有冲突则报错")
    parser.add_argument("--verify-legacy-members", action="store_true",
                        help="用 *_members.csv 只读核验旧集合方案B、时间、观测与成员数")
    parser.add_argument("--primary-model", help="分析主模型；默认第一个 --model")
    parser.add_argument("--previous-model", help="主模型的旧版本显示名；不从名字猜测")
    parser.add_argument("--primary-leads", type=float, nargs="+", help="主要讨论时效")
    parser.add_argument("--exploratory-leads", type=float, nargs="+", help="探索性讨论时效")
    args = parser.parse_args(argv)
    models = []
    for value in args.model:
        name, separator, path = value.partition("=")
        if not separator or not name.strip() or not path.strip():
            parser.error("--model 必须为非空 NAME=目录")
        models.append((name.strip(), Path(path.strip())))
    actual = [sys.executable, *sys.argv] if argv is None else [
        "python", "-m", "reporting.reports.typhoon_report", *argv
    ]
    artifacts = build_typhoon_report(
        models, args.result, leads=args.leads, n_bootstrap=args.bootstrap,
        seed=args.seed, fixed_leads=args.fixed_leads,
        small_cases=args.small_cases, small_storms=args.small_storms,
        command=shlex.join(actual),
        forecast_type=args.forecast_type, verify_legacy_members=args.verify_legacy_members,
        primary_model=args.primary_model, previous_model=args.previous_model,
        primary_leads=args.primary_leads, exploratory_leads=args.exploratory_leads,
    )
    for name, path in artifacts.items():
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
