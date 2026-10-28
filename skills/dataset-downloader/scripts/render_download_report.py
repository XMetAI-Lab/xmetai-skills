#!/usr/bin/env python3
"""Render a download report draft from local evidence.

The renderer reads the Batch/Profile YAML, the downloader plan, the append-only
status JSONL, and optional captured run output. It does not read credentials,
contact remote services, download data, or validate file contents. The rendered
Markdown is a draft that the AI must review and supplement.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import yaml
except ModuleNotFoundError:  # pragma: no cover - exercised by the CLI error path
    yaml = None


STATUS_COMPLETED = "completed"
STATUS_FAILED = "failed"
STATUS_STARTED = "started"
STATUS_RETRYING = "retrying"


def parse_timestamp(value: Any) -> datetime | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def format_timestamp(value: datetime | None) -> str:
    if value is None:
        return "未记录"
    return value.astimezone(timezone.utc).isoformat(timespec="microseconds").replace(
        "+00:00", "Z"
    )


def normalize_path(value: str | Path) -> Path:
    return Path(value).expanduser().resolve(strict=False)


def load_yaml_file(path: Path) -> dict[str, Any]:
    if yaml is None:
        raise RuntimeError(
            "PyYAML is required to render the report; install the downloader requirements"
        )
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return data


def resolve_batch_path(value: str, batch_path: Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = (batch_path.parent / path).resolve(strict=False)
    return path


def compact(value: Any) -> str:
    if isinstance(value, list):
        return ", ".join(compact(item) for item in value)
    if isinstance(value, dict):
        return "; ".join(f"{key}: {compact(item)}" for key, item in value.items())
    return "" if value is None else str(value)


def as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


def cell(value: Any) -> str:
    return compact(value).replace("|", "\\|").replace("\n", " ").strip()


def sanitize_error(value: Any) -> str:
    text = str(value).replace("\n", " ").strip()
    text = re.sub(
        r"(?i)(key|token|password|authorization)\s*[=:]\s*\S+",
        r"\1=<redacted>",
        text,
    )
    return text[:300] + ("…" if len(text) > 300 else "")


@dataclass
class Plan:
    path: Path
    plan_id: str | None = None
    generated_at: datetime | None = None
    batch_task: str | None = None
    task_count: int | None = None
    targets: list[Path] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass
class StatusRecord:
    raw: dict[str, Any]
    timestamp: datetime | None
    status: str
    target: str
    task_id: str
    attempt: int | None


@dataclass
class ProfileInfo:
    index: int
    path: Path
    output_subdir: str | None
    data: dict[str, Any] | None
    error: str | None = None
    label_value: str | None = None

    @property
    def label(self) -> str:
        return self.label_value or self.path.name


@dataclass
class TaskState:
    target: Path
    profile_label: str | None
    state: str
    attempt: int | None
    error: str | None
    exists: bool
    size: int | None
    zero_byte: bool
    partials: list[Path]


@dataclass
class RunLogInfo:
    summary: dict[str, int] | None = None
    request_id: str | None = None
    final_status: str | None = None


def load_plan(path: Path) -> Plan:
    plan = Plan(path=path)
    if not path.is_file():
        raise FileNotFoundError(f"plan file not found: {path}")
    for line in path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text:
            continue
        if text.startswith("#"):
            body = text[1:].strip()
            if ":" not in body:
                continue
            key, value = body.split(":", 1)
            key = key.strip()
            value = value.strip()
            if key == "plan_id":
                plan.plan_id = value or None
            elif key == "generated_at":
                plan.generated_at = parse_timestamp(value)
            elif key == "batch_task":
                plan.batch_task = value or None
            elif key == "task_count":
                try:
                    plan.task_count = int(value)
                except ValueError:
                    plan.warnings.append("计划中的 task_count 不是整数")
            continue
        target = Path(text).expanduser()
        if not target.is_absolute():
            target = (path.parent / target).resolve(strict=False)
        plan.targets.append(target)

    if plan.plan_id is None:
        plan.warnings.append("计划缺少 plan_id")
    if plan.generated_at is None:
        plan.warnings.append("计划缺少可解析的 generated_at")
    if plan.task_count is None:
        plan.warnings.append("计划缺少 task_count")
    elif plan.task_count != len(plan.targets):
        plan.warnings.append(
            f"计划 task_count={plan.task_count} 与实际目标数 {len(plan.targets)} 不一致"
        )
    if not plan.targets:
        plan.warnings.append("计划中没有目标文件")
    return plan


def load_status(path: Path) -> tuple[list[StatusRecord], list[str]]:
    records: list[StatusRecord] = []
    warnings: list[str] = []
    if not path.is_file():
        return records, [f"状态记录不存在：{path}"]
    for number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
        except json.JSONDecodeError:
            warnings.append(f"状态记录第 {number} 行不是合法 JSON，已跳过")
            continue
        if not isinstance(raw, dict):
            warnings.append(f"状态记录第 {number} 行不是对象，已跳过")
            continue
        target = raw.get("target")
        task_id = raw.get("task_id")
        if not isinstance(target, str) or not isinstance(task_id, str):
            warnings.append(f"状态记录第 {number} 行缺少 target/task_id，已跳过")
            continue
        attempt = raw.get("attempt")
        if not isinstance(attempt, int):
            attempt = None
        records.append(
            StatusRecord(
                raw=raw,
                timestamp=parse_timestamp(raw.get("timestamp")),
                status=str(raw.get("status", "unknown")),
                target=target,
                task_id=task_id,
                attempt=attempt,
            )
        )
    return records, warnings


def load_profiles(batch: dict[str, Any], batch_path: Path) -> tuple[list[ProfileInfo], list[str]]:
    profiles: list[ProfileInfo] = []
    warnings: list[str] = []
    raw_profiles = batch.get("profiles")
    if not isinstance(raw_profiles, list) or not raw_profiles:
        return profiles, ["Batch 中没有 profiles 列表"]
    for index, item in enumerate(raw_profiles, start=1):
        if not isinstance(item, dict):
            warnings.append(f"profiles 第 {index} 项不是映射，已跳过")
            continue
        raw_path = item.get("path")
        if not isinstance(raw_path, str) or not raw_path.strip():
            warnings.append(f"profiles 第 {index} 项缺少 path，已跳过")
            continue
        profile_path = resolve_batch_path(raw_path, batch_path)
        output_subdir = item.get("output_subdir")
        if not isinstance(output_subdir, str):
            output_subdir = None
        try:
            data = load_yaml_file(profile_path)
        except (OSError, ValueError, RuntimeError) as error:
            profiles.append(
                ProfileInfo(
                    index=index,
                    path=profile_path,
                    output_subdir=output_subdir,
                    data=None,
                    error=str(error),
                )
            )
            warnings.append(f"Profile 无法读取：{profile_path}：{error}")
            continue
        profiles.append(
            ProfileInfo(
                index=index,
                path=profile_path,
                output_subdir=output_subdir,
                data=data,
            )
        )
    labels: dict[str, int] = {}
    for profile in profiles:
        labels[profile.path.name] = labels.get(profile.path.name, 0) + 1
    for profile in profiles:
        if labels[profile.path.name] > 1:
            profile.label_value = f"{profile.path.name} #{profile.index}"
    return profiles, warnings


def find_partial_files(target: Path) -> list[Path]:
    if not target.parent.is_dir():
        return []
    found: list[Path] = []
    for name in (target.name + ".part", target.name + ".part.nc"):
        candidate = target.with_name(name)
        if candidate.exists():
            found.append(candidate)
    for candidate in target.parent.glob(target.name + "*.part*"):
        if candidate.exists() and candidate not in found:
            found.append(candidate)
    return sorted(set(found))


def match_profile(
    target: Path, profiles: list[ProfileInfo], storage_root: Path | None
) -> ProfileInfo | None:
    if storage_root is None:
        return None
    for profile in profiles:
        if not profile.output_subdir:
            continue
        base = (storage_root / profile.output_subdir).resolve(strict=False)
        try:
            target.resolve(strict=False).relative_to(base)
        except ValueError:
            continue
        return profile
    return None


def evaluate_tasks(
    plan: Plan,
    records: list[StatusRecord],
    run_start: datetime | None,
    mode: str,
    profiles: list[ProfileInfo],
    storage_root: Path | None,
) -> tuple[list[TaskState], list[str]]:
    warnings: list[str] = []
    matching: list[StatusRecord] = []
    for record in records:
        record_plan = record.raw.get("plan_id")
        if plan.plan_id and record_plan and record_plan != plan.plan_id:
            continue
        if plan.plan_id and not record_plan:
            warnings.append("有状态记录缺少 plan_id，已纳入但无法确认是否属于本次计划")
        matching.append(record)

    by_target: dict[str, list[StatusRecord]] = {}
    for record in matching:
        by_target.setdefault(str(normalize_path(record.target)), []).append(record)

    plan_targets = {str(normalize_path(target)) for target in plan.targets}
    for record in matching:
        if str(normalize_path(record.target)) not in plan_targets:
            warnings.append(
                f"状态记录包含计划外目标：{record.target}"
            )
            break

    tasks: list[TaskState] = []
    for target in plan.targets:
        key = str(normalize_path(target))
        events = sorted(
            by_target.get(key, []),
            key=lambda item: item.timestamp
            or datetime.min.replace(tzinfo=timezone.utc),
        )
        undated = [item for item in events if item.timestamp is None]
        if run_start is None:
            current: list[StatusRecord] = []
            historical = [item for item in events if item.timestamp is not None]
            ambiguous = bool(events)
            ambiguous_reason = "计划缺少可解析的 generated_at"
        else:
            current = [
                item
                for item in events
                if item.timestamp is not None and item.timestamp >= run_start
            ]
            historical = [
                item
                for item in events
                if item.timestamp is not None and item.timestamp < run_start
            ]
            ambiguous = bool(undated) and not current
            ambiguous_reason = "状态记录缺少时间戳"
        latest_current = current[-1] if current else None
        historical_completed = any(
            item.status == STATUS_COMPLETED for item in historical
        )
        if undated:
            warnings.append(
                f"`{target}` 有缺少时间戳的状态记录，无法确认是否属于本次执行"
            )

        exists = target.is_file()
        size = target.stat().st_size if exists else None
        zero_byte = size == 0
        partials = find_partial_files(target)
        profile = match_profile(target, profiles, storage_root)
        profile_label = profile.label if profile else None

        if mode == "preview":
            if exists:
                state = "仅预览；文件已存在（未在本流程下载）"
            else:
                state = "未下载（仅预览）"
        elif latest_current is not None:
            if latest_current.status == STATUS_COMPLETED:
                state = "本次完成"
            elif latest_current.status == STATUS_FAILED:
                state = "本次失败"
            elif latest_current.status in (STATUS_STARTED, STATUS_RETRYING):
                state = f"未完成（最后状态：{latest_current.status}）"
            else:
                state = f"未知（最后状态：{latest_current.status}）"
        elif ambiguous:
            state = f"无法区分本次与历史（{ambiguous_reason}）"
        elif historical_completed and exists:
            state = "历史完成并跳过（本次无记录，推定）"
        elif historical_completed:
            state = "历史完成记录存在但目标文件缺失"
        elif exists:
            state = "文件已存在，本次无记录（状态未知）"
        else:
            state = "未知（无本次状态记录）"

        attempts = [item.attempt for item in events if item.attempt is not None]
        tasks.append(
            TaskState(
                target=target,
                profile_label=profile_label,
                state=state,
                attempt=max(attempts) if attempts else None,
                error=latest_current.raw.get("error") if latest_current else None,
                exists=exists,
                size=size,
                zero_byte=zero_byte,
                partials=partials,
            )
        )
    return tasks, warnings


def summarize(tasks: list[TaskState]) -> dict[str, int]:
    summary = {"total": len(tasks), "completed": 0, "skipped": 0, "failed": 0}
    for task in tasks:
        if task.state == "本次完成":
            summary["completed"] += 1
        elif task.state == "本次失败":
            summary["failed"] += 1
        elif task.state.startswith("历史完成并跳过"):
            summary["skipped"] += 1
    summary["unknown"] = (
        summary["total"]
        - summary["completed"]
        - summary["skipped"]
        - summary["failed"]
    )
    return summary


def parse_run_log(path: Path) -> tuple[RunLogInfo, list[str]]:
    info = RunLogInfo()
    warnings: list[str] = []
    if not path.is_file():
        return info, [f"运行日志不存在：{path}"]
    text = path.read_text(encoding="utf-8")
    decoder = json.JSONDecoder()
    required = {"total", "completed", "skipped", "failed"}
    for match in re.finditer(r"\{", text):
        try:
            value, _ = decoder.raw_decode(text[match.start() :])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and required <= set(value):
            try:
                info.summary = {key: int(value[key]) for key in required}
            except (TypeError, ValueError):
                warnings.append("运行日志中的最终汇总不是整数，已忽略")
    request = re.search(r"Request ID is ([0-9a-fA-F-]+)", text)
    if request:
        info.request_id = request.group(1)
    statuses = re.findall(r"status has been updated to (\w+)", text)
    if statuses:
        info.final_status = statuses[-1]
    if info.summary is None:
        warnings.append("运行日志中未找到 total/completed/skipped/failed 汇总")
    return info, warnings


def infer_mode(
    requested: str,
    run_log: RunLogInfo,
    exit_code: int | None,
    records: list[StatusRecord],
    run_start: datetime | None,
) -> str:
    if requested != "auto":
        return requested
    if run_log.summary is not None or run_log.request_id or exit_code is not None:
        return "execute"
    if run_start is not None and any(
        record.timestamp is not None and record.timestamp >= run_start
        for record in records
    ):
        return "execute"
    return "preview"


def decide_conclusion(
    mode: str,
    exit_code: int | None,
    summary: dict[str, int],
    unknown: int,
    conflict: bool,
) -> str:
    if mode == "preview":
        return "仅预览"
    if summary["total"] == 0:
        return "无法确认（计划中没有目标）"
    if conflict:
        return "无法确认（运行汇总与状态记录冲突）"
    if exit_code is None:
        return "无法确认（退出码未记录）"
    if exit_code != 0:
        if summary["completed"] == summary["total"]:
            return "无法确认（退出码非零但汇总显示全部完成）"
        if summary["completed"] > 0:
            return "部分完成"
        return "失败"
    if summary["failed"] > 0:
        return "部分完成" if summary["completed"] > 0 else "失败"
    if unknown > 0:
        return "无法确认（存在状态未知目标）"
    if summary["completed"] + summary["skipped"] == summary["total"]:
        return "完成"
    return "无法确认"


def source_label(profile: dict[str, Any], override: str | None) -> str:
    if override:
        return override.upper()
    dataset = str(profile.get("dataset") or profile.get("dataset_id") or "")
    lowered = dataset.lower()
    if "era5" in lowered:
        return "CDS"
    if lowered.startswith("cmems") or "oras5" in lowered:
        return "Marine"
    if "longitude" in profile or "latitude" in profile:
        return "Marine"
    return "未记录"


def date_range(period: dict[str, Any]) -> str:
    if "start_month" in period or "end_month" in period:
        start = period.get("start_month", "未记录")
        end = period.get("end_month", "未记录")
        return f"{start} 至 {end}（整月，首尾包含）"
    if "start_date" in period or "end_date" in period:
        start = period.get("start_date", "未记录")
        end = period.get("end_date", "未记录")
        return f"{start} 至 {end}（首尾包含）"
    return "未记录"


def time_semantics(profile: dict[str, Any]) -> str:
    if profile.get("time"):
        times = ", ".join(compact(item) for item in as_list(profile["time"]))
        return f"UTC 时次 {times}（瞬时抽样；数据集原生分辨率未核对）"
    if profile.get("statistics"):
        statistics = compact(profile["statistics"])
        frequency = profile.get("frequency", "未记录")
        time_zone = profile.get("time_zone", "未记录")
        return f"日统计 {statistics}；输入采样频率 {frequency}；日界 {time_zone}"
    return "未记录（数据集固有时间分辨率未核对）"


def level_or_depth(profile: dict[str, Any]) -> str:
    if profile.get("pressure_level"):
        levels = ", ".join(
            compact(item) for item in as_list(profile["pressure_level"])
        )
        return f"{levels} hPa"
    if profile.get("depth"):
        depth = profile["depth"]
        if isinstance(depth, dict):
            return f"{depth.get('minimum', '?')}–{depth.get('maximum', '?')} m"
        return f"{compact(depth)} m"
    return "不适用"


def geographic_scope(profile: dict[str, Any]) -> str:
    if "area" in profile:
        area = profile["area"]
        if area is None:
            return "不裁剪（全球）"
        if isinstance(area, list) and len(area) == 4:
            return f"北 {area[0]}、西 {area[1]}、南 {area[2]}、东 {area[3]}"
        return compact(area)
    if "longitude" in profile or "latitude" in profile:
        longitude = profile.get("longitude") or {}
        latitude = profile.get("latitude") or {}
        return (
            f"经度 {longitude.get('minimum', '?')}–{longitude.get('maximum', '?')}，"
            f"纬度 {latitude.get('minimum', '?')}–{latitude.get('maximum', '?')}"
        )
    return "未记录"


def output_format(profile: dict[str, Any]) -> str:
    parts = []
    for key in ("data_format", "download_format", "file_format", "format"):
        if profile.get(key):
            parts.append(f"{key}={profile[key]}")
    return "，".join(parts) if parts else "未记录"


def version_selection(profile: dict[str, Any]) -> str:
    parts = []
    for key in ("dataset_version", "version", "dataset_part", "service"):
        if profile.get(key):
            parts.append(f"{key}={profile[key]}")
    return (
        "，".join(parts)
        if parts
        else "未显式固定，由客户端选择；实际值未知"
    )


def profile_table(
    profile: ProfileInfo,
    batch: dict[str, Any],
    source: str | None,
    force_redownload: bool,
) -> list[str]:
    if profile.data is None:
        return [
            f"### {profile.label}",
            "",
            f"Profile 无法读取：{profile.error or '未知错误'}",
            "",
        ]
    data = profile.data
    period = batch.get("period") or {}
    execution = batch.get("execution") or {}
    rows = [
        ("来源与数据集 ID", f"{source_label(data, source)}；{data.get('dataset') or data.get('dataset_id') or '未记录'}", "Profile / 计划"),
        ("变量", compact(data.get("variable") or data.get("variables") or "未记录"), "Profile / 计划"),
        ("日期范围", date_range(period), "Batch / 计划"),
        ("时次与时间语义", time_semantics(data), "Profile / 计划"),
        ("气压层或深度", level_or_depth(data), "Profile / 计划"),
        ("地理范围", geographic_scope(data), "Profile / 计划"),
        ("格式", f"{output_format(data)}（未打开文件确认实际编码）", "请求 / 下载器规则"),
        ("版本、part、service", version_selection(data), "配置 / 实际元数据"),
        (
            "执行参数",
            f"max_workers={execution.get('max_workers', '未记录')}，"
            f"max_retries={execution.get('max_retries', '未记录')}，"
            f"retry_delay_seconds={execution.get('retry_delay_seconds', '未记录')}，"
            f"force_redownload={'是' if force_redownload else '否'}",
            "Batch / 本次命令",
        ),
    ]
    lines = [f"### {profile.label}", "", "| 项目 | 本次配置或实际请求 | 依据 |", "|---|---|---|"]
    for name, value, basis in rows:
        lines.append(f"| {name} | {cell(value)} | {basis} |")
    lines.append("")
    return lines


def render_report(
    batch_path: Path,
    batch: dict[str, Any],
    task_name: str | None,
    plan: Plan,
    status_path: Path,
    report_path: Path,
    profiles: list[ProfileInfo],
    tasks: list[TaskState],
    summary: dict[str, int],
    unknown: int,
    conclusion: str,
    mode: str,
    exit_code: int | None,
    run_start: datetime | None,
    run_end: datetime | None,
    run_log: RunLogInfo,
    request_id: str | None,
    datakit_root: str | None,
    source: str | None,
    force_redownload: bool,
    storage_root: Path | None,
    anomalies: list[str],
    warnings: list[str],
    report_time: datetime,
    run_summary_used: bool,
    summary_conflict: bool,
) -> str:
    mode_label = {"execute": "正式下载", "preview": "离线预览"}.get(mode, mode)
    if mode == "preview":
        time_range = "未记录（仅预览）"
    elif run_start is not None and run_end is not None:
        time_range = f"{format_timestamp(run_start)} 至 {format_timestamp(run_end)}"
    elif run_start is not None:
        time_range = f"{format_timestamp(run_start)} 起；结束时间未记录"
    else:
        time_range = "未记录"

    if run_summary_used:
        summary_basis = "本次运行输出中的最终汇总；逐目标状态由计划、状态记录和文件检查核对"
    else:
        summary_basis = "按计划目标与状态记录推定；未提供运行输出汇总"
    if summary_conflict:
        summary_basis += "；与推定值存在差异，见异常"

    unknown_tasks = [task for task in tasks if not (
        task.state == "本次完成"
        or task.state == "本次失败"
        or task.state.startswith("历史完成并跳过")
    )]
    unknown_counts: dict[str, int] = {}
    for task in unknown_tasks:
        unknown_counts[task.state] = unknown_counts.get(task.state, 0) + 1
    unknown_reason = "；".join(
        f"{state} × {count}" for state, count in unknown_counts.items()
    )
    if not unknown_reason and unknown > 0:
        unknown_reason = "运行汇总显示存在未完成或未知目标，但逐目标状态未定位"
    if not unknown_reason:
        unknown_reason = "无"

    lines: list[str] = [
        f"# 下载报告：{task_name or batch_path.stem}",
        "",
        "## 执行结果",
        "",
        f"- 报告生成时间（含时区）：{format_timestamp(report_time)}",
        f"- 本次执行起止时间（含时区）：{time_range}",
        f"- 模式：{mode_label}",
        f"- 结论：{conclusion}",
        f"- 进程退出码：{exit_code if exit_code is not None else '未知'}",
        f"- 本次汇总：total={summary['total']}，completed={summary['completed']}，"
        f"skipped={summary['skipped']}，failed={summary['failed']}",
        f"- 未完成或状态未知：{unknown}（{unknown_reason}）",
        f"- 汇总依据：{summary_basis}",
        "",
        "## 下载内容与请求参数",
        "",
    ]
    for profile in profiles:
        lines.extend(profile_table(profile, batch, source, force_redownload))
    if not profiles:
        lines.extend(["Batch 中没有可读取的 Profile。", ""])

    lines.extend(
        [
            "## 配置、记录与数据位置",
            "",
            f"- Datakit 根目录：{datakit_root or '未记录'}",
            f"- Batch：{batch_path}",
        ]
    )
    for profile in profiles:
        lines.append(f"- Profile：{profile.path}")
    lines.extend(
        [
            f"- 数据输出根目录：{storage_root if storage_root is not None else '未记录'}",
            f"- 本次计划：{plan.path}（plan_id：{plan.plan_id or '未记录'}）",
            f"- 状态记录：{status_path}",
            f"- 本报告：{report_path}",
            "",
            "| 目标文件绝对路径 | 所属 Profile | 状态与依据 | 当前存在 | 实际大小（字节） |",
            "|---|---|---|---|---|",
        ]
    )
    for task in tasks:
        profile_label = task.profile_label or "未归属"
        exists = "是" if task.exists else "否"
        size = str(task.size) if task.size is not None else "不存在"
        if task.zero_byte:
            size += "（零字节）"
        lines.append(
            f"| `{task.target}` | {cell(profile_label)} | {cell(task.state)} | {exists} | {size} |"
        )
    if not tasks:
        lines.append("| 计划中没有目标文件 | — | — | — | — |")
    lines.append("")

    lines.extend(
        [
            "## 检查范围与限制",
            "",
            "- 已检查：解析 Batch/Profile YAML、计划文件、状态记录（JSONL）、运行日志汇总（若提供）、目标文件存在性与大小、零字节及残留临时文件。",
            "- 文件内容检查：未执行；未打开 NetCDF/Zarr 检查变量、维度、时间轴、单位、缺测或网格。",
            "- 参数与实测的差异：未检查。",
            f"- 认证：{('请求 ID ' + request_id + ('；服务端最终状态 ' + run_log.final_status if run_log.final_status else '')) if request_id else '未记录'}；未读取或记录凭据内容。",
            "- 本报告不因请求成功或文件存在而宣称科学完整性验收通过。",
            "",
            "## 异常与后续操作",
            "",
        ]
    )
    if anomalies:
        lines.append("- 失败、缺失、零字节、临时文件或记录冲突：")
        for anomaly in anomalies:
            lines.append(f"  - {anomaly}")
    else:
        lines.append("- 失败、缺失、零字节、临时文件或记录冲突：未发现")
    if warnings:
        lines.append("- 其他提示：")
        for warning in warnings:
            lines.append(f"  - {warning}")
    lines.extend(
        [
            "- 恢复方式：使用原 Batch 与入口命令重新执行；默认保留完成记录并跳过已完成任务。仅在确认要覆盖时使用 `--force-redownload`。",
            "- 后续建议：如需确认文件内容，可另行检查变量名、维度、时间范围、层次和单位；该检查未在本流程中自动启动。",
            "",
            "> 本报告由 `scripts/render_download_report.py` 依据本地证据生成初稿；AI 须核对后定稿。脚本不读取凭据、不联网、不验证文件内容。",
            "",
        ]
    )
    return "\n".join(lines)


def choose_report_path(directory: Path, stamp: str) -> Path:
    candidate = directory / f"download_report_{stamp}.md"
    index = 1
    while candidate.exists():
        candidate = directory / f"download_report_{stamp}_{index}.md"
        index += 1
    return candidate


def write_report(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render a download report draft from local plan/status evidence."
    )
    parser.add_argument("batch", type=Path, help="Batch YAML used for the download")
    parser.add_argument("--plan", type=Path, help="plan.txt path; defaults to Batch records")
    parser.add_argument(
        "--status", type=Path, help="status.jsonl path; defaults to Batch records"
    )
    parser.add_argument("--report-dir", type=Path, help="directory for the Markdown report")
    parser.add_argument("--run-log", type=Path, help="captured run output with the final summary")
    parser.add_argument("--exit-code", type=int, help="actual process exit code")
    parser.add_argument("--request-id", help="CDS/Marine request ID seen in the run output")
    parser.add_argument(
        "--mode",
        choices=("auto", "execute", "preview"),
        default="auto",
        help="report mode; auto infers from evidence",
    )
    parser.add_argument("--run-start", help="ISO timestamp for this run start")
    parser.add_argument("--run-end", help="ISO timestamp for this run end")
    parser.add_argument(
        "--report-time",
        help="ISO timestamp override for the report name and header",
    )
    parser.add_argument("--datakit-root", help="Datakit repository root for the report")
    parser.add_argument("--task-name", help="report title; defaults to the Batch filename")
    parser.add_argument("--source", choices=("cds", "marine"), help="override source label")
    parser.add_argument("--force-redownload", action="store_true")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="exit 1 when evidence anomalies are found",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        batch_path = args.batch.expanduser().resolve()
        if not batch_path.is_file():
            raise FileNotFoundError(f"Batch YAML not found: {batch_path}")
        batch = load_yaml_file(batch_path)

        storage = batch.get("storage") or {}
        root_value = storage.get("root_path") if isinstance(storage, dict) else None
        storage_root = (
            resolve_batch_path(root_value, batch_path)
            if isinstance(root_value, str) and root_value
            else None
        )

        records = batch.get("records")
        if isinstance(records, dict):
            status_value = records.get("status_log_path") or records.get("event_log_path")
            if not isinstance(status_value, str):
                raise ValueError(
                    "records must contain status_log_path or event_log_path"
                )
            default_status = resolve_batch_path(status_value, batch_path)
            plan_value = records.get("plan_path") or records.get("target_list_path")
            default_plan = (
                resolve_batch_path(plan_value, batch_path)
                if isinstance(plan_value, str)
                else default_status.with_suffix(".plan.txt")
            )
        elif isinstance(batch.get("log_path"), str):
            default_status = resolve_batch_path(batch["log_path"], batch_path)
            default_plan = default_status.with_suffix(".plan.txt")
        elif storage_root is not None:
            records_root = storage_root / "_records" / batch_path.stem
            default_plan = records_root / "plan.txt"
            default_status = records_root / "status.jsonl"
        else:
            raise ValueError("storage.root_path is required when records are omitted")

        plan_path = (
            args.plan.expanduser().resolve() if args.plan else default_plan
        )
        status_path = (
            args.status.expanduser().resolve() if args.status else default_status
        )
        plan = load_plan(plan_path)
        status_records, status_warnings = load_status(status_path)
        profiles, profile_warnings = load_profiles(batch, batch_path)

        run_log = RunLogInfo()
        run_log_warnings: list[str] = []
        if args.run_log:
            run_log, run_log_warnings = parse_run_log(
                args.run_log.expanduser().resolve()
            )

        run_start = parse_timestamp(args.run_start) or plan.generated_at
        if args.run_start and parse_timestamp(args.run_start) is None:
            raise ValueError(f"invalid --run-start timestamp: {args.run_start}")
        run_end = parse_timestamp(args.run_end)
        if args.run_end and run_end is None:
            raise ValueError(f"invalid --run-end timestamp: {args.run_end}")

        mode = infer_mode(args.mode, run_log, args.exit_code, status_records, run_start)
        tasks, task_warnings = evaluate_tasks(
            plan, status_records, run_start, mode, profiles, storage_root
        )
        derived = summarize(tasks)
        if mode == "execute":
            current_timestamps = [
                record.timestamp
                for record in status_records
                if record.timestamp is not None
                and run_start is not None
                and record.timestamp >= run_start
            ]
            if run_end is None and current_timestamps:
                run_end = max(current_timestamps)

        warnings: list[str] = []
        warnings.extend(plan.warnings)
        warnings.extend(status_warnings)
        warnings.extend(profile_warnings)
        warnings.extend(run_log_warnings)
        warnings.extend(task_warnings)

        anomalies: list[str] = []
        for task in tasks:
            if task.state == "本次完成" and not task.exists:
                anomalies.append(f"`{task.target}` 有本次完成记录，但目标文件缺失")
            if task.state == "本次完成" and task.zero_byte:
                anomalies.append(f"`{task.target}` 有本次完成记录，但文件为零字节")
            if task.state == "历史完成记录存在但目标文件缺失":
                anomalies.append(f"`{task.target}` 有历史完成记录，但目标文件缺失")
            if task.partials:
                names = ", ".join(str(item) for item in task.partials)
                anomalies.append(f"`{task.target}` 存在残留临时文件：{names}")
            if task.state == "本次失败":
                detail = sanitize_error(task.error) if task.error else "未提供错误信息"
                anomalies.append(f"`{task.target}` 本次失败：{detail}")
                if task.exists:
                    warnings.append(
                        f"`{task.target}` 本次失败，但目标文件仍存在，可能为历史文件"
                    )
        if mode == "execute" and not status_path.is_file():
            anomalies.append(f"状态记录缺失：{status_path}")
        if mode == "execute" and args.exit_code is None:
            anomalies.append("未提供进程退出码，结论无法确认")
        if plan.task_count is not None and plan.task_count != len(plan.targets):
            warnings = [
                warning
                for warning in warnings
                if "task_count=" not in warning
            ]
            anomalies.append(
                f"计划 task_count={plan.task_count} 与实际目标数 {len(plan.targets)} 不一致"
            )
        if plan.batch_task and normalize_path(plan.batch_task) != batch_path:
            warnings.append(
                f"计划中的 batch_task 与当前 Batch 不一致：{plan.batch_task}"
            )

        summary = {
            "total": derived["total"],
            "completed": derived["completed"],
            "skipped": derived["skipped"],
            "failed": derived["failed"],
        }
        run_summary_used = run_log.summary is not None
        summary_conflict = False
        if run_summary_used:
            summary = {
                "total": run_log.summary["total"],
                "completed": run_log.summary["completed"],
                "skipped": run_log.summary["skipped"],
                "failed": run_log.summary["failed"],
            }
            for key in ("total", "completed", "skipped", "failed"):
                if derived[key] != summary[key]:
                    summary_conflict = True
                    anomalies.append(
                        f"运行汇总 {key}={summary[key]} 与状态记录推定值 {derived[key]} 不一致"
                    )
        unknown = (
            summary["total"]
            - summary["completed"]
            - summary["skipped"]
            - summary["failed"]
        )
        if unknown < 0:
            summary_conflict = True
            anomalies.append("运行汇总中的完成/跳过/失败数超过 total")
            unknown = derived["unknown"]

        conclusion = decide_conclusion(
            mode, args.exit_code, summary, unknown, summary_conflict
        )
        request_id = args.request_id or run_log.request_id
        parsed_report_time = parse_timestamp(args.report_time)
        if args.report_time and parsed_report_time is None:
            raise ValueError(f"invalid --report-time timestamp: {args.report_time}")
        report_time = parsed_report_time or datetime.now(timezone.utc)
        report_dir = (
            args.report_dir.expanduser().resolve()
            if args.report_dir
            else plan_path.parent
        )
        report_path = choose_report_path(
            report_dir, report_time.strftime("%Y%m%dT%H%M%S%fZ")
        )
        content = render_report(
            batch_path=batch_path,
            batch=batch,
            task_name=args.task_name,
            plan=plan,
            status_path=status_path,
            report_path=report_path,
            profiles=profiles,
            tasks=tasks,
            summary=summary,
            unknown=unknown,
            conclusion=conclusion,
            mode=mode,
            exit_code=args.exit_code,
            run_start=run_start,
            run_end=run_end,
            run_log=run_log,
            request_id=request_id,
            datakit_root=args.datakit_root,
            source=args.source,
            force_redownload=args.force_redownload,
            storage_root=storage_root,
            anomalies=anomalies,
            warnings=warnings,
            report_time=report_time,
            run_summary_used=run_summary_used,
            summary_conflict=summary_conflict,
        )
        write_report(report_path, content)
    except (FileNotFoundError, ValueError, RuntimeError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    print(
        json.dumps(
            {
                "report": str(report_path),
                "mode": mode,
                "conclusion": conclusion,
                "total": summary["total"],
                "completed": summary["completed"],
                "skipped": summary["skipped"],
                "failed": summary["failed"],
                "unknown": unknown,
                "exit_code": args.exit_code,
                "plan_id": plan.plan_id,
                "anomalies": anomalies,
                "warnings": warnings,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    if args.strict and anomalies:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
