# -*- coding: utf-8 -*-
"""统一结果落盘。

一次评测的核心产物**只有两个文件**：

    results/<run_id>/
      scores.csv                  统一长表（唯一评分契约）
      manifest.json               运行记录：输入文件、组件版本、状态、产物索引、完整解析后配置

其余视图（覆盖率、列联表明细、分类检验宽表、JSON 快照）都是长表的派生视图，
由配置的 ``writers`` 声明才写出；派生视图只读长表，不重新计算指标。
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from reporting.sources.contracts import MetricResult
from reporting.sources.errors import OutputError
from reporting.sources.registry import ComponentType, get_registry
from reporting.core.tables import (
    COVERAGE_COLUMNS,
    DETAIL_COLUMNS,
    SCORE_COLUMNS,
    ResultTables,
    build_tables,
)


@dataclass
class RunContext:
    """一次评测运行的稳定标识，会写进长表的每一行。"""

    run_id: str
    model_id: str = ""
    dataset_id: str = ""
    protocol_id: str = ""


def _to_plain(value: Any) -> Any:
    """把配置对象转成可序列化的普通结构。"""
    if is_dataclass(value) and not isinstance(value, type):
        return _to_plain(asdict(value))
    if isinstance(value, dict):
        return {str(key): _to_plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_to_plain(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (datetime,)):
        return value.isoformat()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


class ReportWriter:
    """把统一的 ``MetricResult`` 集合落成标准产物。"""

    def __init__(
        self,
        output_dir: Path,
        context: RunContext,
        defaults: Optional[Dict[str, Any]] = None,
    ):
        """
        Args:
            output_dir: 产物根目录（通常是 results/<run_id>）。
            context: 运行标识。
            defaults: 结果未携带坐标时的兜底值。
        """
        self.output_dir = Path(output_dir)
        self.context = context
        self.defaults = dict(defaults or {})

    def write(
        self,
        results: Sequence[MetricResult],
        *,
        manifest: Optional[Dict[str, Any]] = None,
        resolved_config: Optional[Any] = None,
        writers: Sequence[str] = ("csv_long",),
    ) -> Dict[str, Path]:
        """写入统一长表与运行记录。

        Args:
            results: 指标结果集合。
            manifest: 额外的运行记录（输入文件、集合规模等）。
            resolved_config: 完整解析后的配置，写进 manifest.json 的 ``resolved_config`` 字段。
            writers: 需要额外写入的派生视图名（调用方须先在组件注册表中登记）。

        Returns:
            产物名 -> 路径

        Raises:
            OutputError: 没有可写出的结果。
        """
        results = list(results)
        if not results:
            raise OutputError("没有可写出的指标结果")

        self.output_dir.mkdir(parents=True, exist_ok=True)
        tables = build_tables(
            results,
            run_id=self.context.run_id,
            model_id=self.context.model_id,
            dataset_id=self.context.dataset_id,
            protocol_id=self.context.protocol_id,
            defaults=self.defaults,
        )

        # 核心产物：统一长表。其余视图（覆盖率、列联表明细、宽表、JSON…）
        # 一律由配置里的 writers 决定，不做无条件的"全量输出"。
        artifacts: Dict[str, Path] = {}
        artifacts["scores"] = self._write_frame(tables.score_table, self.output_dir / "scores.csv")

        for writer_name in writers:
            if writer_name == "csv_long":
                continue
            artifacts[writer_name] = self._run_writer(writer_name, tables)

        artifacts["manifest"] = self._write_manifest(
            results,
            tables,
            artifacts,
            manifest=manifest,
            resolved_config=resolved_config,
        )
        return artifacts

    def _write_frame(self, frame, path: Path) -> Path:
        try:
            frame.to_csv(path, index=False, encoding="utf-8")
        except OSError as exc:
            raise OutputError(f"结果写入失败: {path}: {exc}") from exc
        return path

    def _run_writer(self, writer_name: str, tables: ResultTables) -> Path:
        registry = get_registry()
        try:
            writer = registry.build(ComponentType.WRITER, writer_name)
        except Exception as exc:  # ConfigError 等
            available = ", ".join(registry.names(ComponentType.WRITER))
            raise OutputError(
                f"未知的输出视图 '{writer_name}'；已注册: {available}"
            ) from exc
        return writer(tables, self.context, self.output_dir)

    def _write_manifest(
        self,
        results: Sequence[MetricResult],
        tables: ResultTables,
        artifacts: Dict[str, Path],
        *,
        manifest: Optional[Dict[str, Any]],
        resolved_config: Any = None,
    ) -> Path:
        statuses: Dict[str, int] = {}
        input_files: List[str] = []
        for result in results:
            statuses[result.status.value] = statuses.get(result.status.value, 0) + 1
            provenance = result.provenance or {}
            for path in provenance.get("input_files", []) or []:
                text = str(path)
                if text not in input_files:
                    input_files.append(text)

        payload: Dict[str, Any] = {
            "run_id": self.context.run_id,
            "model_id": self.context.model_id,
            "dataset_id": self.context.dataset_id,
            "protocol_id": self.context.protocol_id,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "n_results": len(results),
            "metrics": sorted({result.metric_name for result in results}),
            "statuses": statuses,
            "metric_versions": sorted({result.metric_version for result in results}),
            "scores_rows": int(len(tables.scores)),
            "details_rows": int(len(tables.details)),
            "score_columns": list(SCORE_COLUMNS),
            "coverage_columns": list(COVERAGE_COLUMNS),
            "detail_columns": list(DETAIL_COLUMNS),
            "input_files": input_files,
            "artifacts": {name: str(path) for name, path in artifacts.items()},
        }
        if manifest:
            payload.update(_to_plain(manifest))
        if resolved_config is not None:
            payload["resolved_config"] = _to_plain(resolved_config)

        path = self.output_dir / "manifest.json"
        try:
            with path.open("w", encoding="utf-8") as stream:
                json.dump(payload, stream, indent=2, ensure_ascii=False, default=str)
        except OSError as exc:
            raise OutputError(f"运行记录写入失败: {path}: {exc}") from exc
        return path
