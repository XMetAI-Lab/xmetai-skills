"""Tests for the evidence-based download report renderer."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest

pytest.importorskip("yaml")


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (
    ROOT
    / "skills"
    / "xmetai-datakit-skills"
    / "downloader"
    / "scripts"
    / "render_download_report.py"
)


def write_batch(tmp_path: Path) -> tuple[Path, Path]:
    profile_dir = tmp_path / "profiles"
    profile_dir.mkdir()
    profile = profile_dir / "temperature.yaml"
    profile.write_text(
        """dataset: reanalysis-era5-pressure-levels
product_type: [reanalysis]
variable: [temperature]
time: ["00:00", "06:00"]
pressure_level: ["500"]
area: [60, 40, -10, 150]
data_format: netcdf
download_format: unarchived
""",
        encoding="utf-8",
    )
    batch = tmp_path / "batch.yaml"
    batch.write_text(
        f"""profiles:
  - path: ./profiles/temperature.yaml
    output_subdir: era5
storage:
  root_path: {tmp_path}/data
period:
  start_month: "2024-01"
  end_month: "2024-01"
execution:
  max_workers: 1
  max_retries: 1
  retry_delay_seconds: 60
  continue_on_error: false
""",
        encoding="utf-8",
    )
    target = (
        tmp_path
        / "data"
        / "era5"
        / "temperature"
        / "2024"
        / "temperature_202401.nc"
    )
    return batch, target


def write_plan(
    tmp_path: Path,
    batch: Path,
    target: Path,
    *,
    plan_id: str = "plan-1",
    generated_at: str = "2026-10-28T02:21:24.488184+00:00",
    task_count: int = 1,
) -> tuple[Path, Path]:
    records = tmp_path / "data" / "_records" / "batch"
    records.mkdir(parents=True)
    plan = records / "plan.txt"
    plan.write_text(
        f"""# generated_at: {generated_at}
# plan_id: {plan_id}
# batch_task: {batch}
# task_count: {task_count}
{target}
""",
        encoding="utf-8",
    )
    return plan, records / "status.jsonl"


def event(
    target: Path,
    status: str,
    *,
    plan_id: str = "plan-1",
    task_id: str = "task-1",
    timestamp: str = "2026-10-28T02:21:25+00:00",
    attempt: int = 1,
    error: str | None = None,
) -> dict:
    record = {
        "timestamp": timestamp,
        "plan_id": plan_id,
        "task_id": task_id,
        "status": status,
        "target": str(target),
        "attempt": attempt,
    }
    if error is not None:
        record["error"] = error
    return record


def write_status(path: Path, records: list[dict]) -> None:
    path.write_text(
        "\n".join(json.dumps(record, ensure_ascii=False) for record in records) + "\n",
        encoding="utf-8",
    )


def invoke(batch: Path, *args: object) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), str(batch), *map(str, args)],
        capture_output=True,
        text=True,
        check=False,
    )


def payload(result: subprocess.CompletedProcess[str]) -> dict:
    return json.loads(result.stdout)


def test_completed_run_report(tmp_path: Path) -> None:
    batch, target = write_batch(tmp_path)
    _, status = write_plan(tmp_path, batch, target)
    target.parent.mkdir(parents=True)
    target.write_bytes(b"x" * 321)
    write_status(
        status,
        [
            event(target, "started", timestamp="2026-10-28T02:21:24.578067+00:00"),
            event(target, "completed", timestamp="2026-10-28T02:22:05.408057+00:00"),
        ],
    )

    result = invoke(
        batch,
        "--exit-code",
        "0",
        "--request-id",
        "req-123",
        "--report-time",
        "2026-10-28T02:22:20Z",
    )

    assert result.returncode == 0, result.stderr
    data = payload(result)
    assert data["mode"] == "execute"
    assert data["conclusion"] == "完成"
    assert data["completed"] == 1
    assert data["anomalies"] == []
    report = Path(data["report"])
    assert report.name == "download_report_20261028T022220000000Z.md"
    text = report.read_text(encoding="utf-8")
    assert "结论：完成" in text
    assert "reanalysis-era5-pressure-levels" in text
    assert "temperature" in text
    assert "500 hPa" in text
    assert "321" in text
    assert "req-123" in text
    assert "2024-01" in text


def test_failed_task_is_anomaly_and_error_is_sanitized(tmp_path: Path) -> None:
    batch, target = write_batch(tmp_path)
    _, status = write_plan(tmp_path, batch, target)
    write_status(
        status,
        [
            event(target, "started"),
            event(
                target,
                "failed",
                timestamp="2026-10-28T02:22:05+00:00",
                error="remote rejected token=secret-value",
            ),
        ],
    )

    result = invoke(
        batch,
        "--exit-code",
        "1",
        "--report-time",
        "2026-10-28T02:22:20Z",
        "--strict",
    )

    assert result.returncode == 1
    data = payload(result)
    assert data["conclusion"] == "失败"
    assert data["failed"] == 1
    assert any("本次失败" in item for item in data["anomalies"])
    text = Path(data["report"]).read_text(encoding="utf-8")
    assert "本次失败" in text
    assert "token=<redacted>" in text
    assert "secret-value" not in text


def test_historical_completion_is_reported_as_skipped(tmp_path: Path) -> None:
    batch, target = write_batch(tmp_path)
    _, status = write_plan(
        tmp_path,
        batch,
        target,
        generated_at="2026-10-28T02:21:24+00:00",
    )
    target.parent.mkdir(parents=True)
    target.write_bytes(b"x" * 10)
    write_status(
        status,
        [
            event(
                target,
                "completed",
                timestamp="2026-10-27T10:00:00+00:00",
            )
        ],
    )
    run_log = tmp_path / "run.log"
    run_log.write_text(
        """{
  "total": 1,
  "completed": 0,
  "skipped": 1,
  "failed": 0
}
""",
        encoding="utf-8",
    )

    result = invoke(
        batch,
        "--exit-code",
        "0",
        "--run-log",
        run_log,
        "--report-time",
        "2026-10-28T02:22:20Z",
    )

    assert result.returncode == 0, result.stderr
    data = payload(result)
    assert data["completed"] == 0
    assert data["skipped"] == 1
    assert data["conclusion"] == "完成"
    text = Path(data["report"]).read_text(encoding="utf-8")
    assert "历史完成并跳过" in text


def test_records_from_another_plan_are_ignored(tmp_path: Path) -> None:
    batch, target = write_batch(tmp_path)
    _, status = write_plan(tmp_path, batch, target)
    write_status(
        status,
        [
            event(
                target,
                "completed",
                plan_id="other-plan",
                timestamp="2026-10-28T02:22:05+00:00",
            )
        ],
    )

    result = invoke(
        batch,
        "--exit-code",
        "0",
        "--report-time",
        "2026-10-28T02:22:20Z",
    )

    assert result.returncode == 0, result.stderr
    data = payload(result)
    assert data["completed"] == 0
    assert data["unknown"] == 1
    assert data["conclusion"].startswith("无法确认")
    text = Path(data["report"]).read_text(encoding="utf-8")
    assert "未知（无本次状态记录）" in text


def test_preview_does_not_claim_download(tmp_path: Path) -> None:
    batch, target = write_batch(tmp_path)
    write_plan(tmp_path, batch, target)

    result = invoke(
        batch,
        "--report-time",
        "2026-10-28T02:22:20Z",
    )

    assert result.returncode == 0, result.stderr
    data = payload(result)
    assert data["mode"] == "preview"
    assert data["conclusion"] == "仅预览"
    text = Path(data["report"]).read_text(encoding="utf-8")
    assert "模式：离线预览" in text
    assert "未下载（仅预览）" in text
    assert "结论：完成" not in text


def test_report_filename_is_not_overwritten(tmp_path: Path) -> None:
    batch, target = write_batch(tmp_path)
    _, status = write_plan(tmp_path, batch, target)
    target.parent.mkdir(parents=True)
    target.write_bytes(b"x")
    write_status(status, [event(target, "completed")])

    first = invoke(
        batch,
        "--exit-code",
        "0",
        "--report-time",
        "2026-10-28T02:22:20Z",
    )
    second = invoke(
        batch,
        "--exit-code",
        "0",
        "--report-time",
        "2026-10-28T02:22:20Z",
    )

    assert first.returncode == 0, first.stderr
    assert second.returncode == 0, second.stderr
    first_report = Path(payload(first)["report"])
    second_report = Path(payload(second)["report"])
    assert first_report != second_report
    assert first_report.exists()
    assert second_report.exists()
    assert second_report.name.endswith("_1.md")


def test_zero_byte_and_partial_file_are_anomalies(tmp_path: Path) -> None:
    batch, target = write_batch(tmp_path)
    _, status = write_plan(tmp_path, batch, target)
    target.parent.mkdir(parents=True)
    target.write_bytes(b"")
    target.with_name(target.name + ".part").write_bytes(b"partial")
    write_status(status, [event(target, "completed")])

    result = invoke(
        batch,
        "--exit-code",
        "0",
        "--report-time",
        "2026-10-28T02:22:20Z",
        "--strict",
    )

    assert result.returncode == 1
    data = payload(result)
    assert any("零字节" in item for item in data["anomalies"])
    assert any("临时文件" in item for item in data["anomalies"])


def test_run_summary_conflict_blocks_completion(tmp_path: Path) -> None:
    batch, target = write_batch(tmp_path)
    _, status = write_plan(tmp_path, batch, target)
    target.parent.mkdir(parents=True)
    target.write_bytes(b"x")
    write_status(
        status,
        [
            event(
                target,
                "completed",
                timestamp="2026-10-28T02:22:05+00:00",
            )
        ],
    )
    run_log = tmp_path / "run.log"
    run_log.write_text(
        """{
  "total": 1,
  "completed": 0,
  "skipped": 0,
  "failed": 1
}
""",
        encoding="utf-8",
    )

    result = invoke(
        batch,
        "--exit-code",
        "0",
        "--run-log",
        run_log,
        "--report-time",
        "2026-10-28T02:22:20Z",
    )

    assert result.returncode == 0, result.stderr
    data = payload(result)
    assert data["conclusion"] == "无法确认（运行汇总与状态记录冲突）"
    assert any("不一致" in item for item in data["anomalies"])
