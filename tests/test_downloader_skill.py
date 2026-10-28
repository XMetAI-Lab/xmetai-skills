"""Test only the skill dispatcher, using stub downloaders."""

import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
SKILL = ROOT / "skills" / "dataset-downloader"
ENTRY = SKILL / "scripts/run_download.py"


def invoke(*args):
    return subprocess.run(
        [sys.executable, str(ENTRY), *map(str, args)],
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.fixture
def repository(tmp_path):
    root = tmp_path / "repository"
    root.mkdir()
    (root / "pyproject.toml").write_text("", encoding="utf-8")
    for source, directory in (
        ("cds", "cds_download"),
        ("marine", "copernicusmarine_download"),
    ):
        backend = root / "examples/dataset_downloader" / directory / f"{directory}.py"
        backend.parent.mkdir(parents=True)
        backend.write_text(
            "import json, os, sys\n"
            f"print(json.dumps({{'source': {source!r}, 'args': sys.argv[1:]}}))\n"
            "sys.exit(int(os.environ.get('DOWNLOADER_STUB_EXIT_CODE', '0')))\n",
            encoding="utf-8",
        )
    return root


def test_force_requires_execution(tmp_path):
    result = invoke("--source", "cds", tmp_path / "batch.yaml", "--force-redownload")
    assert result.returncode == 2
    assert "--force-redownload requires --execute" in result.stderr


def test_missing_repository(tmp_path):
    result = invoke("--source", "cds", tmp_path / "batch.yaml", "--repo-root", tmp_path)
    assert result.returncode == 2
    assert "Datakit repository not found" in result.stderr


def test_missing_batch(tmp_path, repository):
    result = invoke("--source", "marine", tmp_path / "missing.yaml", "--repo-root", repository)
    assert result.returncode == 2
    assert "Batch YAML not found" in result.stderr


@pytest.mark.parametrize("source", ["cds", "marine"])
@pytest.mark.parametrize("exit_code", [0, 7])
@pytest.mark.parametrize(
    ("flags", "forwarded"),
    [
        ([], ["--dry-run"]),
        (["--dry-run"], ["--dry-run"]),
        (["--plan-only"], ["--plan-only"]),
        (["--execute"], []),
        (["--execute", "--force-redownload"], ["--force-redownload"]),
    ],
)
def test_copied_entry_dispatches_to_explicit_repository(
    tmp_path, repository, monkeypatch, source, exit_code, flags, forwarded
):
    monkeypatch.setenv("DOWNLOADER_STUB_EXIT_CODE", str(exit_code))
    copied = tmp_path / "run_download.py"
    shutil.copyfile(ENTRY, copied)
    batch = tmp_path / "batch.yaml"
    batch.write_text("", encoding="utf-8")
    result = subprocess.run(
        [
            sys.executable, str(copied), "--source", source,
            "--repo-root", str(repository), str(batch), *flags,
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == exit_code, result.stderr
    assert json.loads(result.stdout) == {"source": source, "args": [str(batch), *forwarded]}


def test_repository_is_found_from_script_ancestors(tmp_path, repository):
    copied = repository / "skills/dataset-downloader/scripts/run_download.py"
    copied.parent.mkdir(parents=True)
    shutil.copyfile(ENTRY, copied)
    batch = tmp_path / "batch.yaml"
    batch.write_text("", encoding="utf-8")
    result = subprocess.run(
        [sys.executable, str(copied), "--source", "cds", str(batch)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {"source": "cds", "args": [str(batch), "--dry-run"]}


@pytest.mark.parametrize(
    "flags",
    [
        ["--execute", "--dry-run"],
        ["--execute", "--plan-only"],
        ["--dry-run", "--plan-only"],
    ],
)
def test_modes_are_mutually_exclusive(tmp_path, flags):
    result = invoke("--source", "cds", tmp_path / "batch.yaml", *flags)
    assert result.returncode == 2
    assert "not allowed with argument" in result.stderr
