"""Validate the copied skill without access to the original repository packages."""

import csv
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SKILL_ROOT = REPO_ROOT / "skills" / "evaluation-reporting"

# 运行库需要科学计算栈；裸环境（只装了 pytest）直接跳过，CI 装齐全后再真正执行。
pytest.importorskip("numpy")
pytest.importorskip("pandas")
pytest.importorskip("xarray")
pytest.importorskip("matplotlib")
pytest.importorskip("scipy")

ENTRYPOINTS = sorted(
    path.name for path in (SKILL_ROOT / "scripts").glob("*.py")
    if not path.name.startswith("_")
)
IMPORT_GUARD = """
import importlib.abc
import sys

class BlockRepositoryPackages(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {"datakit", "pipelines"}:
            raise ImportError("Standalone skill imported repository package: " + fullname)

sys.meta_path.insert(0, BlockRepositoryPackages())
"""


@pytest.fixture(scope="module")
def standalone(tmp_path_factory):
    root = tmp_path_factory.mktemp("standalone-skill")
    skill = root / ".claude" / "skills" / "evaluation-reporting"
    shutil.copytree(
        SKILL_ROOT, skill,
        ignore=shutil.ignore_patterns("__pycache__", "*.egg-info", ".pytest_cache", "build"),
    )
    (skill / "pyproject.toml").unlink()
    (root / "pyproject.toml").write_text("[project]\nname = 'unrelated'\n")
    work = root / "unrelated-working-directory"
    work.mkdir()
    return skill, work


def run_isolated(standalone, code, *args):
    skill, work = standalone
    completed = subprocess.run(
        [sys.executable, "-I", "-c", IMPORT_GUARD + code, str(skill), *map(str, args)],
        cwd=work, capture_output=True, text=True, encoding="utf-8", timeout=120,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    return completed


def run_entrypoint(standalone, name, *args):
    return run_isolated(
        standalone,
        """
from pathlib import Path
import runpy

skill = Path(sys.argv[1])
script = skill / "scripts" / sys.argv[2]
sys.path.insert(0, str(script.parent))
sys.argv = [str(script), *sys.argv[3:]]
runpy.run_path(str(script), run_name="__main__")
""",
        name, *args,
    )


def test_all_runtime_modules_are_self_contained(standalone):
    run_isolated(
        standalone,
        """
from pathlib import Path
import importlib
import pkgutil

skill = Path(sys.argv[1])
sys.path.insert(0, str(skill / "scripts"))
import reporting

modules = list(pkgutil.walk_packages(
    reporting.__path__, reporting.__name__ + "."
))
assert len(modules) >= 30
for info in modules:
    module = importlib.import_module(info.name)
    assert Path(module.__file__).resolve().is_relative_to(skill)
assert not any(
    name.split(".")[0] in {"datakit", "pipelines"} for name in sys.modules
)
""",
    )


@pytest.mark.parametrize("name", ENTRYPOINTS)
def test_cli_help_outside_repository(standalone, name):
    completed = run_entrypoint(standalone, name, "--help")
    assert "usage:" in completed.stdout.lower()


def test_field_report_creates_markdown_and_figures(standalone, tmp_path):
    source = tmp_path / "results"
    source.mkdir()
    manifest = {
        "run_id": "standalone-smoke",
        "resolved_config": {"pipeline": "weather_field_scores"},
        "metrics": ["rmse", "acc", "activity"],
    }
    (source / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    columns = ["lead_h", "variable", "metric", "value", "status", "unit"]
    with (source / "scores.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for lead in (24, 48, 72):
            for metric, value in (
                ("rmse", lead / 24),
                ("acc", 1 - lead / 1000),
                ("activity_ratio", 1 + lead / 1000),
            ):
                writer.writerow({
                    "lead_h": lead, "variable": "z500", "metric": metric,
                    "value": value, "status": "success",
                    "unit": "gpm" if metric == "rmse" else "1",
                })
    before = {path.name: path.read_bytes() for path in source.iterdir()}
    output = tmp_path / "report"
    run_entrypoint(
        standalone, "render_report.py", source, "--result", output, "--model", "Smoke",
    )
    report = output / "REPORT.md"
    assert report.is_file()
    text = report.read_text(encoding="utf-8")
    assert "Smoke" in text and "RMSE" in text
    figures = list(output.rglob("*.png"))
    assert figures and all(path.stat().st_size > 100 for path in figures)
    links = re.findall(r"!\[[^\]]*\]\(([^)]+)\)", text)
    assert links
    assert all((report.parent / link).is_file() for link in links)
    assert before == {path.name: path.read_bytes() for path in source.iterdir()}


def test_ts_report_creates_svg_figures(standalone, tmp_path):
    source = tmp_path / "results"
    diagnostics = source / "diagnostics"
    diagnostics.mkdir(parents=True)
    manifest = {
        "run_id": "standalone-ts",
        "resolved_config": {"pipeline": "weather_ts_det"},
        "metrics": ["ts"],
    }
    (source / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with (diagnostics / "categorical_wide.csv").open(
        "w", newline="", encoding="utf-8",
    ) as stream:
        writer = csv.writer(stream)
        writer.writerow(["lead_h", "grade", "TS", "POD", "FAR", "BIAS"])
        for lead in (24, 48, 72):
            for grade in ("0.1", "10", "25"):
                writer.writerow([lead, grade, 0.5, 0.75, 0.4, 1.25])
    output = tmp_path / "report"
    run_entrypoint(standalone, "render_report.py", source, "--result", output)
    report = output / "REPORT.md"
    assert report.is_file() and "TS" in report.read_text(encoding="utf-8")
    figures = list(output.rglob("*.svg"))
    assert figures and all(path.stat().st_size > 100 for path in figures)


def test_missing_manifest_is_an_explicit_error(standalone, tmp_path):
    run_isolated(
        standalone,
        """
from pathlib import Path
sys.path.insert(0, str(Path(sys.argv[1]) / "scripts"))
from reporting.dispatch import build_report

source, target = map(Path, sys.argv[2:])
try:
    build_report(source, out_dir=target)
except ValueError as error:
    assert "manifest.json" in str(error)
else:
    raise AssertionError("Missing manifest was silently accepted")
assert not target.exists()
""",
        tmp_path, tmp_path / "report",
    )
