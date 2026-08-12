from __future__ import annotations

import py_compile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SKILL = ROOT / "skills" / "xmetai-weather-modeling"


def test_required_skill_files_exist() -> None:
    assert (SKILL / "SKILL.md").exists()
    assert (SKILL / "agents" / "openai.yaml").exists()
    assert (SKILL / "references").is_dir()
    assert (SKILL / "scripts").is_dir()


def test_skill_frontmatter() -> None:
    text = (SKILL / "SKILL.md").read_text(encoding="utf-8")
    assert text.startswith("---\n")
    assert "name: xmetai-weather-modeling" in text
    assert "description:" in text
    assert "Zarr writes" in text


def test_referenced_files_exist() -> None:
    text = (SKILL / "SKILL.md").read_text(encoding="utf-8")
    for rel in [
        "references/zarr-static-contracts.md",
        "references/data-download-planning.md",
        "references/model-contracts.md",
        "references/inference-export-deploy.md",
        "references/shape-debugging.md",
    ]:
        assert rel in text
        assert (SKILL / rel).exists()


def test_data_download_routing() -> None:
    text = (SKILL / "SKILL.md").read_text(encoding="utf-8")
    assert "Data download planning" in text
    assert "download planning" in text.lower()
    assert "overwrite policy" in text.lower()
    assert "credentials" in text.lower()
    assert "stops before network requests" in text


def test_data_requirement_evidence_rules() -> None:
    text = (SKILL / "references" / "data-download-planning.md").read_text(encoding="utf-8")
    assert "### Evidence Classification" in text
    assert "### Config Isolation" in text
    assert "optional or pending items as mandatory downloads" in text
    assert "Do not merge requirements" in text
    assert "main dataset open path separately from sidecar-loading helpers" in text
    assert "Do not download" in text
    assert "Data analysis" in text


def test_data_download_plan_template() -> None:
    template = SKILL / "assets" / "templates" / "data_download_plan.md"
    assert template.exists()
    text = template.read_text(encoding="utf-8")
    assert "## Confirmed Requirements" in text
    assert "## Optional Requirements" in text
    assert "## Pending Confirmation" in text
    assert "## Preflight Status" in text
    assert "## Scope Statement" in text
    reference = (SKILL / "references" / "data-download-planning.md").read_text(encoding="utf-8")
    assert "assets/templates/data_download_plan.md" in reference


def test_data_download_cds_delivery_notes() -> None:
    text = (SKILL / "references" / "data-download-planning.md").read_text(encoding="utf-8")
    assert "download_format" in text
    assert "unarchived" in text
    assert "Duplicate value for month" in text


def test_data_preprocessing_container_notes() -> None:
    text = (SKILL / "references" / "data-preprocessing.md").read_text(encoding="utf-8")
    assert "ZIP" in text
    assert "edition" in text
    assert "filter_by_keys" in text


def test_inspect_data_format_zip_hint() -> None:
    script = (SKILL / "scripts" / "inspect_data_format.py").read_text(encoding="utf-8")
    assert "zipfile" in script
    assert "container" in script
    assert "contains" in script


def test_scripts_compile() -> None:
    for script in (SKILL / "scripts").glob("*.py"):
        py_compile.compile(str(script), doraise=True)
