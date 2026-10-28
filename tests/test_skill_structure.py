from __future__ import annotations

import py_compile
from pathlib import Path
import re

import pytest


ROOT = Path(__file__).resolve().parents[1]
SKILL = ROOT / "skills" / "xmetai-weather-modeling"
PEER_SKILLS = (
    "dataset-downloader",
    "evaluation-reporting",
    "xmetai-weather-modeling",
)


def test_flat_skill_layout_and_router() -> None:
    skills = ROOT / "skills"
    assert sorted(path.name for path in skills.iterdir() if path.is_dir()) == sorted(
        PEER_SKILLS
    )
    router = (skills / "SKILL.md").read_text(encoding="utf-8")
    assert re.search(r"^name: skills$", router, re.MULTILINE)
    assert "description:" in router
    for name in PEER_SKILLS:
        assert f"(./{name}/SKILL.md)" in router


@pytest.mark.parametrize("name", PEER_SKILLS)
def test_peer_skill_metadata(name: str) -> None:
    skill = ROOT / "skills" / name
    text = (skill / "SKILL.md").read_text(encoding="utf-8")
    assert text.startswith("---\n")
    assert re.search(rf"^name: {re.escape(name)}$", text, re.MULTILINE)
    assert "description:" in text
    for metadata in (skill / "agents").glob("*.yaml"):
        content = metadata.read_text(encoding="utf-8")
        # Modeling metadata uses the host's interface schema without a name.
        declared_name = re.search(r"^name: (.+)$", content, re.MULTILINE)
        if declared_name:
            assert declared_name.group(1) == name


@pytest.mark.parametrize("name", ("", *PEER_SKILLS))
def test_skill_frontmatter_discovery_contract(name: str) -> None:
    yaml = pytest.importorskip("yaml")
    skill = ROOT / "skills" / name
    text = (skill / "SKILL.md").read_text(encoding="utf-8")
    metadata = yaml.safe_load(text.split("---", 2)[1])
    assert metadata["name"] == skill.name
    assert isinstance(metadata["description"], str)
    assert 0 < len(metadata["description"]) <= 1024


def documentation_files() -> list[Path]:
    return [
        path for path in ROOT.rglob("*.md")
        if ".git" not in path.parts
        and (path.name == "SKILL.md" or path.name.lower() == "readme.md")
        and ".pytest_cache" not in path.parts
    ]


@pytest.mark.parametrize("path", documentation_files(), ids=lambda path: str(path.relative_to(ROOT)))
def test_skill_and_readme_language(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    assert not re.search(r"[\u3400-\u4dbf\u4e00-\u9fff]", text), (
        f"English documentation contains Chinese text: {path.relative_to(ROOT)}"
    )


@pytest.mark.parametrize("path", documentation_files(), ids=lambda path: str(path.relative_to(ROOT)))
def test_skill_and_readme_relative_links(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    for target in re.findall(r"\[[^\]]*\]\(([^)]+)\)", text):
        if re.match(r"[a-zA-Z][a-zA-Z0-9+.-]*:", target) or target.startswith("#"):
            continue
        destination = target.split("#", 1)[0]
        assert (path.parent / destination).exists(), (
            f"Broken relative link in {path.relative_to(ROOT)}: {target}"
        )


def test_reporting_package_discovery() -> None:
    skill = ROOT / "skills" / "evaluation-reporting"
    assert (skill / "scripts" / "reporting" / "__init__.py").is_file()
    project = (skill / "pyproject.toml").read_text(encoding="utf-8")
    assert 'include = ["reporting*"]' in project
    requirements = (ROOT / "requirements.txt").read_text(encoding="utf-8")
    assert "-e ./skills/evaluation-reporting[test]" in requirements


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
        "references/model-contracts.md",
        "references/inference-export-deploy.md",
        "references/shape-debugging.md",
    ]:
        assert rel in text
        assert (SKILL / rel).exists()


def test_scripts_compile() -> None:
    for script in (SKILL / "scripts").glob("*.py"):
        py_compile.compile(str(script), doraise=True)
