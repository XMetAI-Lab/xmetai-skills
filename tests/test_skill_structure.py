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
    assert "After confirmation, exit this route" in text
    assert "Data analysis" in text
    assert "code path wins" in text
    assert "normalization storage convention" in text


def test_data_download_plan_template() -> None:
    template = SKILL / "assets" / "templates" / "data_download_plan.md"
    assert template.exists()
    text = template.read_text(encoding="utf-8")
    assert "Bottom line" in text
    assert "## Download List" in text
    assert "Purpose" in text
    assert "### Main table" in text
    assert "Source dataset" in text
    assert "Conversion step" in text
    assert "Static fields / sidecars" in text
    assert "only when the model contract requires" in text
    assert "### Download & conversion" in text
    assert "Train" in text
    assert "Validation / test" in text
    assert "Static field acquisition notes" in text
    assert "## Data Source" in text
    assert "## Pending Confirmation" in text
    assert "## Next Step" in text
    reference = (SKILL / "references" / "data-download-planning.md").read_text(encoding="utf-8")
    assert "assets/templates/data_download_plan.md" in reference
    assert "Keep the printed plan short" in reference
    assert "evidence" in reference
    assert "inference configs" in reference.lower()
    assert "Statistics sidecars (`mean`, `std`, `weight`) are computed" in reference
    assert "## Static Fields Acquisition" in reference


def test_data_download_cds_delivery_notes() -> None:
    text = (SKILL / "references" / "data-download-planning.md").read_text(encoding="utf-8")
    assert "download_format" in text
    assert "unarchived" in text
    assert "Duplicate value for month" in text
    assert "Source Retrieval Guidance" in text
    assert "preferring official sources" in text


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


def test_convert_grib_fallback_and_layout() -> None:
    script = (SKILL / "scripts" / "convert_to_zarr.py").read_text(encoding="utf-8")
    assert "indexpath" in script
    assert "merge_to_data" in script
    assert "split_levels" in script
    assert "log1p" in script
    assert "normalize" in script
    assert "write_sidecars" in script
    assert "flatten_step" in script
    assert "shortName" in script
    assert "filter_by_keys" in script
    conversion = SKILL / "scripts" / "conversion"
    assert {
        "inputs.py",
        "grid.py",
        "temporal.py",
        "transforms.py",
        "batching.py",
        "statistics.py",
        "state.py",
        "writers.py",
        "static.py",
    } <= {path.name for path in conversion.glob("*.py")}
    state = (conversion / "state.py").read_text(encoding="utf-8")
    assert 'parent.parent / "zarr_write_guard.py"' in state
    insp = (SKILL / "scripts" / "inspect_data_format.py").read_text(encoding="utf-8")
    assert "indexpath" in insp
    text = (SKILL / "references" / "data-preprocessing.md").read_text(encoding="utf-8")
    assert "merge_to_data" in text
    assert "split_levels" in text
    assert "Normalization Convention" in text
    assert "log-transformed" in text
    assert "mean.nc" in text
    assert "std.nc" in text
    assert "weight.nc" in text
    assert "Training consumes normalized Zarr values directly" in text
    assert "exported ONNX" in text
    assert "verify the selected model's export/inference path" in text
    assert "Directory Layout" in text
    assert "indexpath" in text
    plan = (SKILL / "references" / "data-download-planning.md").read_text(encoding="utf-8")
    assert "Print scripts/configs instead of saving them" in plan


def test_evaluation_scripts_are_split_behind_stable_facades() -> None:
    scripts = SKILL / "scripts"
    evaluate = (scripts / "evaluate_pred.py").read_text(encoding="utf-8")
    visualize = (scripts / "visualize_eval.py").read_text(encoding="utf-8")
    assert "from evaluation.runner import compute_all_metrics" in evaluate
    assert "from evaluation.visualization.maps import" in visualize

    package = scripts / "evaluation"
    expected = {
        "io.py",
        "common.py",
        "runner.py",
        "report.py",
        "metrics/rmse.py",
        "metrics/contingency.py",
        "metrics/tcc.py",
        "metrics/ps.py",
        "metrics/ips.py",
        "visualization/common.py",
        "visualization/maps.py",
        "visualization/composite.py",
        "visualization/curves.py",
        "visualization/animation.py",
    }
    actual = {
        path.relative_to(package).as_posix()
        for path in package.rglob("*.py")
    }
    assert expected <= actual


def test_sidecar_generation_documented() -> None:
    script = SKILL / "scripts" / "compute_sidecars.py"
    assert script.exists()
    assert (SKILL / "scripts" / "merge_normalize.py").exists()
    text = (SKILL / "references" / "data-preprocessing.md").read_text(encoding="utf-8")
    assert "compute_sidecars.py" in text
    assert "before normalization" in text
    assert "zero variance" in text
    assert "Prepare required static fields separately as `const.nc`" in text
    assert "merge_normalize.py" in text


def test_scripts_compile() -> None:
    for script in (SKILL / "scripts").glob("*.py"):
        py_compile.compile(str(script), doraise=True)
