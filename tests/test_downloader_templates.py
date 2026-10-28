"""Check filled templates and opt-in compatibility with real Datakit builders.

Requires pytest and PyYAML; without PyYAML this entire module is skipped.
Local template checks do not require Datakit, credentials, or network access.
Eight compatibility cases (seven profiles and one multi-statistic case) are
skipped unless XMETAI_DATAKIT_ROOT points to a real Datakit repository.
That repository must contain both CDS and Copernicus Marine downloader modules,
and their import dependencies must be installed in the pytest environment.
An invalid configured path or missing builder dependencies causes an error,
not a skip. Compatibility checks build tasks and dry-run; they do not download.

Run from the xmetai-skills repository root (POSIX shell):

    XMETAI_DATAKIT_ROOT=/home/lenovo/workspace_1/xmetai-datakit_main/xmetai-datakit \\
      conda run -n xmetai-skills \\
      python -m pytest tests/test_downloader_templates.py -q -rs

The Datakit path above is a local example; replace it for other machines.
"""

import importlib.util
import os
from pathlib import Path
import subprocess
import sys

import pytest


yaml = pytest.importorskip("yaml")
ROOT = Path(__file__).resolve().parents[1]
SKILL = ROOT / "skills" / "xmetai-datakit-skills" / "downloader"
TEMPLATES = SKILL / "assets" / "templates"
CASES = [
    ("cds_hourly_single", "2m_temperature"),
    ("cds_hourly_pressure", "temperature"),
    ("cds_hourly_land", "soil_temperature_level_1"),
    ("cds_daily_single", "2m_temperature"),
    ("cds_daily_pressure", "temperature"),
    # This checks local structure only, not remote ERA5-Land daily availability.
    ("cds_daily_land", "soil_temperature_level_1"),
    ("marine_oras5", "thetao_oras"),
]


@pytest.fixture(params=CASES, ids=[name for name, _ in CASES])
def filled_profile(request):
    name, variable = request.param
    text = (TEMPLATES / f"{name}_profile.yaml").read_text(encoding="utf-8")
    replacements = {
        "REPLACE_WITH_CDS_VARIABLE": variable,
        "REPLACE_WITH_MARINE_VARIABLE": variable,
        "REPLACE_WITH_HH_00": '"06:00"',
        "REPLACE_WITH_PRESSURE_HPA": '"500"',
        "REPLACE_NORTH": "60",
        "REPLACE_WEST": "70",
        "REPLACE_SOUTH": "10",
        "REPLACE_EAST": "140",
        "REPLACE_WITH_DATA_FORMAT": "netcdf",
        "REPLACE_WITH_DOWNLOAD_FORMAT": "unarchived",
        "REPLACE_WITH_DAILY_STATISTIC": "daily_mean",
        "REPLACE_WITH_FREQUENCY": "1_hourly",
        "REPLACE_WITH_TIME_ZONE": "utc+00:00",
        "REPLACE_WITH_MIN_DEPTH": "0.0",
        "REPLACE_WITH_MAX_DEPTH": "100.0",
        "REPLACE_WITH_FILE_TAG": "selected_oras5",
    }
    for placeholder, value in replacements.items():
        text = text.replace(placeholder, value)
    return name, variable, yaml.safe_load(text)


def test_profile_template_inventory():
    assert {path.stem for path in TEMPLATES.glob("*_profile.yaml")} == {
        f"{name}_profile" for name, _ in CASES
    }


def test_batch_templates_require_confirmed_home_default():
    for name in ("cds_batch.yaml", "marine_batch.yaml"):
        text = (TEMPLATES / name).read_text(encoding="utf-8")
        assert "confirm it with the user" in text
        assert "~/xmetai-downloads/<task-name>/" in text
        assert "Do not default to the skill repo" in text


def test_skill_requires_confirmed_output_root():
    text = (SKILL / "SKILL.md").read_text(encoding="utf-8")
    assert "必须与用户确认 `storage.root_path`" in text
    assert "~/xmetai-downloads/<任务名>/" in text
    assert "不要默认使用\nskill 仓库、`output/`" in text


def test_filled_profile_fields(filled_profile):
    name, variable, profile = filled_profile
    assert "REPLACE_" not in yaml.safe_dump(profile)
    assert not {"year", "month", "day", "period", "storage"}.intersection(profile)
    if name.startswith("marine"):
        assert profile["variables"] == [variable]
        for field in ("longitude", "latitude", "depth"):
            assert set(profile[field]) == {"minimum", "maximum"}
            assert all(isinstance(value, (int, float)) for value in profile[field].values())
            assert profile[field]["minimum"] <= profile[field]["maximum"]
        assert not {"time", "pressure_level", "area"}.intersection(profile)
        return
    assert profile["area"] == [60, 70, 10, 140]
    assert ("pressure_level" in profile) == ("pressure" in name)
    if "pressure" in name:
        assert profile["pressure_level"] == ["500"]
    if "daily" in name:
        assert profile["statistics"] == {"daily_mean": {"variable": [variable]}}
        assert profile["frequency"] == "1_hourly"
        assert profile["time_zone"] == "utc+00:00"
        assert not {"variable", "daily_statistic", "time", "data_format",
                    "download_format"}.intersection(profile)
    else:
        assert profile["variable"] == [variable]
        assert profile["time"] == ["06:00"]
        assert profile["data_format"] == "netcdf"
        assert profile["download_format"] == "unarchived"


@pytest.fixture(scope="module")
def datakit():
    value = os.environ.get("XMETAI_DATAKIT_ROOT")
    if not value:
        pytest.skip("Set XMETAI_DATAKIT_ROOT to test real Datakit compatibility")
    root = Path(value).expanduser().resolve()
    builders = {}
    for source, directory, class_name in (
        ("cds", "cds_download", "CDSTaskBuilder"),
        ("marine", "copernicusmarine_download", "CopernicusMarineTaskBuilder"),
    ):
        path = root / "examples" / "dataset_downloader" / directory / f"{directory}.py"
        assert path.is_file(), f"Datakit downloader not found: {path}"
        spec = importlib.util.spec_from_file_location(f"template_test_{source}", path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        builders[source] = getattr(module, class_name)
    return root, builders


def filled_batch(source):
    text = (TEMPLATES / f"{source}_batch.yaml").read_text(encoding="utf-8")
    replacements = {
        "REPLACE_WITH_PROFILE_PATH.yaml": "./profiles/selected.yaml",
        "REPLACE_WITH_OUTPUT_SUBDIR": "selected",
        "/REPLACE_WITH_ABSOLUTE_OUTPUT_DIRECTORY": "./output",
        "REPLACE_WITH_START_MONTH": "2020-01",
        "REPLACE_WITH_END_MONTH": "2020-02",
        "REPLACE_WITH_START_DATE": "2020-02-28",
        "REPLACE_WITH_END_DATE": "2020-02-29",
    }
    for placeholder, value in replacements.items():
        text = text.replace(placeholder, value)
    return yaml.safe_load(text)


@pytest.mark.parametrize("source", ["cds", "marine"])
def test_filled_batch_fields(source):
    batch = filled_batch(source)
    assert "REPLACE_" not in yaml.safe_dump(batch)
    assert set(batch) == {"profiles", "storage", "period", "execution"}
    assert batch["profiles"] == [
        {"path": "./profiles/selected.yaml", "output_subdir": "selected"}
    ]
    assert batch["storage"] == {"root_path": "./output"}
    assert batch["period"] == (
        {"start_month": "2020-01", "end_month": "2020-02"} if source == "cds"
        else {"start_date": "2020-02-28", "end_date": "2020-02-29"}
    )
    assert batch["execution"] == {
        "max_workers": 1, "max_retries": 3,
        "retry_delay_seconds": 30, "continue_on_error": True,
    }


def write_batch(tmp_path, source, profile):
    profile_path = tmp_path / "profiles" / "selected.yaml"
    profile_path.parent.mkdir()
    profile_path.write_text(yaml.safe_dump(profile), encoding="utf-8")
    batch = filled_batch(source)
    path = tmp_path / "batch.yaml"
    path.write_text(yaml.safe_dump(batch), encoding="utf-8")
    return path


def test_real_datakit_build_and_preview(filled_profile, datakit, tmp_path):
    name, variable, profile = filled_profile
    root, builders = datakit
    source = "marine" if name.startswith("marine") else "cds"
    batch_path = write_batch(tmp_path, source, profile)
    builder = builders[source].from_batch_task(batch_path)
    assert builder.max_workers == 1
    assert builder.max_retries == 3
    assert builder.retry_delay_seconds == 30
    assert builder.continue_on_error is True
    records_root = tmp_path / "output" / "_records" / "batch"
    assert builder.plan_path == records_root / "plan.txt"
    assert builder.status_log_path == records_root / "status.jsonl"
    tasks = builder.build()
    assert len(tasks) == 2
    assert len({task["target"] for task in tasks}) == 2
    for task in tasks:
        assert Path(task["target"]).is_relative_to(tmp_path / "output" / "selected")
        request = task["request"]
        assert "REPLACE_" not in yaml.safe_dump(request)
        if source == "marine":
            assert request["dataset_id"] == profile["dataset_id"]
            assert request["variables"] == [variable]
            assert request["minimum_depth"] == 0.0
            assert request["maximum_depth"] == 100.0
            assert task["target"].endswith("_selected_oras5.nc")
        else:
            assert task["dataset"] == profile["dataset"]
            assert request["variable"] == [variable]
            assert request["area"] == [60, 70, 10, 140]
            assert len(request["day"]) == (31 if request["month"] == ["01"] else 29)
            if "pressure" in name:
                assert request["pressure_level"] == ["500"]
            if "daily" in name:
                assert request["daily_statistic"] == "daily_mean"
                assert request["frequency"] == "1_hourly"
                assert "statistics" not in request
                assert task["target"].endswith(".zip")
            else:
                assert request["time"] == ["06:00"]
                assert task["target"].endswith(".nc")
    result = subprocess.run(
        [sys.executable, str(SKILL / "scripts" / "run_download.py"),
         "--repo-root", str(root), "--source", source, str(batch_path), "--dry-run"],
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr
    plan = builder.plan_path.read_text(encoding="utf-8")
    assert all(task["target"] in plan for task in tasks)

    batch = yaml.safe_load(batch_path.read_text(encoding="utf-8"))
    batch["profiles"].append(
        {"path": "./profiles/selected.yaml", "output_subdir": "another"}
    )
    batch["records"] = {
        "plan_path": "./records/plan.txt",
        "status_log_path": "./records/status.jsonl",
    }
    batch_path.write_text(yaml.safe_dump(batch), encoding="utf-8")
    builder = builders[source].from_batch_task(batch_path)
    tasks = builder.build()
    assert len(tasks) == 4
    assert len({task["target"] for task in tasks}) == 4
    assert builder.plan_path == tmp_path / "records" / "plan.txt"
    assert builder.status_log_path == tmp_path / "records" / "status.jsonl"


def test_real_datakit_multiple_daily_groups(datakit, tmp_path):
    _, builders = datakit
    template = yaml.safe_load(
        (TEMPLATES / "cds_daily_single_profile.yaml").read_text(encoding="utf-8")
    )
    template.update(
        statistics={
            "daily_mean": {"variable": ["2m_temperature"]},
            "daily_sum": {"variable": ["total_precipitation"]},
            "daily_max": {
                "variable": ["maximum_2m_temperature_since_previous_post_processing"]
            },
            "daily_min": {
                "variable": ["minimum_2m_temperature_since_previous_post_processing"]
            },
        },
        frequency="1_hourly", time_zone="utc+00:00", area=[60, 70, 10, 140],
    )
    batch_path = write_batch(tmp_path, "cds", template)
    tasks = builders["cds"].from_batch_task(batch_path).build()
    assert len(tasks) == 8
    assert len({task["target"] for task in tasks}) == 8
    assert {task["request"]["daily_statistic"] for task in tasks} == set(
        template["statistics"]
    )
    for task in tasks:
        statistic = task["request"]["daily_statistic"]
        assert task["request"]["variable"] == template["statistics"][statistic]["variable"]
        assert statistic in Path(task["target"]).parts
