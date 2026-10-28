"""Opt-in CDS download test; never contacts CDS during the default test suite.

Run from the xmetai-skills repository root (POSIX shell):

    XMETAI_RUN_CDS_LIVE=1 \\
    XMETAI_DATAKIT_ROOT=/home/lenovo/workspace_1/xmetai-datakit_main/xmetai-datakit \\
      conda run --no-capture-output -n xmetai-skills \\
      python -m pytest tests/test_downloader_cds_live.py -v -s

The Datakit path above is a local example; replace it for other machines.
Prerequisites:
- Install Datakit's CDS requirements and pytest, PyYAML, numpy, xarray,
  h5netcdf, and h5py in the environment running pytest.
- Configure ~/.cdsapirc, or set CDSAPI_RC to another credential file path.
  Never put the API key in test code, commands, or version control.
- Accept the CDS "ERA5 hourly data on single levels" dataset terms.

Without XMETAI_RUN_CDS_LIVE=1, only the offline configuration check runs
(it requires PyYAML); the download test is skipped even if credentials exist.
After opt-in, missing configuration or dependencies and request failures fail
the test rather than skip it. Both tests should pass when fully configured.

The live test submits one real request for February 2021 daily 00 UTC 2 m
temperature over 30-30.25 N, 120-120.25 E. It uses pytest's temporary directory,
not the repository's existing smoke output, and checks records and the data
(28 times, a 2-by-2 grid, K units, and 112 finite values).
There are no retries; the download subprocess timeout is 600 seconds.
A local timeout does not guarantee cancellation of the remote CDS task.
"""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures" / "dataset-downloader"
ENTRY = (
    ROOT / "skills" / "dataset-downloader"
    / "scripts" / "run_download.py"
)


def test_smoke_configuration():
    yaml = pytest.importorskip("yaml")
    profile = yaml.safe_load((FIXTURES / "cds_smoke_profile.yaml").read_text())
    batch = yaml.safe_load((FIXTURES / "cds_smoke_batch.yaml").read_text())
    assert profile == {
        "dataset": "reanalysis-era5-single-levels",
        "product_type": ["reanalysis"],
        "variable": ["2m_temperature"],
        "time": ["00:00"],
        "area": [30.25, 120.0, 30.0, 120.25],
        "data_format": "netcdf",
        "download_format": "unarchived",
    }
    assert batch["profiles"] == [
        {"path": "./cds_smoke_profile.yaml", "output_subdir": "era5_t2m"}
    ]
    assert batch["period"] == {"start_month": "2021-02", "end_month": "2021-02"}
    assert batch["execution"]["max_workers"] == 1
    assert batch["execution"]["max_retries"] == 0
    assert batch["execution"]["continue_on_error"] is False
    assert (FIXTURES / batch["storage"]["root_path"]).resolve() == (
        ROOT / "output" / "downloader_smoke"
    )


@pytest.mark.skipif(
    os.environ.get("XMETAI_RUN_CDS_LIVE") != "1",
    reason="Set XMETAI_RUN_CDS_LIVE=1 to explicitly allow one real CDS request",
)
def test_cds_live_smoke_download(tmp_path):
    datakit_value = os.environ.get("XMETAI_DATAKIT_ROOT")
    if not datakit_value:
        pytest.fail("Live test requires XMETAI_DATAKIT_ROOT")
    datakit = Path(datakit_value).expanduser().resolve()
    assert (datakit / "pyproject.toml").is_file(), "Invalid Datakit repository"
    for directory in ("cds_download", "copernicusmarine_download"):
        assert (
            datakit / "examples" / "dataset_downloader" / directory / f"{directory}.py"
        ).is_file(), f"Missing Datakit downloader: {directory}"

    # Only the official client reads credentials; the test checks file existence.
    rc = Path(os.environ.get("CDSAPI_RC", "~/.cdsapirc")).expanduser().resolve()
    assert rc.is_file(), "Configure CDSAPI_RC or the default ~/.cdsapirc first"
    # Missing dependencies are failures after explicit opt-in, not skipped downloads.
    import cdsapi  # noqa: F401
    import h5netcdf  # noqa: F401
    import h5py  # noqa: F401
    import numpy as np
    import xarray as xr
    import yaml

    profile_path = tmp_path / "cds_smoke_profile.yaml"
    shutil.copyfile(FIXTURES / profile_path.name, profile_path)
    batch = yaml.safe_load(
        (FIXTURES / "cds_smoke_batch.yaml").read_text(encoding="utf-8")
    )
    output = tmp_path / "output"
    batch["storage"]["root_path"] = str(output)
    batch_path = tmp_path / "cds_smoke_batch.yaml"
    batch_path.write_text(yaml.safe_dump(batch), encoding="utf-8")
    env = os.environ.copy()
    env["CDSAPI_RC"] = str(rc)
    result = subprocess.run(
        [sys.executable, str(ENTRY), "--repo-root", str(datakit),
         "--source", "cds", str(batch_path), "--execute"],
        env=env, capture_output=True, text=True, check=False, timeout=600,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {
        "total": 1, "completed": 1, "skipped": 0, "failed": 0,
    }

    target = output / "era5_t2m" / "2m_temperature" / "2021" / "2m_temperature_202102.nc"
    assert target.is_file() and target.stat().st_size > 0
    assert not target.with_name(target.name + ".part").exists()
    records_root = output / "_records" / "cds_smoke_batch"
    records = [
        json.loads(line)
        for line in (records_root / "status.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert records[-1]["status"] == "completed"
    assert str(target) in (records_root / "plan.txt").read_text(encoding="utf-8")
    with xr.open_dataset(target, engine="h5netcdf") as ds:
        assert "t2m" in ds.data_vars
        assert ds["t2m"].attrs["units"] == "K"
        expected = np.arange(
            np.datetime64("2021-02-01"), np.datetime64("2021-03-01"),
            np.timedelta64(1, "D"),
        )
        np.testing.assert_array_equal(ds["valid_time"].values, expected)
        np.testing.assert_allclose(ds["latitude"].values, [30.25, 30.0])
        np.testing.assert_allclose(ds["longitude"].values, [120.0, 120.25])
        values = ds["t2m"].values
        assert values.size == 112
        assert np.isfinite(values).all()
