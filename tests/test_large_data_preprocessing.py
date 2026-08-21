from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest


xr = pytest.importorskip("xarray")
pytest.importorskip("dask.array")

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "skills" / "xmetai-weather-modeling" / "scripts" / "convert_to_zarr.py"
SPEC = importlib.util.spec_from_file_location("convert_to_zarr", SCRIPT)
assert SPEC and SPEC.loader
convert = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(convert)


def test_parse_chunks() -> None:
    assert convert.parse_chunks("time=4,level=-1") == {"time": 4, "level": -1}
    with pytest.raises(SystemExit):
        convert.parse_chunks("time=0")


def test_default_channel_order_matches_cla_and_keeps_flexible_extras() -> None:
    incoming = ["tp", "custom", "z50", "t2m", "z1000"]
    assert convert.default_channel_order(incoming) == ["z1000", "z50", "t2m", "tp", "custom"]


def test_merge_to_data_uses_cla_relative_order_by_default() -> None:
    ds = xr.Dataset(
        {
            "tp": (("time",), [76.0]),
            "z50": (("time",), [13.0]),
            "z1000": (("time",), [1.0]),
        }
    )
    merged = convert.apply_steps(ds, [{"merge_to_data": {"coord": "level"}}])
    assert merged.level.values.tolist() == ["z1000", "z50", "tp"]
    np.testing.assert_allclose(merged.data.values, [[1.0, 13.0, 76.0]])


def test_merge_to_data_respects_explicit_order() -> None:
    ds = xr.Dataset({"a": (("time",), [1.0]), "b": (("time",), [2.0])})
    merged = convert.apply_steps(ds, [{"merge_to_data": {"coord": "level", "order": ["b", "a"]}}])
    assert merged.level.values.tolist() == ["b", "a"]
    np.testing.assert_allclose(merged.data.values, [[2.0, 1.0]])


def test_vectorized_channel_stats_preserve_order_and_nan_policy() -> None:
    values = np.array(
        [
            [[[1.0, 2.0]], [[10.0, np.nan]]],
            [[[3.0, 4.0]], [[20.0, 30.0]]],
        ],
        dtype=np.float32,
    )
    ds = xr.Dataset(
        {"data": (("time", "level", "lat", "lon"), values)},
        coords={"level": ["a", "b"], "lat": [0.0], "lon": [0.0, 1.0]},
    ).chunk({"time": 1})

    names, mean, std, coord = convert.compute_channel_stats(ds)

    assert names == ["a", "b"]
    assert coord == "level"
    np.testing.assert_allclose(mean, [2.5, 20.0])
    np.testing.assert_allclose(std, [np.std([1.0, 2.0, 3.0, 4.0]), np.std([10.0, 20.0, 30.0])])


def test_normalize_ds_stays_lazy() -> None:
    values = np.arange(16, dtype=np.float32).reshape(2, 2, 2, 2)
    ds = xr.Dataset(
        {"data": (("time", "level", "lat", "lon"), values)},
        coords={"level": ["a", "b"]},
    ).chunk({"time": 1})

    normalized = convert.normalize_ds(ds, np.array([1.0, 2.0]), np.array([2.0, 4.0]), "level", ["a", "b"])

    assert hasattr(normalized["data"].data, "chunks")
