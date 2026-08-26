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


def test_regrid_era5_quarter_degree_to_exact_s2s_grid_stays_lazy() -> None:
    lat = np.linspace(90.0, -90.0, 721)
    lon = np.arange(-180.0, 180.0, 0.25)
    values = (
        lat[:, None].astype(np.float32) + np.mod(lon[None, :], 360.0).astype(np.float32)
    )[None, ...]
    ds = xr.Dataset(
        {"t2m": (("time", "latitude", "longitude"), values)},
        coords={"time": [0], "latitude": lat, "longitude": lon},
    ).chunk({"time": 1, "latitude": 181, "longitude": 360})

    result = convert.apply_steps(ds, [{"regrid": {"target": "s2s_1.5deg"}}])

    assert result.sizes["lat"] == 121
    assert result.sizes["lon"] == 240
    np.testing.assert_array_equal(result.lat.values, np.linspace(90.0, -90.0, 121))
    np.testing.assert_array_equal(result.lon.values, np.arange(240) * 1.5)
    assert hasattr(result["t2m"].data, "chunks")
    np.testing.assert_allclose(result.t2m.isel(time=0, lat=60, lon=120).compute(), 180.0)


def test_regrid_supports_nearest_per_variable_and_rejects_bad_method() -> None:
    lat = np.linspace(90.0, -90.0, 361)
    lon = np.arange(0.0, 360.0, 0.5)
    ds = xr.Dataset(
        {
            "t2m": (("lat", "lon"), lat[:, None] + lon[None, :]),
            "lsm": (("lat", "lon"), np.broadcast_to((lon >= 180.0).astype(np.int8), (361, 720))),
        },
        coords={"lat": lat, "lon": lon},
    )

    result = convert.regrid_dataset(
        ds,
        {"target": "s2s_1.5deg", "method": "linear", "variable_methods": {"lsm": "nearest"}},
    )

    assert set(np.unique(result.lsm.values)) <= {0, 1}
    with pytest.raises(SystemExit, match="unsupported regrid method"):
        convert.regrid_dataset(ds, {"method": "conservative"})


def test_merge_static_removes_single_time_and_preserves_order() -> None:
    ds = xr.Dataset(
        {
            "z": (("valid_time", "lat", "lon"), [[[1.0, 2.0], [3.0, 4.0]]]),
            "lsm": (("valid_time", "lat", "lon"), [[[0.0, 1.0], [0.0, 1.0]]]),
        },
        coords={"valid_time": [np.datetime64("2015-01-01")], "lat": [1.0, 0.0], "lon": [0.0, 1.0]},
    )

    result = convert.apply_steps(
        ds,
        [{"merge_static": {"order": ["z", "lsm"], "coord": "channel", "name": "const"}}],
    )

    assert list(result.data_vars) == ["const"]
    assert result.const.dims == ("channel", "lat", "lon")
    assert result.channel.values.tolist() == ["z", "lsm"]
    np.testing.assert_allclose(result.const.values[0], [[1.0, 2.0], [3.0, 4.0]])
    assert convert.static_dataarray(result).name == "const"


def test_merge_static_rejects_multiple_time_values() -> None:
    ds = xr.Dataset(
        {"z": (("time", "lat", "lon"), np.ones((2, 1, 1), dtype=np.float32))},
        coords={"time": [0, 1], "lat": [0.0], "lon": [0.0]},
    )

    with pytest.raises(SystemExit, match="expected exactly one"):
        convert.apply_steps(ds, [{"merge_static": {"order": ["z"]}}])
