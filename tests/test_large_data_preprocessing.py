from __future__ import annotations

import importlib.util
import subprocess
import sys
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


def test_split_cli_dry_run(tmp_path: Path) -> None:
    source = tmp_path / "input.nc"
    xr.Dataset(
        {"x": (("time", "lat", "lon"), np.ones((1, 1, 1), dtype=np.float32))},
        coords={"time": [np.datetime64("2023-01-01")], "lat": [0.0], "lon": [0.0]},
    ).to_netcdf(source)
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--input", str(source), "--output", str(tmp_path / "out.zarr")],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "DRY-RUN: nothing was written" in result.stdout
    assert not (tmp_path / "out.zarr").exists()


def test_parse_chunks() -> None:
    assert convert.parse_chunks("time=4,level=-1") == {"time": 4, "level": -1}
    with pytest.raises(SystemExit):
        convert.parse_chunks("time=0")


def test_netcdf_engine_prefers_h5netcdf_for_hdf5(tmp_path: Path) -> None:
    path = tmp_path / "sample.nc"
    xr.Dataset({"x": (("time",), [1.0])}, coords={"time": [0]}).to_netcdf(
        path, engine="h5netcdf"
    )
    assert convert.netcdf_engine_for([path]) == "h5netcdf"


def test_netcdf_engine_keeps_classic_compatible(tmp_path: Path) -> None:
    path = tmp_path / "classic.nc"
    xr.Dataset({"x": (("time",), [1.0])}, coords={"time": [0]}).to_netcdf(
        path, engine="netcdf4", format="NETCDF3_CLASSIC"
    )
    assert convert.netcdf_engine_for([path]) in {"netcdf4", None}


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


def test_streaming_chan_statistics_match_full_reduction() -> None:
    values = np.array(
        [
            [[1.0, np.nan], [10.0, 20.0]],
            [[3.0, 5.0], [30.0, 40.0]],
            [[7.0, 9.0], [50.0, np.nan]],
        ],
        dtype=np.float32,
    )
    ds = xr.Dataset(
        {"data": (("time", "level", "point"), values)},
        coords={"time": np.arange(3), "level": ["a", "b"]},
    ).chunk({"time": 1})
    left = convert.channel_moments(ds.isel(time=slice(0, 1)))
    right = convert.channel_moments(ds.isel(time=slice(1, None)))
    names, mean, std, coord = convert.finalize_channel_moments(
        convert.merge_channel_moments(left, right)
    )
    full_names, full_mean, full_std, full_coord = convert.compute_channel_stats(ds)
    assert (names, coord) == (full_names, full_coord)
    np.testing.assert_allclose(mean, full_mean, rtol=1e-6)
    np.testing.assert_allclose(std, full_std, rtol=1e-6)


def test_streaming_moments_ignore_empty_batch_without_nan_poisoning() -> None:
    empty = xr.Dataset(
        {"data": (("time", "level"), np.array([[np.nan]], dtype=np.float32))},
        coords={"time": [0], "level": ["tp"]},
    )
    valid = xr.Dataset(
        {"data": (("time", "level"), np.array([[1.0], [3.0]], dtype=np.float32))},
        coords={"time": [1, 2], "level": ["tp"]},
    )
    merged = convert.merge_channel_moments(
        convert.channel_moments(empty), convert.channel_moments(valid)
    )
    names, mean, std, _ = convert.finalize_channel_moments(merged)
    assert names == ["tp"]
    np.testing.assert_allclose(mean, [2.0])
    np.testing.assert_allclose(std, [1.0])

    with pytest.raises(SystemExit, match="no valid samples"):
        convert.finalize_channel_moments(convert.channel_moments(empty))


def test_file_period_batches_bound_open_files_and_preserve_times(tmp_path: Path) -> None:
    for day in range(4):
        times = np.arange(
            np.datetime64("2023-01-01T00") + np.timedelta64(day * 24, "h"),
            np.datetime64("2023-01-01T00") + np.timedelta64((day + 1) * 24, "h"),
            np.timedelta64(1, "h"),
        )
        xr.Dataset(
            {"x": (("time",), np.full(24, day + 1, dtype=np.float32))},
            coords={"time": times},
        ).to_netcdf(tmp_path / f"day-{day}.nc")
    paths = sorted(tmp_path.glob("*.nc"))
    loaded = []
    open_counts = []
    for batch, open_count in convert.prepared_file_batches(
        paths,
        period_batch_size=1,
        overlap_periods=1,
        chunks={"time": 6},
        steps=[],
    ):
        loaded.append(batch.load())
        open_counts.append(open_count)
    combined = xr.concat(loaded, dim="time")
    assert combined.sizes["time"] == 96
    assert max(open_counts) == 3
    np.testing.assert_array_equal(combined.time.values, np.arange(
        np.datetime64("2023-01-01T00"), np.datetime64("2023-01-05T00"), np.timedelta64(1, "h")
    ))


def test_file_period_overlap_preserves_daily_window_across_boundary(tmp_path: Path) -> None:
    for day in range(3):
        start = np.datetime64("2023-06-30") + np.timedelta64(day, "D")
        times = np.arange(start, start + np.timedelta64(1, "D"), np.timedelta64(1, "h"))
        xr.Dataset(
            {"acc": (("time",), np.ones(24, dtype=np.float32))},
            coords={"time": times},
        ).to_netcdf(tmp_path / f"day-{day}.nc")
    loaded = []
    for batch, _ in convert.prepared_file_batches(
        sorted(tmp_path.glob("*.nc")),
        period_batch_size=1,
        overlap_periods=1,
        chunks={"time": 6},
        steps=[{"daily_aggregation": {"variables": {"acc": {"operator": "sum"}}}}],
    ):
        loaded.append(batch.load())
    combined = xr.concat(loaded, dim="time")
    np.testing.assert_array_equal(
        combined.time.values,
        np.array(["2023-07-01", "2023-07-02"], dtype="datetime64[ns]"),
    )
    np.testing.assert_allclose(combined.acc.values, [24.0, 24.0])


def test_file_period_overlap_does_not_emit_incomplete_lookahead_state(tmp_path: Path) -> None:
    xr.Dataset(
        {"state": (("time",), np.array([30.0, 31.0], dtype=np.float32))},
        coords={"time": np.array(["2023-07-30", "2023-07-31"], dtype="datetime64[ns]")},
    ).to_netcdf(tmp_path / "state-202307.nc")
    xr.Dataset(
        {"state": (("time",), np.array([1.0, 2.0], dtype=np.float32))},
        coords={"time": np.array(["2023-08-01", "2023-08-02"], dtype="datetime64[ns]")},
    ).to_netcdf(tmp_path / "state-202308.nc")

    july_hours = np.arange(
        np.datetime64("2023-07-30T00"),
        np.datetime64("2023-08-01T00"),
        np.timedelta64(1, "h"),
    )
    august_hours = np.arange(
        np.datetime64("2023-08-01T00"),
        np.datetime64("2023-08-03T00"),
        np.timedelta64(1, "h"),
    )
    xr.Dataset(
        {"acc": (("time",), np.ones(july_hours.size, dtype=np.float32))},
        coords={"time": july_hours},
    ).to_netcdf(tmp_path / "acc-202307.nc")
    xr.Dataset(
        {"acc": (("time",), np.full(august_hours.size, 2.0, dtype=np.float32))},
        coords={"time": august_hours},
    ).to_netcdf(tmp_path / "acc-202308.nc")

    loaded = []
    for batch, _ in convert.prepared_file_batches(
        sorted(tmp_path.glob("*.nc")),
        period_batch_size=1,
        overlap_periods=1,
        chunks={"time": 6},
        steps=[{"daily_aggregation": {"variables": {"acc": {"operator": "sum"}}}}],
    ):
        loaded.append(batch.load())

    combined = xr.concat(loaded, dim="time")
    np.testing.assert_array_equal(
        combined.time.values,
        np.array(["2023-07-31", "2023-08-01", "2023-08-02"], dtype="datetime64[ns]"),
    )
    np.testing.assert_allclose(combined.state.values, [31.0, 1.0, 2.0])
    np.testing.assert_allclose(combined.acc.values, [24.0, 25.0, 48.0])
    assert np.isfinite(combined.acc.values).all()


def test_normalize_ds_stays_lazy() -> None:
    values = np.arange(16, dtype=np.float32).reshape(2, 2, 2, 2)
    ds = xr.Dataset(
        {"data": (("time", "level", "lat", "lon"), values)},
        coords={"level": ["a", "b"]},
    ).chunk({"time": 1})

    normalized = convert.normalize_ds(ds, np.array([1.0, 2.0]), np.array([2.0, 4.0]), "level", ["a", "b"])

    assert hasattr(normalized["data"].data, "chunks")


def test_weight_sidecar_uses_latitude_not_channels(tmp_path: Path) -> None:
    latitudes = np.array([90.0, 60.0, 0.0, -60.0, -90.0], dtype=np.float32)
    ds = xr.Dataset(
        {"data": (("time", "level", "lat", "lon"), np.zeros((1, 2, 5, 1), dtype=np.float32))},
        coords={"time": [0], "level": ["z1000", "tp"], "lat": latitudes, "lon": [0.0]},
    )
    lat, weight = convert.latitude_weights(ds)
    np.testing.assert_array_equal(lat, latitudes)
    np.testing.assert_allclose(weight, np.cos(np.deg2rad(np.abs(latitudes))), atol=1e-7)

    convert.write_sidecars(
        tmp_path,
        ["z1000", "tp"],
        np.array([1.0, 2.0], dtype=np.float32),
        np.array([3.0, 4.0], dtype=np.float32),
        lat,
        weight,
        "level",
    )
    with xr.open_dataarray(tmp_path / "mean.nc") as mean:
        assert mean.dims == ("level",)
        assert mean.shape == (2,)
    with xr.open_dataarray(tmp_path / "weight.nc") as spatial:
        assert spatial.dims == ("lat",)
        assert spatial.shape == (5,)
        np.testing.assert_array_equal(spatial.lat.values, latitudes)

    # Matches core after weight.reshape(1, H, 1): channel and spatial
    # weights broadcast independently over (B, T, C, H, W).
    output = np.zeros((1, 1, 2, 5, 3), dtype=np.float32)
    channel_weight = np.ones((2, 1, 1), dtype=np.float32)
    spatial_weight = weight.reshape(1, 5, 1)
    assert (output * channel_weight * spatial_weight).shape == output.shape


def test_latitude_weight_rejects_missing_or_multidimensional_lat() -> None:
    with pytest.raises(SystemExit, match="one-dimensional lat"):
        convert.latitude_weights(xr.Dataset(coords={"latitude": [0.0]}))
    with pytest.raises(SystemExit, match="one-dimensional lat"):
        convert.latitude_weights(
            xr.Dataset(coords={"lat": (("y", "x"), np.zeros((2, 2), dtype=np.float32))})
        )


def test_resume_requires_existing_times_to_be_exact_prefix() -> None:
    planned = xr.Dataset(
        {"data": (("time", "level"), np.arange(8, dtype=np.float32).reshape(4, 2))},
        coords={"time": np.arange(4), "level": ["a", "b"]},
    )
    existing = planned.isel(time=slice(0, 2))
    assert convert.validate_resume_prefix(planned, existing) == 2

    with pytest.raises(SystemExit, match="exact prefix"):
        convert.validate_resume_prefix(planned, planned.isel(time=[0, 2]))

    incompatible = existing.assign_coords(level=["a", "c"])
    with pytest.raises(SystemExit, match="coordinate 'level' differs"):
        convert.validate_resume_prefix(planned, incompatible)


def test_incremental_write_resumes_from_store_not_stale_state(tmp_path: Path) -> None:
    pytest.importorskip("zarr")
    output = tmp_path / "out.zarr"
    state_path = convert.conversion_state_path(output)
    ds = xr.Dataset(
        {
            "data": (
                ("time", "level", "lat", "lon"),
                np.arange(20, dtype=np.float32).reshape(5, 1, 2, 2),
            )
        },
        coords={"time": np.arange(5), "level": ["a"], "lat": [1.0, 0.0], "lon": [0.0, 1.0]},
    ).chunk({"time": 1})

    # Mimic a process that successfully appended the first batch but stopped
    # before its audit file was updated. Recovery must trust the store prefix.
    ds.isel(time=slice(0, 2)).to_zarr(output, mode="w", consolidated=False)
    convert.write_conversion_state(
        state_path,
        {"status": "failed", "completed_time_steps": 0, "contract_sha256": "test"},
    )
    state = {"contract_sha256": "test", "output": str(output)}
    convert.write_incremental_zarr(
        ds,
        output,
        batch_time=2,
        resume=True,
        state_path=state_path,
        state=state,
    )

    restored = xr.open_zarr(output, consolidated=True)
    try:
        np.testing.assert_array_equal(restored.time.values, np.arange(5))
        np.testing.assert_allclose(restored.data.values, ds.data.compute().values)
    finally:
        restored.close()
    saved = __import__("json").loads(state_path.read_text(encoding="utf-8"))
    assert saved["status"] == "completed"
    assert saved["completed_time_steps"] == 5


def test_resume_with_state_but_no_store_creates_output(tmp_path: Path) -> None:
    pytest.importorskip("zarr")
    output = tmp_path / "not-created-yet.zarr"
    state_path = convert.conversion_state_path(output)
    convert.write_conversion_state(state_path, {"status": "failed", "phase": "statistics"})
    ds = xr.Dataset({"x": (("time",), [1.0, 2.0])}, coords={"time": [0, 1]})

    convert.write_incremental_zarr(
        ds,
        output,
        batch_time=1,
        resume=True,
        state_path=state_path,
        state={"contract_sha256": "same"},
    )

    restored = xr.open_zarr(output, consolidated=True)
    try:
        np.testing.assert_allclose(restored.x.values, [1.0, 2.0])
    finally:
        restored.close()


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


def test_s2s_daily_accumulation_uses_ending_24_hour_window_and_units() -> None:
    times = np.arange(
        np.datetime64("2023-06-30T01"),
        np.datetime64("2023-07-02T01"),
        np.timedelta64(1, "h"),
    )
    ds = xr.Dataset(
        {
            "tp": (("time", "lat", "lon"), np.full((48, 1, 1), 0.001, dtype=np.float32)),
            "ttr": (("time", "lat", "lon"), np.full((48, 1, 1), 3600.0, dtype=np.float32)),
            "t2m": (("time", "lat", "lon"), np.arange(48, dtype=np.float32)[:, None, None]),
        },
        coords={"time": times, "lat": [0.0], "lon": [0.0]},
    ).chunk({"time": 6})

    result = convert.apply_steps(ds, [{"s2s_daily_accumulation": {}}])

    np.testing.assert_array_equal(
        result.time.values,
        np.array(["2023-07-01", "2023-07-02"], dtype="datetime64[ns]"),
    )
    assert hasattr(result.tp.data, "chunks")
    np.testing.assert_allclose(result.tp.compute(), 24.0)
    np.testing.assert_allclose(result.ttr.compute(), 1.0)
    np.testing.assert_allclose(result.t2m.values[:, 0, 0], [23.0, 47.0])
    assert result.tp.attrs["units"] == "mm"
    assert result.ttr.attrs["units"] == "W m-2"


def test_daily_aggregation_supports_custom_variables_and_operators() -> None:
    times = np.arange(
        np.datetime64("2023-06-30T01"),
        np.datetime64("2023-07-01T01"),
        np.timedelta64(1, "h"),
    )
    ds = xr.Dataset(
        {
            "ssr": (("time",), np.full(24, 3600.0, dtype=np.float32)),
            "custom_peak": (("time",), np.arange(24, dtype=np.float32)),
            "custom_mean": (("time",), np.arange(24, dtype=np.float32)),
        },
        coords={"time": times},
    ).chunk({"time": 6})

    result = convert.apply_steps(
        ds,
        [
            {
                "daily_aggregation": {
                    "variables": {
                        "ssr": {"operator": "sum", "factor": 1.0 / 86400.0, "units": "W m-2"},
                        "custom_peak": {"operator": "max", "offset": 2.0, "units": "custom"},
                        "custom_mean": {"operator": "mean"},
                    }
                }
            }
        ],
    )

    assert list(result.data_vars) == ["ssr", "custom_peak", "custom_mean"]
    assert hasattr(result.ssr.data, "chunks")
    np.testing.assert_allclose(result.ssr.compute(), 1.0)
    np.testing.assert_allclose(result.custom_peak.compute(), 25.0)
    np.testing.assert_allclose(result.custom_mean.compute(), 11.5)
    assert result.ssr.attrs["units"] == "W m-2"

    with pytest.raises(SystemExit, match="unsupported operator"):
        convert.daily_aggregation(
            ds,
            {"variables": {"ssr": {"operator": "median"}}},
        )


def test_s2s_daily_accumulation_rejects_or_drops_incomplete_windows() -> None:
    times = np.arange(
        np.datetime64("2023-07-01T00"),
        np.datetime64("2023-07-03T01"),
        np.timedelta64(1, "h"),
    )
    values = np.ones((49, 1, 1), dtype=np.float32)
    ds = xr.Dataset(
        {"tp": (("time", "lat", "lon"), values), "ttr": (("time", "lat", "lon"), values)},
        coords={"time": times, "lat": [0.0], "lon": [0.0]},
    )

    with pytest.raises(SystemExit, match="complete 24-hour windows"):
        convert.s2s_daily_accumulation(ds, {})

    result = convert.s2s_daily_accumulation(ds, {"incomplete": "drop"})
    np.testing.assert_array_equal(
        result.time.values,
        np.array(["2023-07-02", "2023-07-03"], dtype="datetime64[ns]"),
    )


def test_s2s_daily_accumulation_detects_missing_hour() -> None:
    times = np.arange(
        np.datetime64("2023-06-30T01"),
        np.datetime64("2023-07-02T01"),
        np.timedelta64(1, "h"),
    )
    times = np.delete(times, 10)
    values = np.ones((47, 1, 1), dtype=np.float32)
    ds = xr.Dataset(
        {"tp": (("time", "lat", "lon"), values), "ttr": (("time", "lat", "lon"), values)},
        coords={"time": times, "lat": [0.0], "lon": [0.0]},
    )

    with pytest.raises(SystemExit, match="incomplete daily window"):
        convert.s2s_daily_accumulation(ds, {})


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
