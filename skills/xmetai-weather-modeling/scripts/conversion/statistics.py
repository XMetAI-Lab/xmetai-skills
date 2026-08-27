"""Numerically stable per-channel and latitude statistics."""
from __future__ import annotations

from pathlib import Path
from typing import Any
import warnings

import numpy as np

try:
    import xarray as xr
except ImportError:  # pragma: no cover
    xr = None

from .state import write_conversion_state


def compute_channel_stats(ds):
    """Return (channel names, mean, std, channel coord) from the prepared dataset."""
    coord = next((c for c in ("level", "channel") if c in ds.coords), "")
    if coord and "data" in ds.data_vars:
        names = [str(v) for v in ds.coords[coord].values]
        da = ds["data"]
        dims = [d for d in da.dims if d != coord]
        stats = xr.Dataset(
            {"mean": da.mean(dim=dims, skipna=True), "std": da.std(dim=dims, skipna=True)}
        ).compute()
        mean = np.asarray(stats["mean"].values, dtype=np.float32)
        std = np.asarray(stats["std"].values, dtype=np.float32)
        return names, mean, std, coord
    names = list(ds.data_vars)
    stats = xr.Dataset(
        {
            **{f"mean_{i}": ds[name].mean(skipna=True) for i, name in enumerate(names)},
            **{f"std_{i}": ds[name].std(skipna=True) for i, name in enumerate(names)},
        }
    ).compute()
    mean = np.asarray([stats[f"mean_{i}"].item() for i in range(len(names))], dtype=np.float32)
    std = np.asarray([stats[f"std_{i}"].item() for i in range(len(names))], dtype=np.float32)
    return names, mean, std, ""


def channel_moments(ds):
    """Compute mergeable per-channel count/mean/M2 for one prepared batch."""
    coord = next((c for c in ("level", "channel") if c in ds.coords), "")
    if coord and "data" in ds.data_vars:
        names = [str(v) for v in ds.coords[coord].values]
        da = ds["data"]
        dims = [d for d in da.dims if d != coord]
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*invalid value encountered in divide.*")
            warnings.filterwarnings("ignore", message=".*Degrees of freedom <= 0.*")
            reduced = xr.Dataset(
                {
                    "count": da.count(dim=dims),
                    "mean": da.mean(dim=dims, skipna=True),
                    "var": da.var(dim=dims, skipna=True),
                }
            ).compute()
        count = np.asarray(reduced["count"].values, dtype=np.int64)
        mean = np.asarray(reduced["mean"].values, dtype=np.float64)
        variance = np.asarray(reduced["var"].values, dtype=np.float64)
        m2 = np.multiply(variance, count, out=np.zeros_like(variance), where=count > 0)
        mean = np.where(count > 0, mean, 0.0)
        return names, count, mean, m2, coord
    names = list(ds.data_vars)
    count = []
    mean = []
    m2 = []
    for name in names:
        da = ds[name]
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*invalid value encountered in divide.*")
            warnings.filterwarnings("ignore", message=".*Degrees of freedom <= 0.*")
            reduced = xr.Dataset(
                {"count": da.count(), "mean": da.mean(skipna=True), "var": da.var(skipna=True)}
            ).compute()
        n = int(reduced["count"].item())
        avg = float(reduced["mean"].item()) if n else 0.0
        count.append(n)
        mean.append(avg)
        m2.append(float(reduced["var"].item()) * n if n else 0.0)
    return (
        names,
        np.asarray(count, dtype=np.int64),
        np.asarray(mean, dtype=np.float64),
        np.asarray(m2, dtype=np.float64),
        "",
    )


def merge_channel_moments(left, right):
    """Merge two Chan/Welford sufficient-statistic tuples."""
    l_names, l_count, l_mean, l_m2, l_coord = left
    r_names, r_count, r_mean, r_m2, r_coord = right
    if l_names != r_names or l_coord != r_coord:
        raise SystemExit("streaming statistics batches have different channel contracts")
    total = l_count + r_count
    delta = r_mean - l_mean
    ratio = np.divide(r_count, total, out=np.zeros_like(r_mean), where=total > 0)
    mean = l_mean + delta * ratio
    cross = np.divide(
        l_count * r_count,
        total,
        out=np.zeros_like(r_mean),
        where=total > 0,
    )
    m2 = l_m2 + r_m2 + delta * delta * cross
    return l_names, total, mean, m2, l_coord


def finalize_channel_moments(moments):
    """Convert count/mean/M2 into float32 population mean/std sidecars."""
    names, count, mean, m2, coord = moments
    empty = [name for name, value in zip(names, count) if value <= 0]
    invalid = [
        name
        for name, n, avg, total in zip(names, count, mean, m2)
        if n > 0 and (not np.isfinite(avg) or not np.isfinite(total))
    ]
    if empty:
        raise SystemExit(f"normalization statistics have no valid samples for channels: {empty}")
    if invalid:
        raise SystemExit(f"normalization statistics are non-finite for channels: {invalid}")
    variance = np.divide(m2, count, out=np.zeros_like(m2), where=count > 0)
    std = np.sqrt(np.maximum(variance, 0.0))
    if not np.all(np.isfinite(std)):
        bad = [name for name, value in zip(names, std) if not np.isfinite(value)]
        raise SystemExit(f"normalization standard deviations are non-finite for channels: {bad}")
    return names, mean.astype(np.float32), std.astype(np.float32), coord


def normalize_ds(ds, mean, std, coord, names):
    """Scale the dataset with (x - mean) / std per channel (std 0 -> 1)."""
    safe_std = np.where(std == 0, 1.0, std)
    if coord and "data" in ds.data_vars:
        m = xr.DataArray(mean, dims=[coord], coords={coord: names})
        s = xr.DataArray(safe_std, dims=[coord], coords={coord: names})
        return ds.assign(data=(ds["data"] - m) / s)
    for i, name in enumerate(names):
        ds[name] = (ds[name] - mean[i]) / safe_std[i]
    return ds


def latitude_weights(ds):
    """Return core-compatible spatial weights ``cos(abs(latitude))``."""
    if "lat" not in ds.coords or ds["lat"].ndim != 1:
        raise SystemExit("weight.nc generation requires a one-dimensional lat coordinate")
    lat = np.asarray(ds["lat"].values, dtype=np.float64)
    if lat.size == 0 or not np.isfinite(lat).all():
        raise SystemExit("weight.nc generation requires finite non-empty latitude values")
    weights = np.maximum(np.cos(np.deg2rad(np.abs(lat))), 0.0).astype(np.float32)
    return lat.astype(np.float32), weights


def streaming_channel_stats(batch_factory, state_path: Path | None = None, state=None):
    """Compute global channel statistics by merging bounded prepared batches."""
    merged = None
    batches = 0
    try:
        if state_path is not None and state is not None:
            state.update({"status": "running", "phase": "statistics", "statistics_batches": 0})
            write_conversion_state(state_path, state)
        for prepared, _ in batch_factory():
            current = channel_moments(prepared)
            merged = current if merged is None else merge_channel_moments(merged, current)
            batches += 1
            if state_path is not None and state is not None:
                names, count, mean, m2, coord = merged
                state.update(
                    {
                        "statistics_batches": batches,
                        "statistics": {
                            "names": names,
                            "coord": coord,
                            "count": count.tolist(),
                            "mean": mean.tolist(),
                            "m2": m2.tolist(),
                        },
                    }
                )
                write_conversion_state(state_path, state)
    except BaseException as exc:
        if state_path is not None and state is not None:
            state.update({"status": "failed", "error": f"{type(exc).__name__}: {exc}"})
            write_conversion_state(state_path, state)
        raise
    if merged is None:
        raise SystemExit("streaming statistics found no prepared time batches")
    return finalize_channel_moments(merged)
