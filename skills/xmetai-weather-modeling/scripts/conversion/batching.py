"""Bounded input batching with overlap-aware output ownership."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .grid import canonicalize_latlon
from .inputs import open_input, open_inputs
from .state import _time_values
from .transforms import apply_steps


def steps_for_file_batch(steps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Drop only batch-edge partial windows; overlaps recover owned outputs."""
    adjusted = json.loads(json.dumps(steps))
    for step in adjusted:
        if "daily_aggregation" in step and isinstance(step["daily_aggregation"], dict):
            step["daily_aggregation"]["incomplete"] = "drop"
        if "s2s_daily_accumulation" in step and isinstance(step["s2s_daily_accumulation"], dict):
            step["s2s_daily_accumulation"]["incomplete"] = "drop"
    return adjusted


def input_file_batches(paths: list[Path], size: int, overlap: int = 1):
    """Yield bounded path groups with neighboring files for temporal windows."""
    if size <= 0 or overlap < 0:
        raise SystemExit("input batch size must be positive and overlap non-negative")
    for start in range(0, len(paths), size):
        stop = min(start + size, len(paths))
        group_start = max(0, start - overlap)
        group_stop = min(len(paths), stop + overlap)
        yield paths[group_start:group_stop], paths[start:stop]


def catalog_time_periods(paths: list[Path]):
    """Inspect files one at a time and group files with identical time coverage."""
    periods: dict[tuple[Any, Any], list[Path]] = {}
    for path in paths:
        ds = open_input(path)
        try:
            time_name = next((name for name in ("time", "valid_time") if name in ds.coords), None)
            if time_name is None:
                raise SystemExit(f"streaming multi-file input has no time coordinate: {path}")
            values = np.asarray(ds[time_name].values).reshape(-1)
            if values.size == 0:
                raise SystemExit(f"streaming multi-file input has an empty time coordinate: {path}")
            key = (values.min(), values.max())
            periods.setdefault(key, []).append(path)
        finally:
            ds.close()
    return [(start, end, periods[(start, end)]) for start, end in sorted(periods)]


def period_batches(periods, size: int, overlap: int = 1):
    """Yield bounded chronological period groups and their owned time interval."""
    if size <= 0 or overlap < 0:
        raise SystemExit("input period batch size must be positive and overlap non-negative")
    for start in range(0, len(periods), size):
        stop = min(start + size, len(periods))
        selected = periods[max(0, start - overlap) : min(len(periods), stop + overlap)]
        paths = [path for _, _, group in selected for path in group]
        yield paths, periods[start][0], periods[stop - 1][1]


def prepared_file_batches(
    paths: list[Path],
    *,
    period_batch_size: int,
    overlap_periods: int,
    chunks,
    steps,
):
    """Open, transform, yield, and close only a bounded set of time periods."""
    periods = catalog_time_periods(paths)
    adjusted_steps = steps_for_file_batch(steps)
    last_emitted = None
    for group, owned_start, owned_end in period_batches(periods, period_batch_size, overlap_periods):
        ds = open_inputs(group, chunks)
        try:
            prepared = canonicalize_latlon(apply_steps(ds, adjusted_steps))
            values = _time_values(prepared)
            # Neighboring periods are opened only to supply temporal context for
            # transformations such as rolling daily accumulations.  Do not emit
            # their output timestamps from this batch: a look-ahead state file
            # can otherwise introduce a boundary row whose accumulated fields
            # are still NaN, and the later complete row is then discarded as a
            # duplicate by ``last_emitted``.
            mask = (values >= owned_start) & (values <= owned_end)
            if last_emitted is not None:
                mask &= values > last_emitted
            if np.any(mask):
                selected = prepared.isel(time=np.flatnonzero(mask))
                last_emitted = _time_values(selected)[-1]
                yield selected, len(group)
        finally:
            ds.close()
