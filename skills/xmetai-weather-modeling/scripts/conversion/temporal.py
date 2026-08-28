"""Time-window aggregation primitives."""
from __future__ import annotations

from typing import Any

import numpy as np

try:
    import xarray as xr
except ImportError:  # pragma: no cover
    xr = None


def daily_aggregation(ds, config):
    """Aggregate configurable hourly variables into interval-ending daily fields."""
    if not isinstance(config, dict):
        raise SystemExit("daily_aggregation must be a mapping")
    if "time" not in ds.dims or "time" not in ds.coords:
        raise SystemExit("daily_aggregation requires a time coordinate")

    variables = config.get("variables")
    if not isinstance(variables, dict) or not variables:
        raise SystemExit("daily_aggregation variables must be a non-empty mapping")
    window_hours = int(config.get("window_hours", 24))
    label_hour = int(config.get("label_hour", 0))
    incomplete = str(config.get("incomplete", "error"))
    if window_hours <= 0 or window_hours > 24:
        raise SystemExit("daily_aggregation window_hours must be between 1 and 24")
    if label_hour < 0 or label_hour > 23:
        raise SystemExit("daily_aggregation label_hour must be between 0 and 23")
    if incomplete not in {"error", "drop"}:
        raise SystemExit("daily_aggregation incomplete must be 'error' or 'drop'")

    variable_specs = {}
    for raw_name, raw_spec in variables.items():
        name = str(raw_name)
        if not isinstance(raw_spec, dict):
            raise SystemExit(f"daily_aggregation: specification for {name!r} must be a mapping")
        operator = str(raw_spec.get("operator", "sum"))
        if operator not in {"sum", "mean", "min", "max"}:
            raise SystemExit(
                f"daily_aggregation: unsupported operator {operator!r} for {name!r}; "
                "use sum, mean, min, or max"
            )
        variable_specs[name] = {
            "operator": operator,
            "factor": float(raw_spec.get("factor", 1.0)),
            "offset": float(raw_spec.get("offset", 0.0)),
            "units": raw_spec.get("units"),
        }

    missing_vars = [name for name in variable_specs if name not in ds.data_vars]
    if missing_vars:
        raise SystemExit(f"daily_aggregation variables not found: {missing_vars}")
    for name in variable_specs:
        if "time" not in ds[name].dims:
            raise SystemExit(f"daily_aggregation: {name!r} has no time dimension")

    times = np.asarray(ds.time.values)
    if times.ndim != 1 or times.size == 0 or not np.issubdtype(times.dtype, np.datetime64):
        raise SystemExit("daily_aggregation requires a non-empty datetime64 time coordinate")
    times_ns = times.astype("datetime64[ns]")
    if np.any(np.isnat(times_ns)):
        raise SystemExit("daily_aggregation time contains NaT")
    if np.any(np.diff(times_ns) <= np.timedelta64(0, "ns")):
        raise SystemExit("daily_aggregation time must be strictly increasing and unique")
    if np.any(times_ns != times_ns.astype("datetime64[h]")):
        raise SystemExit("daily_aggregation timestamps must be aligned to whole hours")

    hour = np.timedelta64(1, "h")
    time_hours = times_ns.astype("datetime64[h]").astype(np.int64)
    available = set(time_hours.tolist())
    candidates = times_ns[np.mod(time_hours, 24) == label_hour]
    complete = []
    incomplete_ends = []
    for end in candidates:
        end_hour = int(end.astype("datetime64[h]").astype(np.int64))
        if all(end_hour - offset in available for offset in range(window_hours)):
            complete.append(end)
        else:
            incomplete_ends.append(end)

    trailing_partial = bool(candidates.size and times_ns[-1] > candidates[-1])
    leading_partial = bool(candidates.size and times_ns[0] > candidates[0] - (window_hours - 1) * hour)
    if not candidates.size:
        trailing_partial = True
    if incomplete == "error" and (incomplete_ends or leading_partial or trailing_partial):
        details = []
        if incomplete_ends:
            labels = [str(value.astype("datetime64[h]")) for value in incomplete_ends[:3]]
            details.append(f"incomplete daily window(s) ending {labels}")
        if trailing_partial:
            details.append("trailing hours do not end at 00 UTC")
        raise SystemExit(
            f"daily_aggregation requires complete {window_hours}-hour windows "
            f"ending at {label_hour:02d} UTC: " + "; ".join(details)
        )
    if not complete:
        raise SystemExit("daily_aggregation found no complete window")

    target_times = np.asarray(complete, dtype="datetime64[ns]")
    output_vars = {}
    for name in ds.data_vars:
        da = ds[name]
        if name not in variable_specs:
            output_vars[name] = da.sel(time=target_times) if "time" in da.dims else da
            continue
        spec = variable_specs[name]
        rolling = da.rolling(time=window_hours, min_periods=window_hours)
        aggregated = getattr(rolling, spec["operator"])(skipna=False).sel(time=target_times)
        aggregated = aggregated * spec["factor"] + spec["offset"]
        aggregated.attrs.update(da.attrs)
        aggregated.attrs.update(
            aggregation=spec["operator"],
            aggregation_window=f"{window_hours} hours",
            time_label="window_end",
        )
        if spec["units"] is not None:
            aggregated.attrs["units"] = str(spec["units"])
        output_vars[name] = aggregated
    return xr.Dataset(output_vars, attrs=ds.attrs)


def s2s_daily_accumulation(ds, config):
    """Backward-compatible tp/ttr shorthand for ``daily_aggregation``."""
    if not isinstance(config, dict):
        raise SystemExit("s2s_daily_accumulation must be a mapping")
    tp_name = str(config.get("tp", "tp"))
    ttr_name = str(config.get("ttr", "ttr"))
    return daily_aggregation(
        ds,
        {
            "window_hours": 24,
            "label_hour": 0,
            "incomplete": config.get("incomplete", "error"),
            "variables": {
                tp_name: {"operator": "sum", "factor": 1000.0, "units": "mm"},
                ttr_name: {"operator": "sum", "factor": 1.0 / 86400.0, "units": "W m-2"},
            },
        },
    )
