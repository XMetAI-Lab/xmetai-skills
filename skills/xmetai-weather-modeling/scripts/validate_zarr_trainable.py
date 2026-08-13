#!/usr/bin/env python3
"""Validate prepared Zarr stores against the core training layout.

For each Zarr store this checks, in order:

1. Single-``data`` layout with dims exactly ``(time, level|channel, lat, lon)``.
2. Sidecars readable by core ``get_data_info`` (``mean``/``std``/``weight``).
3. One sample loadable via ``MultiZarrDataset``.

Reports PASS/FAIL per store. Read-only; requires xarray, numpy, torch, and the
core package (pass ``--core-path`` or rely on an installed ``xmetai``).

Usage:

    python validate_zarr_trainable.py --stores out/*.zarr --core-path D:/xmetai-core
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import xarray as xr


def infer_freq_hours(times) -> float | None:
    import pandas as pd

    idx = pd.DatetimeIndex(times)
    if len(idx) < 2:
        return None
    deltas = idx.to_series().diff().dropna()
    if deltas.empty:
        return None
    return float(deltas.mode().iloc[0].total_seconds() / 3600.0)


def infer_years(times) -> tuple[str, str]:
    import pandas as pd

    idx = pd.DatetimeIndex(times)
    if len(idx) == 0:
        return ("19790101", "20991231")
    return (idx.min().strftime("%Y%m%d"), idx.max().strftime("%Y%m%d"))


def validate_store(store: str, core_path: str | None) -> tuple[bool, str]:
    if core_path:
        sys.path.insert(0, str(Path(core_path).resolve()))
    from xmetai.data.weather.data_util import get_data_info
    from xmetai.data.weather.multi_zarr_dataset import MultiZarrDataset

    ds = xr.open_zarr(store, consolidated=True)
    try:
        if "data" not in ds.data_vars:
            return False, "missing 'data' variable"
        dims = list(ds["data"].dims)
        if len(dims) != 4 or dims[0] != "time" or dims[2] != "latitude" or dims[3] != "longitude":
            return False, f"dims {dims} do not match (time, level|channel, lat, lon)"
        if dims[1] not in ("level", "channel"):
            return False, f"channel dim {dims[1]!r} not level/channel"
        coord = "level" if "level" in ds.coords else "channel"
        n_ch = len(ds.coords[coord])
        times = ds["time"].values
        freq = infer_freq_hours(times)
        years = infer_years(times)

        info = get_data_info([store])
        missing = {"mean", "std", "weight"} - set(info["buffers"])
        if missing:
            return False, f"sidecars missing: {sorted(missing)}"

        if freq is None:
            return False, "cannot infer time frequency"
        dset = MultiZarrDataset(
            data_paths=[store],
            hist_frames=2,
            fcst_frames=[1],
            freq=int(freq),
            training=False,
            interval=1,
            years=years,
        )
        if len(dset) == 0:
            return False, "no regular sequences found"
        sample = dset[0]
        shape = tuple(sample["inputs"].shape)
        return True, f"{n_ch} channels, freq={freq:g}h, sample inputs {shape}"
    finally:
        ds.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--stores", nargs="+", required=True, help="Zarr store paths")
    parser.add_argument("--core-path", default=None, help="path to xmetai-core (else rely on installed xmetai)")
    args = parser.parse_args(argv)

    results = []
    for store in args.stores:
        try:
            ok, detail = validate_store(store, args.core_path)
        except Exception as exc:  # pragma: no cover - varied load errors
            ok, detail = False, f"{type(exc).__name__}: {exc}"
        results.append((Path(store).name, ok, detail))

    for name, ok, detail in results:
        print(f"{'PASS' if ok else 'FAIL'}  {name}  {detail}")
    failed = sum(1 for _, ok, _ in results if not ok)
    print(f"\nsummary: {len(results) - failed}/{len(results)} ready for training")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
