"""Shared map preparation and rendering primitives."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature

CMAP_BY_VAR = {
    "sst": "RdYlBu_r", "t2m": "RdYlBu_r", "msl": "viridis",
    "z500": "viridis", "u200": "RdBu_r", "u850": "RdBu_r",
    "ttr": "magma", "tp": "Blues",
}


def lead_label(index: int, freq: int) -> str:
    if freq >= 24:
        return f"Day {index + 1}"
    return f"{freq * (index + 1)}h"


def prepare_grid(lon: np.ndarray, lat: np.ndarray, field: np.ndarray):
    """Convert 0..360 lon to -180..180 and sort, matching the pred file convention."""
    lon = np.asarray(lon, dtype=float)
    lat = np.asarray(lat, dtype=float)
    field = np.asarray(field, dtype=float)
    if lon.max() > 180:
        if lon.size > 1 and np.isclose(lon[-1], 360.0):
            lon = lon[:-1]
            field = field[..., :-1]
        lon = ((lon + 180.0) % 360.0) - 180.0
        order = np.argsort(lon)
        lon = lon[order]
        field = field[..., order]
    return lon, lat, field


def render_map(ax, field, lat, lon, title: str, cmap: str, vmin: float, vmax: float, extent=None):
    lon, lat, field = prepare_grid(lon, lat, field)
    if extent:
        ax.set_extent(extent, crs=ccrs.PlateCarree())
    else:
        ax.set_global()
    ax.add_feature(cfeature.OCEAN.with_scale("110m"), facecolor="#d7e9f7", zorder=0)
    ax.add_feature(cfeature.LAND.with_scale("110m"), facecolor="#f2efe9", edgecolor="none", zorder=1)
    ax.coastlines(linewidth=0.5, resolution="110m", zorder=2)
    gl = ax.gridlines(draw_labels=True, linewidth=0.3, color="gray", alpha=0.5, linestyle="--")
    gl.top_labels = False
    gl.right_labels = False
    mesh = ax.pcolormesh(
        lon,
        lat,
        field,
        transform=ccrs.PlateCarree(),
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        shading="auto",
        rasterized=True,
    )
    ax.set_title(title, fontsize=10)
    return mesh


def robust_limits(field: np.ndarray):
    values = field[np.isfinite(field)]
    if values.size == 0:
        return 0.0, 1.0
    low = float(np.percentile(values, 2))
    high = float(np.percentile(values, 98))
    if low == high:
        high = low + 1e-6
    return low, high
