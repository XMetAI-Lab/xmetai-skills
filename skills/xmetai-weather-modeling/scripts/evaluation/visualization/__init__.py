"""Map, curve, composite, and animation rendering."""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "mplcache"))
os.environ.setdefault("CARTOPY_DATA_DIR", str(Path(tempfile.gettempdir()) / "cartopy_data"))
if "PROJ_DATA" not in os.environ:
    try:
        import pyproj
        os.environ["PROJ_DATA"] = pyproj.datadir.get_data_dir()
    except Exception:
        pass

import matplotlib
matplotlib.use("Agg")
