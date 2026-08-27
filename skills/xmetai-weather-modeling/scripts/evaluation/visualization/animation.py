"""GIF frame conversion and animation output."""
from __future__ import annotations

import io
from pathlib import Path
from PIL import Image

from .maps import render_compare_frame


def figure_to_pil(fig, dpi: int = 120, tight: bool = True) -> Image.Image:
    """Render a matplotlib figure to a palette-mode PIL image (GIF frame)."""
    buf = io.BytesIO()
    kwargs: dict = {"format": "png", "dpi": dpi, "facecolor": "white"}
    if tight:
        kwargs["bbox_inches"] = "tight"
    fig.savefig(buf, **kwargs)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("P", palette=Image.Palette.ADAPTIVE)


def save_compare_gif(channel: str, pred, obs, levels, lat, lon, lead_times, output_dir: Path, duration_ms: int, freq: int = 24, extent=None, mode: str = "compare") -> None:
    """Animate the three-panel compare frames over leads, core-style GIF."""
    frames = [
        figure_to_pil(render_compare_frame(channel, lead, pred, obs, levels, lat, lon, lead_times, freq, extent, mode), tight=False)
        for lead in range(pred.shape[0])
    ]
    out = output_dir / f"compare_{channel}_leads.gif"
    frames[0].save(
        out,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
        optimize=False,
    )
    print(f"WROTE {out} ({len(frames)} frames)")
