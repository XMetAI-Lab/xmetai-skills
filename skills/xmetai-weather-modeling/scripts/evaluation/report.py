"""Console report formatting helpers."""
from __future__ import annotations


def fmt(v: float) -> str:
    return "   -  " if not np.isfinite(v) else f"{v:.4f}"


def print_table(headers: list[str], rows: list[list[str]]) -> None:
    widths = [max(len(headers[i]), *(len(r[i]) for r in rows)) for i in range(len(headers))]
    line = "| " + " | ".join(h.ljust(widths[i]) for i, h in enumerate(headers)) + " |"
    sep = "|" + "|".join("-" * (w + 2) for w in widths) + "|"
    print(line)
    print(sep)
    for r in rows:
        print("| " + " | ".join(c.ljust(widths[i]) for i, c in enumerate(r)) + " |")
