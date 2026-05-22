#!/usr/bin/env python3
"""Print a concise summary of an XMetAI LazyConfig."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any


def add_repo_to_path(repo_root: Path | None) -> None:
    if not repo_root:
        return
    root = repo_root.resolve()
    for path in (root, root / "configs"):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))


def load_config(path: Path) -> Any:
    try:
        from xmetai.config import LazyConfig
    except ImportError as exc:
        raise SystemExit(f"Could not import LazyConfig: {exc}") from exc
    return LazyConfig.load(str(path))


def keys(obj: Any) -> list[str]:
    if hasattr(obj, "keys"):
        try:
            return sorted(str(k) for k in obj.keys())
        except Exception:
            pass
    return sorted(k for k in dir(obj) if not k.startswith("_"))


def get(obj: Any, key: str, default: Any = None) -> Any:
    if hasattr(obj, "get"):
        try:
            return obj.get(key, default)
        except Exception:
            pass
    try:
        return obj[key]
    except Exception:
        return getattr(obj, key, default)


def target_name(obj: Any) -> str:
    target = get(obj, "_target_")
    if target is None:
        target = get(obj, "target")
    return str(target) if target is not None else type(obj).__name__


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", "-c", type=Path, required=True, help="Path to LazyConfig Python file.")
    parser.add_argument("--repo-root", type=Path, help="Repository root to add to Python path before loading.")
    args = parser.parse_args()

    add_repo_to_path(args.repo_root)
    cfg = load_config(args.config)
    print(f"Config: {args.config}")
    print(f"Top-level keys: {', '.join(keys(cfg))}")
    for section in ("model", "dataloader", "train", "optimizer", "scheduler"):
        item = get(cfg, section)
        if item is None:
            print(f"\n{section}: <missing>")
            continue
        print(f"\n{section}: {target_name(item)}")
        section_keys = keys(item)
        print(f"  keys: {', '.join(section_keys[:40])}")
        if len(section_keys) > 40:
            print(f"  ... {len(section_keys) - 40} more")


if __name__ == "__main__":
    main()
