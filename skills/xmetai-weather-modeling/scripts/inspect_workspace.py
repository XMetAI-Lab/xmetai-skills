#!/usr/bin/env python3
"""Inspect an XMetAI-style workspace without modifying it."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any


MARKERS = ("AGENTS.md", "pyproject.toml", "xmetai", "configs", "tools", "scripts")


def run_git(root: Path, *args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=root,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except OSError:
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def discover_root(start: Path) -> Path:
    current = start.resolve()
    for candidate in (current, *current.parents):
        if (candidate / ".git").exists() or (candidate / "pyproject.toml").exists() or (candidate / "xmetai").is_dir():
            return candidate
    return current


def inspect(root: Path) -> dict[str, Any]:
    root = root.resolve()
    files = {name: (root / name).exists() for name in MARKERS}
    subdirs = sorted(p.name for p in root.iterdir() if p.is_dir() and not p.name.startswith("."))
    config_files = sorted(str(p.relative_to(root)) for p in (root / "configs").glob("*.py")) if (root / "configs").is_dir() else []
    tool_files = sorted(str(p.relative_to(root)) for p in (root / "tools").glob("*.py")) if (root / "tools").is_dir() else []

    return {
        "root": str(root),
        "markers": files,
        "subdirectories": subdirs,
        "configs": config_files,
        "tools": tool_files,
        "git_branch": run_git(root, "branch", "--show-current"),
        "git_status_short": run_git(root, "status", "--short"),
        "submodules": run_git(root, "submodule", "status"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="Workspace or repository root to inspect.")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    args = parser.parse_args()

    root = discover_root(args.root)
    report = inspect(root)

    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
        return

    print(f"Root: {report['root']}")
    print(f"Git branch: {report['git_branch'] or '<unknown>'}")
    print("\nMarkers:")
    for name, exists in report["markers"].items():
        print(f"  {'yes' if exists else ' no'}  {name}")
    print("\nConfigs:")
    for item in report["configs"][:40]:
        print(f"  {item}")
    if len(report["configs"]) > 40:
        print(f"  ... {len(report['configs']) - 40} more")
    print("\nTools:")
    for item in report["tools"][:40]:
        print(f"  {item}")
    if len(report["tools"]) > 40:
        print(f"  ... {len(report['tools']) - 40} more")
    if report["git_status_short"]:
        print("\nGit status:")
        print(report["git_status_short"])


if __name__ == "__main__":
    main()
