#!/usr/bin/env python3
"""Collect a lightweight experiment report from repository metadata."""

from __future__ import annotations

import argparse
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def run(root: Path, args: list[str]) -> str:
    try:
        result = subprocess.run(["git", *args], cwd=root, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    except OSError:
        return "<git unavailable>"
    if result.returncode != 0:
        return result.stderr.strip() or "<git command failed>"
    return result.stdout.strip() or "<empty>"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True, help="Repository root.")
    parser.add_argument("--config", type=Path, help="Config path.")
    parser.add_argument("--command", help="Training/eval/export command.")
    parser.add_argument("--output", type=Path, help="Optional Markdown report path. Prints to stdout when omitted.")
    args = parser.parse_args()

    root = args.repo_root.resolve()
    lines = [
        "# Experiment Report",
        "",
        f"- Generated: {datetime.now(timezone.utc).isoformat()}",
        f"- Repository: {root}",
        f"- Config: {args.config or '<not provided>'}",
        f"- Command: {args.command or '<not provided>'}",
        f"- Python: {sys.version.split()[0]}",
        f"- Platform: {platform.platform()}",
        f"- Git branch: {run(root, ['branch', '--show-current'])}",
        "",
        "## Git Status",
        "",
        "```text",
        run(root, ["status", "--short"]),
        "```",
        "",
        "## Latest Commit",
        "",
        "```text",
        run(root, ["log", "-1", "--oneline"]),
        "```",
        "",
    ]
    text = "\n".join(lines)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
        print(f"Wrote report to {args.output}")
    else:
        print(text)


if __name__ == "__main__":
    main()
