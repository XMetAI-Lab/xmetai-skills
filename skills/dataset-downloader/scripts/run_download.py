"""Dispatch to Datakit downloaders without duplicating their implementation."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


DOWNLOADERS = {
    "cds": Path("examples/dataset_downloader/cds_download/cds_download.py"),
    "marine": Path(
        "examples/dataset_downloader/copernicusmarine_download/copernicusmarine_download.py"
    ),
}


def repository_root(explicit: Path | None) -> Path:
    candidates = (explicit.expanduser().resolve(),) if explicit else Path(__file__).resolve().parents
    for candidate in candidates:
        if (candidate / "pyproject.toml").is_file() and all(
            (candidate / script).is_file() for script in DOWNLOADERS.values()
        ):
            return candidate
    raise ValueError("Datakit repository not found; supply a valid --repo-root")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("batch", type=Path, help="Existing Batch YAML")
    parser.add_argument("--source", choices=DOWNLOADERS, required=True)
    parser.add_argument("--repo-root", type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="Preview offline (default)")
    mode.add_argument("--plan-only", action="store_true", help="Write only the target plan")
    mode.add_argument("--execute", action="store_true", help="Start authenticated downloads")
    parser.add_argument("--force-redownload", action="store_true")
    args = parser.parse_args()
    if args.force_redownload and not args.execute:
        parser.error("--force-redownload requires --execute")
    try:
        root = repository_root(args.repo_root)
    except ValueError as error:
        parser.error(str(error))
    batch = args.batch.expanduser().resolve()
    if not batch.is_file():
        parser.error(f"Batch YAML not found: {batch}")
    command = [sys.executable, str(root / DOWNLOADERS[args.source]), str(batch)]
    if args.plan_only:
        command.append("--plan-only")
    elif not args.execute:
        command.append("--dry-run")
    if args.force_redownload:
        command.append("--force-redownload")
    return subprocess.run(command, check=False).returncode


if __name__ == "__main__":
    sys.exit(main())
