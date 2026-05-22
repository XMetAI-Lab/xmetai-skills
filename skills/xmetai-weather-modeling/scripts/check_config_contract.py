#!/usr/bin/env python3
"""Load an XMetAI LazyConfig and perform lightweight contract checks."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def add_repo_to_path(repo_root: Path | None) -> None:
    if repo_root is None:
        return
    root = repo_root.resolve()
    for path in (root, root / "configs"):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))


def load_config(config: Path) -> Any:
    try:
        from xmetai.config import LazyConfig
    except ImportError as exc:
        raise SystemExit(f"Could not import xmetai.config.LazyConfig: {exc}") from exc
    return LazyConfig.load(str(config))


def has_key(obj: Any, key: str) -> bool:
    if isinstance(obj, dict):
        return key in obj
    try:
        return key in obj
    except Exception:
        return hasattr(obj, key)


def get_key(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    try:
        return obj[key]
    except Exception:
        return getattr(obj, key, default)


def stringify(value: Any) -> str:
    if value is None:
        return "<missing>"
    if isinstance(value, (str, int, float, bool)):
        return str(value)
    return type(value).__name__


def inspect_contract(cfg: Any) -> dict[str, Any]:
    required = ["train", "optimizer", "scheduler", "dataloader", "model"]
    report: dict[str, Any] = {"required": {}, "warnings": [], "model": {}, "train": {}, "dataloader": {}}

    for key in required:
        report["required"][key] = has_key(cfg, key)
    missing = [k for k, present in report["required"].items() if not present]
    if missing:
        report["warnings"].append(f"Missing top-level config fields: {missing}")

    model = get_key(cfg, "model")
    train = get_key(cfg, "train")
    dataloader = get_key(cfg, "dataloader")

    for key in ("in_frames", "out_frames", "test_frames", "freq", "in_chans", "out_chans", "test_chans", "test_names", "buffers"):
        if model is not None and has_key(model, key):
            value = get_key(model, key)
            try:
                if key in {"test_frames", "test_chans", "test_names"}:
                    value = list(value)
                elif key == "buffers":
                    value = sorted(value.keys()) if hasattr(value, "keys") else stringify(value)
            except Exception:
                value = stringify(value)
            report["model"][key] = value

    for key in ("output_dir", "max_iter", "init_checkpoint", "device_type", "use_ddp", "use_fsdp", "use_fsdp2"):
        if train is not None and has_key(train, key):
            report["train"][key] = stringify(get_key(train, key))

    if dataloader is not None:
        for split in ("train", "test", "evaluator"):
            report["dataloader"][split] = has_key(dataloader, split)

    in_chans = report["model"].get("in_chans")
    out_chans = report["model"].get("out_chans")
    test_chans = report["model"].get("test_chans")
    test_names = report["model"].get("test_names")
    if isinstance(test_chans, list) and isinstance(test_names, list) and len(test_chans) != len(test_names):
        report["warnings"].append("model.test_chans and model.test_names lengths differ.")
    if isinstance(in_chans, int) and isinstance(out_chans, int) and in_chans != out_chans:
        report["warnings"].append("model.in_chans and model.out_chans differ; verify this is intentional.")

    return report


def print_text(report: dict[str, Any]) -> None:
    print("Required fields:")
    for key, present in report["required"].items():
        print(f"  {'yes' if present else ' no'}  {key}")
    print("\nModel:")
    for key, value in report["model"].items():
        print(f"  {key}: {value}")
    print("\nTrain:")
    for key, value in report["train"].items():
        print(f"  {key}: {value}")
    print("\nDataloader:")
    for key, value in report["dataloader"].items():
        print(f"  {key}: {value}")
    if report["warnings"]:
        print("\nWarnings:")
        for item in report["warnings"]:
            print(f"  - {item}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", "-c", type=Path, required=True, help="Path to LazyConfig Python file.")
    parser.add_argument("--repo-root", type=Path, help="Repository root to add to Python path before loading.")
    parser.add_argument("--json", action="store_true", help="Print JSON.")
    args = parser.parse_args()

    add_repo_to_path(args.repo_root)
    cfg = load_config(args.config)
    report = inspect_contract(cfg)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print_text(report)


if __name__ == "__main__":
    main()
