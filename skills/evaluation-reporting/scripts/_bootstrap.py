# -*- coding: utf-8 -*-
"""入口脚本共用样板：skill 运行库、UTF-8 控制台、``NAME=目录`` 参数。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


SCRIPTS_ROOT = Path(__file__).resolve().parent


def bootstrap() -> Path:
    """优先加载随 skill 复制的运行库，并把 stdout / stderr 切成 UTF-8。

    运行库就在本目录下（``scripts/reporting/``），所以把 ``scripts/``
    本身插进 ``sys.path`` 即可——**不需要 pip install，也不要求从仓库根运行**。
    """
    if not (SCRIPTS_ROOT / "reporting" / "__init__.py").is_file():
        raise FileNotFoundError(f"Skill 运行库缺失，请完整复制目录：{SCRIPTS_ROOT}")
    root = str(SCRIPTS_ROOT)
    if root in sys.path:
        sys.path.remove(root)
    sys.path.insert(0, root)
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    return SCRIPTS_ROOT


def parse_model(value: str):
    """``NAME=目录`` → ``(NAME, Path)``。"""
    if "=" not in value:
        raise argparse.ArgumentTypeError(f"--model 要写成 NAME=目录，收到的是 {value!r}")
    name, directory = value.split("=", 1)
    name = name.strip()
    if not name:
        raise argparse.ArgumentTypeError(f"--model 的模型名不能是空的: {value!r}")
    return name, Path(directory.strip())
