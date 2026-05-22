#!/usr/bin/env python3
"""Build, but do not run, an XMetAI training/eval/export command."""

from __future__ import annotations

import argparse
import shlex
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True, help="Repository containing scripts/train.bash.")
    parser.add_argument("--config", "-c", required=True, help="Config path, absolute or relative to repo root.")
    parser.add_argument("--stage", choices=["train", "eval", "export", "ltrain", "leval"], default="train")
    parser.add_argument("--gpu", "-g", type=int)
    parser.add_argument("--batch", "-b", type=int)
    parser.add_argument("--lr", "-l", type=float)
    parser.add_argument("--depth-list", "-d")
    parser.add_argument("--members", type=int)
    parser.add_argument("--nnodes", "-n", type=int)
    parser.add_argument("--master-addr", "-a")
    parser.add_argument("--save", action="store_true")
    parser.add_argument("opts", nargs="*", help="Additional raw options appended to the command.")
    args = parser.parse_args()

    script = args.repo_root / "scripts" / "train.bash"
    if not script.exists():
        raise SystemExit(f"Training wrapper not found: {script}")

    cmd = ["bash", str(script), "--stage", args.stage, "-m", args.config]
    if args.gpu is not None:
        cmd += ["--gpu", str(args.gpu)]
    if args.batch is not None:
        cmd += ["--batch", str(args.batch)]
    if args.lr is not None:
        cmd += ["--lr", str(args.lr)]
    if args.depth_list:
        cmd += ["--depth-list", args.depth_list]
    if args.members is not None:
        cmd += ["--members", str(args.members)]
    if args.nnodes is not None:
        cmd += ["--nnodes", str(args.nnodes)]
    if args.master_addr:
        cmd += ["--master-addr", args.master_addr]
    if args.save:
        cmd.append("--save")
    cmd.extend(args.opts)

    print(shlex.join(cmd))


if __name__ == "__main__":
    main()
