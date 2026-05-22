#!/usr/bin/env python3
"""Validate explicit authorization flags before a script mutates a Zarr store."""

from __future__ import annotations

import argparse
from pathlib import Path


MUTATING_OPS = {"write", "overwrite", "append", "delete", "merge", "rechunk", "normalize", "convert"}


def normalized(path: Path) -> Path:
    return path.expanduser().resolve()


def validate(args: argparse.Namespace) -> None:
    operation = args.operation.lower()
    if operation not in MUTATING_OPS:
        raise SystemExit(f"Unknown mutating operation '{args.operation}'. Expected one of: {sorted(MUTATING_OPS)}")

    if not args.allow_write:
        raise SystemExit("Refusing Zarr mutation: pass --allow-write only after explicit user approval.")
    if args.ack_risk != "I-understand-this-mutates-zarr":
        raise SystemExit("Refusing Zarr mutation: pass --ack-risk I-understand-this-mutates-zarr after approval.")
    if not args.input:
        raise SystemExit("Refusing Zarr mutation: at least one --input store is required.")
    if args.output is None and operation not in {"delete"}:
        raise SystemExit("Refusing Zarr mutation: --output is required for non-delete operations.")

    inputs = [normalized(path) for path in args.input]
    output = normalized(args.output) if args.output else None
    if output and output in inputs and not args.allow_in_place:
        raise SystemExit("Refusing in-place Zarr mutation: output equals an input. Pass --allow-in-place only after specific approval.")

    if output and output.exists() and operation in {"write", "convert", "merge", "normalize"} and not args.overwrite:
        raise SystemExit(f"Output already exists: {output}. Use --overwrite only after approval.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--operation", required=True, help=f"One of: {', '.join(sorted(MUTATING_OPS))}")
    parser.add_argument("--input", "-i", type=Path, action="append", default=[], help="Input Zarr store. May be repeated.")
    parser.add_argument("--output", "-o", type=Path, help="Output Zarr store.")
    parser.add_argument("--allow-write", action="store_true", help="Confirms user approved this write operation.")
    parser.add_argument("--allow-in-place", action="store_true", help="Confirms user approved input mutation.")
    parser.add_argument("--overwrite", action="store_true", help="Confirms user approved replacing existing output.")
    parser.add_argument("--ack-risk", help="Must equal: I-understand-this-mutates-zarr")
    args = parser.parse_args()

    validate(args)
    print("Zarr write guard passed. This script performed no data mutation.")


if __name__ == "__main__":
    main()
