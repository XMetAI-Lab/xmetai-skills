#!/usr/bin/env python3
"""Summarize ONNX graph inputs and outputs."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any


def dim_to_str(dim: Any) -> str:
    if getattr(dim, "dim_param", ""):
        return dim.dim_param
    if getattr(dim, "dim_value", 0):
        return str(dim.dim_value)
    return "?"


def dtype_name(elem_type: int) -> str:
    try:
        import onnx

        return onnx.TensorProto.DataType.Name(elem_type)
    except Exception:
        return str(elem_type)


def describe_value(value: Any) -> str:
    tensor = value.type.tensor_type
    shape = [dim_to_str(dim) for dim in tensor.shape.dim]
    return f"{value.name}: dtype={dtype_name(tensor.elem_type)} shape=[{', '.join(shape)}]"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", "-i", type=Path, required=True, help="Path to ONNX model.")
    args = parser.parse_args()

    if not args.input.exists():
        raise SystemExit(f"ONNX file does not exist: {args.input}")
    try:
        import onnx
    except ImportError as exc:
        raise SystemExit("onnx is required for this script.") from exc

    model = onnx.load(str(args.input), load_external_data=False)
    print(f"Model: {args.input}")
    print(f"IR version: {model.ir_version}")
    print("Opsets:")
    for opset in model.opset_import:
        domain = opset.domain or "ai.onnx"
        print(f"  {domain}: {opset.version}")
    print("\nInputs:")
    for value in model.graph.input:
        print(f"  {describe_value(value)}")
    print("\nOutputs:")
    for value in model.graph.output:
        print(f"  {describe_value(value)}")

    external_refs = []
    for initializer in model.graph.initializer:
        if initializer.data_location == onnx.TensorProto.EXTERNAL:
            external_refs.append(initializer.name)
    if external_refs:
        print(f"\nExternal initializers: {len(external_refs)}")
        for name in external_refs[:20]:
            print(f"  {name}")
        if len(external_refs) > 20:
            print(f"  ... {len(external_refs) - 20} more")


if __name__ == "__main__":
    main()
