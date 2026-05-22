# Inference Export Deploy

Export checks:

- Export forward exists, often `onnx_export`.
- Input names/shapes/dtypes/device are correct.
- Normalization and static buffers match deployment inputs.
- Merge adapters such as LoRA if required.
- Handle ONNX external data for large weights.

Post-export: run `check_onnx_io.py --input <model.onnx>`.

Deployment must preserve channel order, frame order, normalization, static channels, precip transforms, spatial grid, and lead-time semantics.
