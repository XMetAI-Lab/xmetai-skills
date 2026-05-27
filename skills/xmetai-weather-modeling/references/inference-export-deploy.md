# Evaluation And Inference

Use this reference for evaluation, test cascade, inference, ONNX export, deployment, experiment reports, and runtime checks.

## Evaluation

- Confirm config, checkpoint, output dir, dataset accessibility, GPUs/nodes, batch, and stage.
- Preserve channel order, frame order, normalization, static channels, precip transforms, spatial grid, and lead-time semantics.
- For ensemble/probabilistic models, report member count, seed policy, stochastic mode, aggregation, and deterministic baseline if available.
- Report command, config, output, checkpoint, GPUs/nodes, and validation that actually ran.
- Use `build_train_command.py` to print eval/export commands; it does not launch jobs.

Export checks:

- Export forward exists, often `onnx_export`.
- Input names/shapes/dtypes/device are correct.
- Normalization and static buffers match deployment inputs.
- Merge adapters such as LoRA if required.
- Handle ONNX external data for large weights.

Post-export: run `check_onnx_io.py --input <model.onnx>`.

Deployment must preserve channel order, frame order, normalization, static channels, precip transforms, spatial grid, and lead-time semantics.

## Experiment Reporting

- Use `collect_experiment_report.py` for lightweight summaries when available.
- Separate facts, inferences, assumptions, and pending approvals.
- Never claim train/export/deploy/runtime validation unless it actually ran.
