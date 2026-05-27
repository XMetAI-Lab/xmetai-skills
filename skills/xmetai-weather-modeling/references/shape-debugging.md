# Debugging

Use this reference for failures, tensor-shape bugs, config import issues, dataset mismatches, GPU memory problems, optional dependency errors, and tool/runtime constraints.

## Tool And Runtime Posture

- Core is Markdown plus standalone Python CLIs.
- `agents/openai.yaml` is optional metadata.
- If commands can run, prefer scripts for deterministic checks.
- If commands cannot run, follow checklists manually and say so.
- Use runtime approval/sandbox mechanisms for dataset writes, installs, long training, export, and remote operations.
- Report facts, inferences, assumptions, and pending approvals separately.

## Shape Debugging

Reason from the tensor passed into the failing module, not model family labels.

Trace: entry shape -> frame/channel flattening -> patch/window partition -> sequence length into attention/Mamba/rotary -> reshape/view/rearrange/concat/split/group -> output reshape.

Windowed rotary: use `window_h * window_w` when attention sees one local window per batch row; use `H * W` only for true full-sequence attention. If `max_seq_len // seq_len > 1` triggers grouping, full-image length on window-local input can change behavior.

Explain with side-by-side flows, concrete shapes, divergence point, and actual call-site tensor.

## Common Failures

- Config import: add repo to path/install project; report exact missing optional dependency; inspect side effects before long commands.
- Dataset: retry non-consolidated Zarr; list vars/coords; compare channels, time frequency/sort, static grid.
- Training: check output dir, resume checkpoint, distributed env/port, GPU memory, optional kernels.
- Export: confirm export forward, matching checkpoint/config data contract, ONNX opset, external data.
