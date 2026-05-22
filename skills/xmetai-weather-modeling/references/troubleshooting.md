# Troubleshooting

- Config import: add repo to path/install project; report exact missing optional dependency; inspect side effects before long commands.
- Dataset: retry non-consolidated Zarr; list vars/coords; compare channels, time frequency/sort, static grid.
- Training: check output dir, resume checkpoint, distributed env/port, GPU memory, optional kernels.
- Export: confirm export forward, matching checkpoint/config data contract, ONNX opset, external data.
