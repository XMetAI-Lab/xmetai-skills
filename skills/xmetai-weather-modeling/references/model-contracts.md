# Model Development

Use this reference for model code, LazyConfig design, train-time behavior, losses, rollout training, and ensemble or probabilistic predictors.

## Workspace Contract

- Read local rules before edits: `AGENTS.md`, `RTK.md`, `CONTRIBUTING.md`, README.
- Check local diffs; preserve user/prior-agent changes.
- Identify whether `configs/` is in the core repo, parent workspace, or both.
- Prefer existing `xmetai/`, `configs/`, `tools/`, `scripts/`, `tests/` patterns.
- Do not commit, push, open PRs, or change remotes unless explicitly asked.
- Keep changes scoped; add focused tests for shared behavior.
- Explain with file/function/tensor/command evidence.

## Predictor Contract

New predictors should:

- Build from `base_predictor` unless the repo already has a stricter task-specific base.
- Implement `forward(data) -> loss_dict` and `test_cascade(data) -> outputs`.
- Register in the project registry and package namespace if needed.
- Keep the base predictor contracts for normalization, masks, static fields, autoregressive rollout, logging, checkpoint IO, and train/eval mode handling.

Check:

- Invalid config fails early.
- Required buffers are registered in init.
- Normalization, masks, static channels, precip transforms match the base predictor.
- Tensor boundaries have explicit shapes.
- Required optional dependencies fail clearly.
- No silent fallback changes semantics.
- Ensemble or probabilistic heads expose deterministic and stochastic paths clearly, including seed/dropout/posterior-sampling controls.

Tests should cover behavior, not only shape: init, buffers, registry, loss, reshapes, window/sequence grouping, export paths.

## LazyConfig Contract

Trainable configs usually define `train`, `optimizer`, `scheduler`, `dataloader`, and `model`.

Review:

- Reuse base config fragments.
- Set output dir, max iter, LR, batch, eval cadence.
- Align train/test dataset settings.
- Align `hist_frames`, `fcst_frames`, `interval`, `freq`.
- Align `model.in_frames`, `out_frames`, `test_frames`, evaluator times.
- Align `in_chans`, `out_chans`, `test_chans`, `test_names`, channel lists.
- Match buffers to dataset stats; add static buffers only when expected.
- User-visible names should describe the actual task/data.
- Make ensemble, rollout, variable loss weights, static fields, and perturbation settings explicit.

Scripts: `check_config_contract.py`, `summarize_lazy_config.py`. If imports fail, report exact missing dependency and fall back to static review.

## Training Design

- Before running: confirm repo, config, local diffs, stage, GPUs/nodes, batch, LR, output dir, checkpoint, dataset accessibility.
- Stages: `train`, `eval`, `export`, `ltrain`, `leval`.
- Use `build_train_command.py` to print a command; it does not launch training.
- Start model work from the repo's `base_predictor` contract. Preserve its data normalization, mask/static handling, rollout API, logging, checkpoint, and train/eval mode behavior unless the user explicitly wants a new base.
- For ensemble forecasting, define the perturbation mechanism before choosing the loss and validation metrics. Current supported patterns:
  - Ensemble training with random perturbations constrained by CRPS loss.
  - Random dropout at train and/or inference time, with explicit mode and seed handling.
  - Learned posterior distribution for perturbation sampling, with a clear prior/posterior interface and sampling count.
- Rollout training is important for autoregressive weather models. Before increasing rollout length, check memory growth from saved activations, optimizer states, temporal unroll length, and retained outputs.
- If rollout memory rises too much, consider gradient checkpointing, AMP/bfloat16, smaller micro-batches with accumulation, freezing stable submodules, detaching selected context tensors only when scientifically valid, or staged rollout-length schedules.
- Report command, config, output, checkpoint, GPUs/nodes, and validation that actually ran.
