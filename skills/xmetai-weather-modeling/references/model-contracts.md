# Model Contracts

New predictors should:

- Follow existing hierarchy, usually base predictor inheritance.
- Implement `forward(data) -> loss_dict` and `test_cascade(data) -> outputs`.
- Register in the project registry and package namespace if needed.

Check:

- Invalid config fails early.
- Required buffers are registered in init.
- Normalization, masks, static channels, precip transforms match the base predictor.
- Tensor boundaries have explicit shapes.
- Required optional dependencies fail clearly.
- No silent fallback changes semantics.

Tests should cover behavior, not only shape: init, buffers, registry, loss, reshapes, window/sequence grouping, export paths.
