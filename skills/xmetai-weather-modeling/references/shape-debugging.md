# Shape Debugging

Reason from the tensor passed into the failing module, not model family labels.

Trace: entry shape -> frame/channel flattening -> patch/window partition -> sequence length into attention/Mamba/rotary -> reshape/view/rearrange/concat/split/group -> output reshape.

Windowed rotary: use `window_h * window_w` when attention sees one local window per batch row; use `H * W` only for true full-sequence attention. If `max_seq_len // seq_len > 1` triggers grouping, full-image length on window-local input can change behavior.

Explain with side-by-side flows, concrete shapes, divergence point, and actual call-site tensor.
