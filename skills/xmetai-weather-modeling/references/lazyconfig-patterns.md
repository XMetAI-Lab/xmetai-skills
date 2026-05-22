# LazyConfig Patterns

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

Scripts: `check_config_contract.py`, `summarize_lazy_config.py`. If imports fail, report exact missing dependency and fall back to static review.
