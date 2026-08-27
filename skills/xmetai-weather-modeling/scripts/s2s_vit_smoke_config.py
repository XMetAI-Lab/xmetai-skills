"""Small standalone LazyConfig for an S2S ViT training smoke test.

This file intentionally lives outside xmetai-core.  The WSL launcher adds the
core checkout to PYTHONPATH and points this config at the converted C76 store.
"""

from __future__ import annotations

import os
from pathlib import Path

from xmetai import LazyCall as L
from xmetai import LazyConfig, MODELS, get_data_info


CORE_DIR = Path(os.environ.get("XMETAI_CORE_DIR", "/mnt/d/xmetai-core")).resolve()
DATA_DIR = Path(
    os.environ.get(
        "S2S_SMOKE_DATA_DIR",
        "/mnt/d/data/s2s-test/converted/s2s.20230702-20231011.c76",
    )
).resolve()
OUTPUT_DIR = Path(os.environ.get("S2S_SMOKE_OUTPUT_DIR", "/tmp/xmetai-s2s-vit-smoke")).resolve()

MAX_ITER = int(os.environ.get("S2S_SMOKE_MAX_ITER", "5"))
EMBED_DIM = int(os.environ.get("S2S_SMOKE_EMBED_DIM", "128"))
NUM_HEADS = int(os.environ.get("S2S_SMOKE_NUM_HEADS", "8"))
DEPTHS = [int(value) for value in os.environ.get("S2S_SMOKE_DEPTHS", "1,1,1,1").split(",")]
if len(DEPTHS) != 4 or any(value <= 0 for value in DEPTHS):
    raise ValueError("S2S_SMOKE_DEPTHS must contain four positive integers")
if EMBED_DIM <= 0 or NUM_HEADS <= 0 or EMBED_DIM % NUM_HEADS != 0:
    raise ValueError("S2S_SMOKE_EMBED_DIM must be positive and divisible by S2S_SMOKE_NUM_HEADS")

data_info = get_data_info([str(DATA_DIR)])
in_names = data_info["in_names"]
out_names = data_info["out_names"]
buffers = data_info["buffers"]
logid = data_info["logid"]

if len(in_names) != 76 or len(out_names) != 76:
    raise ValueError(f"Expected a C76 dataset, got {len(in_names)} inputs and {len(out_names)} outputs")

test_names = [name for name in ("z500", "t2m", "ttr", "tp") if name in out_names]
test_chans = [out_names.index(name) for name in test_names]
test_frames = [1]

train = LazyConfig.load(str(CORE_DIR / "common" / "train.py")).train
train.output_dir = str(OUTPUT_DIR)
train.max_iter = MAX_ITER
train.checkpointer.period = MAX_ITER
train.eval_period = MAX_ITER
train.eval_after_train = True
train.log_period = 1
train.device_type = "cuda"
train.use_ddp = False
train.use_fsdp = False
train.use_fsdp2 = False
train.activation_checkpointing = True
train.use_ema = False
train.memory_trace = True
train.teacher_forcing.teach_iters = -1
train.val_metric = "RMSE/z500"

optimizer = LazyConfig.load(str(CORE_DIR / "common" / "optim.py")).AdamW
optimizer.lr = float(os.environ.get("S2S_SMOKE_LR", "0.0001"))
optimizer.weight_decay = 0.01
scheduler = LazyConfig.load(str(CORE_DIR / "common" / "schedule.py")).cosine_lr

dataloader = LazyConfig.load(
    str(CORE_DIR / "common" / "dataloader.py"), "zarr_dataloader"
)(
    data_paths=[str(DATA_DIR)],
    in_frames=2,
    train_fcst_frames=1,
    test_frames=test_frames,
    freq=24,
    train_times=("20230702", "20230731"),
    test_times=("20230801", "20230808"),
    total_batch_size=1,
    train_workers=0,
    test_workers=0,
    train_pin_memory=False,
    test_pin_memory=False,
    eval_test_names=test_names,
)

model = L(MODELS.get("ViTModel"))(
    embed_dim=EMBED_DIM,
    depths=DEPTHS,
    num_heads=NUM_HEADS,
    patch_size=2,
    image_size=(120, 240),
    window_size=10,
    drop_rate=0.0,
    in_chans=76,
    out_chans=76,
    const_chans=2,
    in_frames=2,
    out_frames=1,
    freq=24,
    accumid=logid,
    logid=logid,
    maskid=[],
    zero_accum=False,
    test_frames=test_frames,
    test_chans=test_chans,
    test_names=test_names,
    step_range=[1],
    members=1,
    save_dir=str(OUTPUT_DIR / "pred"),
    layer_type=["swin", "swin", "swin", "swin"],
    pinn_enable=False,
    in_names=in_names,
    pinn_use_denorm=True,
    pertub_mixup=0.0,
    pertub_mask=0.0,
)
model.buffers = buffers
model.precision = "bf16"

