#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="${SCRIPT_DIR}/s2s_vit_smoke_config.py"

export XMETAI_CORE_DIR="${XMETAI_CORE_DIR:-/mnt/d/xmetai-core}"
export S2S_SMOKE_DATA_DIR="${S2S_SMOKE_DATA_DIR:-/mnt/d/data/s2s-test/converted/s2s.20230702-20231011.c76}"
export S2S_SMOKE_OUTPUT_DIR="${S2S_SMOKE_OUTPUT_DIR:-/tmp/xmetai-s2s-vit-smoke}"
export S2S_SMOKE_VENV="${S2S_SMOKE_VENV:-${HOME}/.venvs/xmetai-s2s-smoke}"

if [[ -n "${S2S_SMOKE_PYTHON:-}" ]]; then
    PYTHON_BIN="${S2S_SMOKE_PYTHON}"
elif [[ -x "${S2S_SMOKE_VENV}/bin/python" ]]; then
    PYTHON_BIN="${S2S_SMOKE_VENV}/bin/python"
elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python3)"
elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python)"
else
    echo "No Python environment found." >&2
    echo "Run: bash ${SCRIPT_DIR}/setup_s2s_vit_smoke_wsl.sh" >&2
    exit 2
fi

if [[ ! -f "${XMETAI_CORE_DIR}/tools/train.py" ]]; then
    echo "xmetai-core not found: ${XMETAI_CORE_DIR}" >&2
    exit 2
fi
if [[ ! -f "${S2S_SMOKE_DATA_DIR}/.zgroup" && ! -f "${S2S_SMOKE_DATA_DIR}/zarr.json" ]]; then
    echo "S2S Zarr not found: ${S2S_SMOKE_DATA_DIR}" >&2
    exit 2
fi

"${PYTHON_BIN}" - <<'PY'
import torch

if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable in this WSL Python environment")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"CUDA: {torch.version.cuda}; PyTorch: {torch.__version__}")
PY

export PYTHONPATH="${XMETAI_CORE_DIR}:${XMETAI_CORE_DIR}/configs:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128,expandable_segments:True"

echo "Core:   ${XMETAI_CORE_DIR}"
echo "Data:   ${S2S_SMOKE_DATA_DIR}"
echo "Output: ${S2S_SMOKE_OUTPUT_DIR}"
echo "Python: ${PYTHON_BIN}"
echo "Model:  embed=${S2S_SMOKE_EMBED_DIM:-128}, depths=${S2S_SMOKE_DEPTHS:-1,1,1,1}, heads=${S2S_SMOKE_NUM_HEADS:-8}"
echo "Iters:  ${S2S_SMOKE_MAX_ITER:-5}"

exec "${PYTHON_BIN}" "${XMETAI_CORE_DIR}/tools/train.py" \
    --config-file "${CONFIG_FILE}" \
    --num-gpus 1 \
    --num-machines 1
