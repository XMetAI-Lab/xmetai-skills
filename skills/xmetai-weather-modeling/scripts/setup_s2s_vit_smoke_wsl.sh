#!/usr/bin/env bash
set -euo pipefail

export XMETAI_CORE_DIR="${XMETAI_CORE_DIR:-/mnt/d/xmetai-core}"
export S2S_SMOKE_VENV="${S2S_SMOKE_VENV:-${HOME}/.venvs/xmetai-s2s-smoke}"

if [[ ! -f "${XMETAI_CORE_DIR}/pyproject.toml" ]]; then
    echo "xmetai-core not found: ${XMETAI_CORE_DIR}" >&2
    exit 2
fi

if ! command -v python3 >/dev/null 2>&1; then
    if ! command -v sudo >/dev/null 2>&1 || ! command -v apt-get >/dev/null 2>&1; then
        echo "python3 is missing and this script cannot install it without sudo + apt-get" >&2
        exit 2
    fi
    sudo apt-get update
    sudo apt-get install -y python3 python3-venv python3-pip
fi

if ! python3 -m venv --help >/dev/null 2>&1; then
    sudo apt-get update
    sudo apt-get install -y python3-venv
fi

python3 -m venv "${S2S_SMOKE_VENV}"
PYTHON_BIN="${S2S_SMOKE_VENV}/bin/python"

"${PYTHON_BIN}" -m pip install --upgrade pip setuptools wheel
TORCH_INDEX_URL="${S2S_TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"
PYTHON_MINOR="$("${PYTHON_BIN}" -c 'import sys; print(sys.version_info.minor)')"
if (( PYTHON_MINOR >= 14 )); then
    # PyTorch 2.7 does not publish CPython 3.14 wheels.  2.9.1 is the
    # earliest stable cu128 version available for the Python shipped by this
    # WSL distribution and still satisfies xmetai-core's torch>=2.7 contract.
    TORCH_VERSION="${S2S_SMOKE_TORCH_VERSION:-2.9.1}"
else
    TORCH_VERSION="${S2S_SMOKE_TORCH_VERSION:-2.7.0}"
fi
echo "Installing torch ${TORCH_VERSION} for Python 3.${PYTHON_MINOR}"
echo "Torch index: ${TORCH_INDEX_URL}"
"${PYTHON_BIN}" -m pip install \
    "torch==${TORCH_VERSION}" \
    --index-url "${TORCH_INDEX_URL}" \
    --progress-bar on \
    --timeout 600 \
    --retries 10
"${PYTHON_BIN}" -m pip install -e "${XMETAI_CORE_DIR}" --progress-bar on --timeout 600 --retries 10
"${PYTHON_BIN}" -m pip install "zarr<3" --progress-bar on --timeout 600 --retries 10

"${PYTHON_BIN}" - <<'PY'
import torch

print(f"Python environment ready: {torch.__version__}")
print(f"CUDA runtime in wheel: {torch.version.cuda}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
PY

echo "Environment: ${S2S_SMOKE_VENV}"
echo "Next: bash /mnt/d/xmetai-skills/skills/xmetai-weather-modeling/scripts/run_s2s_vit_smoke_wsl.sh"
