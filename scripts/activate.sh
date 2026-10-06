#!/usr/bin/env bash
# Source this file: source /absolute/path/openmocap-vfx/scripts/activate.sh
_OPENMOCAP_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export OPENMOCAP_REPO="$(dirname -- "$_OPENMOCAP_SCRIPT_DIR")"
export OPENMOCAP_INSTALL_ROOT="${OPENMOCAP_INSTALL_ROOT:-$(dirname -- "$OPENMOCAP_REPO")}"
export UV_PROJECT_ENVIRONMENT="$OPENMOCAP_INSTALL_ROOT/env"
export PIP_CACHE_DIR="$OPENMOCAP_INSTALL_ROOT/cache/pip"
export UV_CACHE_DIR="$OPENMOCAP_INSTALL_ROOT/cache/uv"
export UV_PYTHON_INSTALL_DIR="$OPENMOCAP_INSTALL_ROOT/tools/python"
export UV_TOOL_DIR="$OPENMOCAP_INSTALL_ROOT/tools/uv-tools"
export UV_TOOL_BIN_DIR="$OPENMOCAP_INSTALL_ROOT/tools/bin"
export HF_HOME="$OPENMOCAP_INSTALL_ROOT/cache/huggingface"
export HF_HUB_CACHE="$OPENMOCAP_INSTALL_ROOT/cache/huggingface/hub"
export TRANSFORMERS_CACHE="$OPENMOCAP_INSTALL_ROOT/cache/huggingface/transformers"
export TORCH_HOME="$OPENMOCAP_INSTALL_ROOT/models/torch"
export XDG_CACHE_HOME="$OPENMOCAP_INSTALL_ROOT/cache/xdg"
export XDG_CONFIG_HOME="$OPENMOCAP_INSTALL_ROOT/cache/config"
export XDG_DATA_HOME="$OPENMOCAP_INSTALL_ROOT/cache/data"
export MPLCONFIGDIR="$OPENMOCAP_INSTALL_ROOT/cache/matplotlib"
export CUDA_CACHE_PATH="$OPENMOCAP_INSTALL_ROOT/cache/cuda"
export NUMBA_CACHE_DIR="$OPENMOCAP_INSTALL_ROOT/cache/numba"
export TMPDIR="$OPENMOCAP_INSTALL_ROOT/temp"
# Tiny geometry/IK matrices are slower when BLAS oversubscribes cloud CPUs.
export OPENMOCAP_NUM_THREADS="${OPENMOCAP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="$OPENMOCAP_NUM_THREADS"
export OMP_NUM_THREADS="$OPENMOCAP_NUM_THREADS"
export MKL_NUM_THREADS="$OPENMOCAP_NUM_THREADS"
export QT_LOGGING_RULES="${QT_LOGGING_RULES:-qt.qpa.*=false}"
export PATH="$OPENMOCAP_INSTALL_ROOT/tools:$OPENMOCAP_INSTALL_ROOT/tools/bin:$PATH"
if [[ -f "$UV_PROJECT_ENVIRONMENT/bin/activate" ]]; then
    source "$UV_PROJECT_ENVIRONMENT/bin/activate"
else
    printf 'Project environment is absent: run %s/scripts/bootstrap.sh\n' "$OPENMOCAP_REPO" >&2
    return 1 2>/dev/null || exit 1
fi
unset _OPENMOCAP_SCRIPT_DIR
