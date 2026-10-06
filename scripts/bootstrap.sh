#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
OPENMOCAP_INSTALL_ROOT="${OPENMOCAP_INSTALL_ROOT:-$(dirname -- "$REPO_ROOT")}"; export OPENMOCAP_INSTALL_ROOT
printf 'INSTALL_ROOT=%s\nREPOSITORY=%s\nENVIRONMENT=%s/env\nTOOLS=%s/tools\nCACHE=%s/cache\nMODELS=%s/models\n' "$OPENMOCAP_INSTALL_ROOT" "$REPO_ROOT" "$OPENMOCAP_INSTALL_ROOT" "$OPENMOCAP_INSTALL_ROOT" "$OPENMOCAP_INSTALL_ROOT" "$OPENMOCAP_INSTALL_ROOT"
mkdir -p "$OPENMOCAP_INSTALL_ROOT"/{env,tools,models/body,models/pose,models/segmentation,cache,downloads,temp,projects}
export PIP_CACHE_DIR="$OPENMOCAP_INSTALL_ROOT/cache/pip"
export UV_CACHE_DIR="$OPENMOCAP_INSTALL_ROOT/cache/uv"
export UV_PROJECT_ENVIRONMENT="$OPENMOCAP_INSTALL_ROOT/env"
export UV_PYTHON_INSTALL_DIR="$OPENMOCAP_INSTALL_ROOT/tools/python"
export TMPDIR="$OPENMOCAP_INSTALL_ROOT/temp"
PYTHON_EXECUTABLE="${OPENMOCAP_PYTHON:-$(command -v python3)}"
"$PYTHON_EXECUTABLE" -c 'import sys; assert (3,11)<=sys.version_info[:2]<(3,14), "Python 3.11–3.13 is required"'
UV_EXECUTABLE="$OPENMOCAP_INSTALL_ROOT/tools/uv"
if [[ ! -x "$UV_EXECUTABLE" ]]; then
    # A copy of an existing executable is isolated; no system environment is modified.
    if command -v uv >/dev/null 2>&1 && [[ "$(uv --version | cut -d' ' -f2)" == "0.12.19" ]]; then
        cp "$(command -v uv)" "$UV_EXECUTABLE"
    else
        "$PYTHON_EXECUTABLE" -m venv "$OPENMOCAP_INSTALL_ROOT/tools/uv-env"
        "$OPENMOCAP_INSTALL_ROOT/tools/uv-env/bin/python" -m pip install --disable-pip-version-check 'uv==0.12.19'
        cp "$OPENMOCAP_INSTALL_ROOT/tools/uv-env/bin/uv" "$UV_EXECUTABLE"
    fi
fi
"$UV_EXECUTABLE" sync --frozen --project "$REPO_ROOT" --python "$PYTHON_EXECUTABLE"
source "$REPO_ROOT/scripts/activate.sh"
"$UV_PROJECT_ENVIRONMENT/bin/python" "$REPO_ROOT/scripts/install_launchers.py"
"$UV_PROJECT_ENVIRONMENT/bin/openmocap" doctor
printf '\nReady. Launch: %s/OpenMocap.sh\nActivate: source %s/scripts/activate.sh\n' "$OPENMOCAP_INSTALL_ROOT" "$REPO_ROOT"
