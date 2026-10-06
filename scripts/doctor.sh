#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIRECTORY="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIRECTORY/activate.sh"
exec "$UV_PROJECT_ENVIRONMENT/bin/openmocap" doctor "$@"
