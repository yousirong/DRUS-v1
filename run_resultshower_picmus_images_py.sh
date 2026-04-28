#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_ACTIVATE="${ROOT_DIR}/venv/bin/activate"
PY_SCRIPT="${ROOT_DIR}/DRUS-v1/MATLAB/Results/02_picmus/ResultShower_picmus_images.py"

if [[ -f "${VENV_ACTIVATE}" ]]; then
    # shellcheck disable=SC1090
    source "${VENV_ACTIVATE}"
fi

python3 "${PY_SCRIPT}" "$@"

