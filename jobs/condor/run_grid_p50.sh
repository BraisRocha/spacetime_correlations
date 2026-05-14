#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "${PROJECT_ROOT}/stc_venv/bin/activate"

python "${PROJECT_ROOT}/scripts/montecarlo/run_grid_p50.py" \
    --flare-duration-days "$1" \
    --flare-intensity     "$2" \
    --seed                "$3" \
    --job-id              "$4"
