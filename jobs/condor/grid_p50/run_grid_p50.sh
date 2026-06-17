#!/usr/bin/bash
set -euo pipefail

export HOME='/home2/brais.rocha'

# Activate the project virtualenv that lives on the shared (Lustre) filesystem.
# This script is copied to the worker's scratch dir, so the venv must be
# referenced by absolute path, not relative to this script.
REPO_DIR='/lustre/Auger/brais.rocha/spacetime_correlations'
source "${REPO_DIR}/stc_venv/bin/activate"

python "${REPO_DIR}/scripts/montecarlo/run_grid_p50.py" \
    --flare-duration-days "$1" \
    --flare-intensity     "$2" \
    --seed                "$3" \
    --job-id              "$4" \
    --submission-id       "$5"
