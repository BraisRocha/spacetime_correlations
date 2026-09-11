#!/usr/bin/bash
# Runs ONE point of the (flare duration, flare intensity) p50 grid.
# HTCondor runs this once per grid point; see grid_p50.sub.
#
# Arguments: <duration_days> <intensity> <seed> <job_id> <submission_id>
#
# The per-job files go to the scratch tree, not to output/: output/ is
# written once, by finalize_grid_p50.sh, when the whole submission is done.
#
# The project's Python is the interpreter of the conda environment stc_env,
# called by absolute path. This is what "conda activate stc_env" ends up
# doing, without needing conda itself on the worker node.

set -euo pipefail

export HOME='/home2/brais.rocha'

PYTHON='/home2/brais.rocha/.conda/envs/stc_env/bin/python'
REPO_DIR='/lustre/Auger/brais.rocha/spacetime_correlations'

"${PYTHON}" "${REPO_DIR}/scripts/montecarlo/run_grid_p50.py" \
    --flare-duration-days "$1" \
    --flare-intensity     "$2" \
    --seed                "$3" \
    --job-id              "$4" \
    --submission-id       "$5" \
    --output-base         "${REPO_DIR}/scratch"
