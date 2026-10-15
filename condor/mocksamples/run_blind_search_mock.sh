#!/usr/bin/bash
# Runs the blind search on ONE mock sample.
# HTCondor runs this once per dataset; see blind_search_mock.sub.
#
# Arguments: <dataset> <submission_id>
#
# Each job writes a complete run (results.npz, metadata.json, run.log) to
# output/analysis/blind_search_mock/<submission_id>/<dataset>/. There is
# nothing to merge, so unlike grid_p50 there is no scratch tree.
#
# The project's Python is the interpreter of the conda environment stc_env,
# called by absolute path. This is what "conda activate stc_env" ends up
# doing, without needing conda itself on the worker node.

set -euo pipefail

export HOME='/home2/brais.rocha'

PYTHON='/home2/brais.rocha/.conda/envs/stc_env/bin/python'
REPO_DIR='/lustre/Auger/brais.rocha/spacetime_correlations'

"${PYTHON}" "${REPO_DIR}/scripts/analysis/mocksamples/run_blind_search_mock.py" \
    --dataset       "$1" \
    --submission-id "$2"
