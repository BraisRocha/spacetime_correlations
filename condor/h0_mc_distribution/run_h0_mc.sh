#!/usr/bin/bash
# Runs ONE node of the h0_mc_distribution DAG: h0_pipeline.py with the
# subcommand and the arguments given (see prepare.sub and skies.sub).
#
# The per-job files go to the scratch tree, not to output/: output/ is
# written once, by finalize_h0_mc.sh, when the whole submission is done.
#
# The project's Python is the interpreter of the conda environment stc_env,
# called by absolute path. This is what "conda activate stc_env" ends up
# doing, without needing conda itself on the worker node.

set -euo pipefail

export HOME='/home2/brais.rocha'

# One thread per job, the one core HTCondor allocates: numpy's linear
# algebra would otherwise spread over all the cores of the node.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

PYTHON='/home2/brais.rocha/.conda/envs/stc_env/bin/python'
REPO_DIR='/lustre/Auger/brais.rocha/spacetime_correlations'

"${PYTHON}" "${REPO_DIR}/scripts/analysis/mocksamples/h0_mc_distribution/h0_pipeline.py" "$@"
