#!/usr/bin/bash
# Merge one grid_p50 submission from scratch/ into output/.
#
# DAGMan runs this on the submit host once every job of the submission has
# finished (the SCRIPT POST line of the .dag). It reads only the filesystem,
# so it is also the way to redo the merge by hand:
#
#     bash condor/grid_p50/finalize_grid_p50.sh <submission_id> [--keep-scratch]
#
# Unlike run_grid_p50.sh this never leaves the submit host, so the project is
# located from this script's own path. The interpreter is the same hardcoded
# one, and the two scripts are the only places that name it.

set -euo pipefail

if [ "$#" -lt 1 ]; then
    echo "usage: $0 <submission_id> [--keep-scratch]" >&2
    exit 2
fi

SUBMISSION_ID="$1"
shift

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

PYTHON='/home2/brais.rocha/.conda/envs/stc_env/bin/python'

"${PYTHON}" "${SCRIPT_DIR}/finalize_grid_p50.py" \
    --scratch-dir "${REPO_DIR}/scratch/grid_p50/${SUBMISSION_ID}" \
    --output-dir  "${REPO_DIR}/output/montecarlo/grid_p50/${SUBMISSION_ID}" \
    "$@"
