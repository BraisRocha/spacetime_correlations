#!/usr/bin/bash
# Merge one h0_mc_distribution submission from scratch/ into output/.
#
# DAGMan runs this on the submit host once every band job of the submission
# has finished (the SCRIPT POST line of the .dag). It reads only the
# filesystem, so it is also the way to redo the merge by hand:
#
#     bash condor/h0_mc_distribution/finalize_h0_mc.sh <submission_id> [--keep-scratch]
#
# Unlike run_h0_mc.sh this never leaves the submit host, so the project is
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

"${PYTHON}" "${REPO_DIR}/scripts/analysis/mocksamples/h0_mc_distribution/h0_pipeline.py" merge \
    --scratch-dir "${REPO_DIR}/scratch/h0_mc_distribution/${SUBMISSION_ID}" \
    --output-dir  "${REPO_DIR}/output/analysis/mocksamples/h0_mc_distribution/${SUBMISSION_ID}" \
    "$@"
