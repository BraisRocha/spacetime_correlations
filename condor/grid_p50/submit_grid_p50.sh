#!/usr/bin/env bash
# Build the parameter grid and submit it to HTCondor as a DAG.
#
# Usage:
#   bash condor/grid_p50/submit_grid_p50.sh                 submit
#   bash condor/grid_p50/submit_grid_p50.sh --keep-scratch  submit, keep the
#                                                           scratch directory
#
# The DAG holds a single node, the one job per grid point described by
# grid_p50.sub, plus a POST script that DAGMan runs on this host once every
# one of them has finished: finalize_grid_p50.sh, which merges the per-job
# outputs into output/ and then removes the scratch directory. DAGMan waits
# in the queue itself, so that happens whether or not you stay logged in.
#
# Everything the submission produces on the way lives in
# scratch/grid_p50/<submission_id>/ and is disposable. A scratch directory
# that is still there once a submission has finished means that submission
# needs looking at.
#
# numpy is needed here to build the grid, so activate stc_env first.

set -euo pipefail

KEEP_SCRATCH=""
if [ "${1:-}" = "--keep-scratch" ]; then
    KEEP_SCRATCH="--keep-scratch"
elif [ "$#" -gt 0 ]; then
    echo "usage: $0 [--keep-scratch]" >&2
    exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

SCRATCH_DIR="${REPO_DIR}/scratch/grid_p50"
PARAMS="${SCRATCH_DIR}/params.txt"

SUBMISSION_ID="$(date +%Y%m%d_%H%M%S)"
RUN_SCRATCH="${SCRATCH_DIR}/${SUBMISSION_ID}"
DAG="${SCRATCH_DIR}/${SUBMISSION_ID}.dag"

seed=42

mkdir -p "${RUN_SCRATCH}"

# --- Clear the leftovers of submissions that finished cleanly ---
# DAGMan keeps its own log files open while a submission runs, so they cannot
# be removed by the finalize step that deletes the rest. A .dag whose scratch
# directory is gone belongs to a submission that completed, so it is the
# next submission that clears it.
for dag_file in "${SCRATCH_DIR}"/*.dag; do
    [ -e "${dag_file}" ] || continue
    old_id="$(basename "${dag_file}" .dag)"
    if [ ! -d "${SCRATCH_DIR}/${old_id}" ]; then
        rm -f "${SCRATCH_DIR}/${old_id}".dag*
    fi
done

# --- Generate parameter grid ---
# grid_p50.sub reads this by absolute path, so it keeps a fixed name; the
# copy inside the submission's scratch directory is the record of what this
# particular batch was asked to compute, and is what finalize reads back.
python3 <<EOF > "${PARAMS}"
import numpy as np

# dT = flare_duration / (10 years)
# log10(dT) from -3.9 to 0

durations = 3650 * 10**np.arange(-3.9, 0.0001, 0.1)

# intensity from 0.025 to 0.5 in steps of 0.025
intensities = np.arange(0.025, 0.5001, 0.025)

seed = ${seed}

for d in durations:
    for i in intensities:
        print(f"{d:.8g} {i:.3f} {seed}")
EOF

cp "${PARAMS}" "${RUN_SCRATCH}/params.txt"
echo "Generated ${PARAMS} with $(wc -l < "${PARAMS}") jobs"

# --- The DAG: all the jobs, then the merge ---
cat > "${DAG}" <<EOF
JOB         grid ${SCRIPT_DIR}/grid_p50.sub
VARS        grid submission_id="${SUBMISSION_ID}"
SCRIPT POST grid ${SCRIPT_DIR}/finalize_grid_p50.sh ${SUBMISSION_ID} ${KEEP_SCRATCH}
EOF

# --- Submit ---
echo "Submission ID: ${SUBMISSION_ID}"
echo "Scratch       : ${RUN_SCRATCH}"
echo "Results       : ${REPO_DIR}/output/montecarlo/grid_p50/${SUBMISSION_ID}"
condor_submit_dag "${DAG}"
