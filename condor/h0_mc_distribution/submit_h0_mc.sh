#!/usr/bin/env bash
# Plan the Monte Carlo distribution of Lambda per declination band and
# submit it to HTCondor as a DAG.
#
# Usage:
#   bash condor/h0_mc_distribution/submit_h0_mc.sh                 submit
#   bash condor/h0_mc_distribution/submit_h0_mc.sh --keep-scratch  submit, keep
#                                                                  the scratch
#                                                                  directory
#
# The DAG holds two nodes and a POST script:
#
#   prepare  one job (prepare.sub): zenith angles of the sample and the
#            local-to-ICRS matrix of every arrival time, shared by all the
#            band jobs so that they all build the same skies.
#   bands    one job per line of params.txt (band.sub): a range of skies of
#            one band.
#   POST     finalize_h0_mc.sh, run on this host once every band job has
#            finished: merges the bands into output/ and then removes the
#            scratch directory.
#
# Everything the submission produces on the way lives in
# scratch/h0_mc_distribution/<submission_id>/ and is disposable. A scratch
# directory that is still there once a submission has finished means that
# submission needs looking at.
#
# The plan runs here with the same interpreter as the jobs, hardcoded as in
# run_h0_mc.sh and finalize_h0_mc.sh, so no environment needs activating.

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
PIPELINE="${REPO_DIR}/scripts/analysis/mocksamples/h0_mc_distribution/h0_pipeline.py"

PYTHON='/home2/brais.rocha/.conda/envs/stc_env/bin/python'

SCRATCH_DIR="${REPO_DIR}/scratch/h0_mc_distribution"

SUBMISSION_ID="$(date +%Y%m%d_%H%M%S)"
RUN_SCRATCH="${SCRATCH_DIR}/${SUBMISSION_ID}"
DAG="${SCRATCH_DIR}/${SUBMISSION_ID}.dag"

dataset="dataset_0"
n_min=100000              # minimum number of Lambda values per band
max_skies_per_job=30000   # ~30 min for the 4-pixel ring
seed=0

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

# --- Plan: one job per line, "dataset ring k_start k_stop seed" ---
# Written to the submission's own scratch directory, where band.sub queues
# the jobs from and merge reads them back.
"${PYTHON}" "${PIPELINE}" plan \
    --dataset "${dataset}" \
    --n-min "${n_min}" \
    --max-skies-per-job "${max_skies_per_job}" \
    --seed "${seed}" \
    --output "${RUN_SCRATCH}/params.txt"

# --- The DAG: prepare, then all the band jobs, then the merge ---
cat > "${DAG}" <<EOF
JOB         prepare ${SCRIPT_DIR}/prepare.sub
VARS        prepare submission_id="${SUBMISSION_ID}" dataset="${dataset}"
JOB         bands   ${SCRIPT_DIR}/band.sub
VARS        bands   submission_id="${SUBMISSION_ID}"
PARENT      prepare CHILD bands
SCRIPT POST bands   ${SCRIPT_DIR}/finalize_h0_mc.sh ${SUBMISSION_ID} ${KEEP_SCRATCH}
EOF

# --- Submit ---
echo "Submission ID: ${SUBMISSION_ID}"
echo "Scratch       : ${RUN_SCRATCH}"
echo "Results       : ${REPO_DIR}/output/analysis/mocksamples/h0_mc_distribution/${SUBMISSION_ID}"
condor_submit_dag "${DAG}"
