#!/usr/bin/env bash
# Submit the blind search on the mock samples to HTCondor, one job per sample.
#
# Usage:
#   bash condor/mocksamples/submit_blind_search_mock.sh           every sample
#   bash condor/mocksamples/submit_blind_search_mock.sh 0 3 7     dataset_0,
#                                                                 dataset_3,
#                                                                 dataset_7
#
# With no arguments, every .dat file in inputs/mocksamples/dataset_stcwg_vert/
# is submitted.
#
# Each job writes its own complete run to
# output/analysis/blind_search_mock/<submission_id>/<dataset>/, so there is
# no merge step and no DAG. The HTCondor logs (.log, and one .out/.err per
# dataset) go to scratch/mocksamples/<submission_id>/, together with the list
# of datasets this batch was asked to run.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

INPUT_DIR="${REPO_DIR}/inputs/mocksamples/dataset_stcwg_vert"
SCRATCH_DIR="${REPO_DIR}/scratch/mocksamples"
DATASETS="${SCRATCH_DIR}/datasets.txt"

SUBMISSION_ID="$(date +%Y%m%d_%H%M%S)"
RUN_SCRATCH="${SCRATCH_DIR}/${SUBMISSION_ID}"

mkdir -p "${SCRATCH_DIR}"

# --- List of datasets ---
# blind_search_mock.sub reads this by absolute path, so it keeps a fixed
# name; the copy inside the submission's scratch directory is the record of
# what this particular batch was asked to run.
if [ "$#" -eq 0 ]; then
    for f in "${INPUT_DIR}"/*.dat; do
        basename "${f}" .dat
    done | sort -V > "${DATASETS}"
else
    : > "${DATASETS}"
    for n in "$@"; do
        if [ ! -f "${INPUT_DIR}/dataset_${n}.dat" ]; then
            echo "No such sample: ${INPUT_DIR}/dataset_${n}.dat" >&2
            exit 2
        fi
        echo "dataset_${n}" >> "${DATASETS}"
    done
fi

if [ ! -s "${DATASETS}" ]; then
    echo "No samples to submit in ${INPUT_DIR}" >&2
    exit 2
fi

mkdir -p "${RUN_SCRATCH}"
cp "${DATASETS}" "${RUN_SCRATCH}/datasets.txt"
echo "Generated ${DATASETS} with $(wc -l < "${DATASETS}") jobs"

# --- Submit ---
echo "Submission ID: ${SUBMISSION_ID}"
echo "Logs          : ${RUN_SCRATCH}"
echo "Results       : ${REPO_DIR}/output/analysis/blind_search_mock/${SUBMISSION_ID}"
condor_submit "submission_id=${SUBMISSION_ID}" "${SCRIPT_DIR}/blind_search_mock.sub"
