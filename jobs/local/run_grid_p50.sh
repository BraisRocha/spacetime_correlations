#!/usr/bin/env bash
# Run the p50 grid script locally, iterating over the shared parameter grid.
# Usage: bash jobs/local/run_grid_p50.sh [grid_file]
#   grid_file defaults to jobs/condor/grid_p50/grid_p50_params.txt
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
GRID="${1:-${PROJECT_ROOT}/jobs/condor/grid_p50/grid_p50_params.txt}"

# One shared submission ID for the entire local run (mirrors Condor ClusterId).
SUBMISSION_ID="local_$(date +%Y%m%d_%H%M%S)"
JOB_ID=0

while IFS=' ' read -r duration intensity seed; do
    # Skip comments and blank lines
    [[ "$duration" =~ ^[[:space:]]*# ]] && continue
    [[ -z "${duration// /}" ]] && continue

    python "${PROJECT_ROOT}/scripts/montecarlo/run_grid_p50.py" \
        --flare-duration-days "${duration// /}" \
        --flare-intensity     "${intensity// /}" \
        --seed                "${seed// /}" \
        --job-id              "${JOB_ID}" \
        --submission-id       "${SUBMISSION_ID}"

    JOB_ID=$(( JOB_ID + 1 ))
done < "$GRID"
