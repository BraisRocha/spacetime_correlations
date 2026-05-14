#!/usr/bin/env bash
# Run the p50 grid script locally, iterating over the shared parameter grid.
# Usage: bash jobs/local/run_grid_p50.sh [grid_file]
#   grid_file defaults to jobs/condor/grid_p50_params.txt
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
GRID="${1:-${PROJECT_ROOT}/jobs/condor/grid_p50_params.txt}"

while IFS=, read -r duration intensity seed; do
    # Skip comments and blank lines
    [[ "$duration" =~ ^[[:space:]]*# ]] && continue
    [[ -z "${duration// }" ]] && continue

    python "${PROJECT_ROOT}/scripts/montecarlo/run_grid_p50.py" \
        --flare-duration-days "${duration// /}" \
        --flare-intensity     "${intensity// /}" \
        --seed                "${seed// /}"
done < "$GRID"
