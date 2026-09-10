#!/usr/bin/env bash
# Generate the parameter grid and submit it to HTCondor.
# Usage: bash condor/grid_p50/submit_grid_p50.sh
#
# Runs on the submit host and needs numpy to build the grid, so activate
# stc_env first. The jobs themselves do not depend on it: run_grid_p50.sh
# calls the environment's interpreter by absolute path.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PARAMS="${SCRIPT_DIR}/grid_p50_params.txt"
LOG_DIR="${REPO_DIR}/logs/condor/grid_p50"

seed=42

# --- Generate parameter grid ---
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

echo "Generated ${PARAMS} with $(wc -l < "${PARAMS}") jobs"

# --- Log directory ---
# Condor does not create it: without it the jobs go on hold on the spot.
mkdir -p "${LOG_DIR}"

# --- Submit ---
SUBMISSION_ID="$(date +%Y%m%d_%H%M%S)"
echo "Submission ID: ${SUBMISSION_ID}"

condor_submit "${SCRIPT_DIR}/grid_p50.sub" \
    "submission_id=${SUBMISSION_ID}"
