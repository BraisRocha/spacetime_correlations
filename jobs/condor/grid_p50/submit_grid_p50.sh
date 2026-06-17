#!/usr/bin/env bash
# Generate the parameter grid and submit it to HTCondor.
# Usage: bash jobs/condor/grid_p50/submit_grid_p50.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PARAMS="${SCRIPT_DIR}/grid_p50_params.txt"

seed=42

# --- Generate parameter grid ---
python3 <<EOF > "${PARAMS}"
import numpy as np

# dT = flare_duration / (10 years)
# log10(dT) from -4 to 0

durations = 3650 * 10**np.arange(-5.0, 0.0001, 0.1)

# intensity from 0.025 to 0.5 in steps of 0.025
intensities = np.arange(0.05, 2.0001, 0.05)

seed = ${seed}

for d in durations:
    for i in intensities:
        print(f"{d:.8g} {i:.3f} {seed}")
EOF

echo "Generated ${PARAMS} with $(wc -l < "${PARAMS}") jobs"

# --- Submit ---
SUBMISSION_ID="$(date +%Y%m%d_%H%M%S)"
echo "Submission ID: ${SUBMISSION_ID}"

condor_submit "${SCRIPT_DIR}/grid_p50.sub" \
    "submission_id=${SUBMISSION_ID}"
