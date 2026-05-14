#!/usr/bin/bash

export HOME='/home2/brais.rocha'

python3 "/lustre/Auger/brais.rocha/spacetime_correlations/scripts/montecarlo/run_grid_p50.py" \
    --flare-duration-days "$1" \
    --flare-intensity     "$2" \
    --seed                "$3" \
    --job-id              "$4" \
    --submission-id       "$5"
