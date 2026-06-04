from pathlib import Path

import astropy.units as u
import numpy as np
import scipy.stats as scp
import time

from astropy.time import Time
from tqdm import tqdm

import spacetimecorr as stc
from spacetimecorr.io import setup_logger, make_run_dir, write_metadata

def main(seed:int) -> None:
    """Generate a multiple-targeted search. We define multiple flares within
    Auger's FoV, define windows around them using an SkyGrid and compute 
    coth a Poisson counting and the Lambda estimator. Flares' intensity 
    is defined trough a signal-to-noise ratio"""

    start_time = time.time()

    # ------------------------------------------------------------------
    # Simulation parameters
    # ------------------------------------------------------------------
    n_total = int(5e5)
    n_simulations = int(1e3)
    max_attempts = int(3 * n_simulations)

    # Observational interval
    T_obs = 10 * u.year
    t0 = Time("2013-01-01T00:00:00", scale="utc")
    tf = t0 + T_obs


    # Flare parameters
    flare_centres = np.array([
        [  0.0, -60.0],
        [ 36.0, -50.0],
        [ 72.0, -40.0],
        [108.0, -30.0],
        [144.0, -20.0],
        [180.0, -10.0],
        [216.0,   0.0],
        [252.0,   5.0],
        [288.0,  10.0],
        [324.0, -15.0],
    ])
    flare_radii = 2.
    flare_duration = 7 * u.day
    flare_sigma = 1.0 # deg
    flare_intensity =  np.array([
        1.27,
        4.63,
        2.18,
        0.74,
        3.91,
        1.55,
        4.21,
        2.87,
        0.96,
        3.34,
    ])

    # SkyGrid parameters (RA [deg], Dec [deg], radius [deg])
    grid = stc.SkyGrid(centres=flare_centres, radii= flare_radii)

    #Pierre Auger Observatory
    latitude_pa = -35.15
    longitude_pa = -69.15
    altitude_pa = 1425

    # ------------------------------------------------------------------
    # Output directory
    # ------------------------------------------------------------------
    project_root = Path(__file__).resolve().parents[2]
    base_dir = project_root / "output"/ "scripts"

    outdir, sim_ID = make_run_dir(
        base_dir=base_dir,
        run_code="targeted_search"
        seed=seed,
    )

    # ------------------------------------------------------------------
    # Logger and metadata
    # ------------------------------------------------------------------
    logger = setup_logger(log_path=outdir/"run.log", name="targeted_search")

    logger.info("Starting targeted search run")
    logger.info("Simulation ID: %s", sim_ID)
    logger.info("Output directory: %s", outdir)
    logger.info("Seed: %d", seed)

    # ------------------------------------------------------------------
    # RNGs and models
    # ------------------------------------------------------------------

    rng_manager = stc.RNGManager(seed=seed)




