"""Run spacetime correlation simulations."""

from pathlib import Path

import astropy.units as u
import numpy as np
from astropy.time import Time

import spacetimecorr as stc
import spacetimecorr.plotting as stcp

from tqdm import tqdm
import time

import matplotlib.pyplot as plt


def main(seed:int) -> None:
    """
    Generate simulated event samples and compute window exposures.
    """

    # Simulation parameters
    n_events = int(1e5)
    n_simulations = int(1e4)

    # Observation interval
    t0 = Time("2026-01-01T00:00:00", scale="utc")
    tf = t0 + 1 * u.week

    # Sky window parameters (RA [deg], Dec [deg], radius [deg])
    centre = np.array([30.0, 0.0])
    radius = 2.0

    # Pierre Auger Observatory coordinates
    latitude_pa = -35.15
    longitude_pa = -69.15
    altitude_pa = 1425

    rng_manager = stc.RNGManager(seed=seed)
    rng_events = rng_manager.get("events")
    rng_exposure = rng_manager.get("exposure")

    window = stc.SkyWindow(centre=centre, radius=radius)
    observatory = stc.Observatory(
        latitude=latitude_pa,
        longitude=longitude_pa,
        altitude=altitude_pa,
    )
    exposure_model = stc.ExposureModel(
        observatory=observatory,
        t0=t0,
        tf=tf,
        rng=rng_exposure,
    )

    project_root = Path(__file__).resolve().parents[2]
    outdir = project_root / "output" / "main"
    outdir.mkdir(parents=True, exist_ok=True)

    lambda_stats_mc = []
    p_values_mc = []

    lambda_theory = []
    p_values_theory = []

    for _ in tqdm(range(n_simulations), desc="Simulations"):

        parent_sample = stc.EventSample(
            n_events=n_events,
            t0=t0,
            tf=tf,
            rng=rng_events,
        )
        parent_sample.sample_equatorial_coordinates()

        subsample = parent_sample.select_subsample(window=window)
        subsample.add_directional_exposure(
            window=window,
            exposure_model=exposure_model,
        )

        # Apply the statistical method
        lambda_stat, p_val = stc.lambda_estimator(sample=subsample)

        lambda_stats_mc.append(lambda_stat)
        p_values_mc.append(p_val)

        lambda_stat, p_val = stc.theoretical_lambda_estimator(
            sample=subsample)
        lambda_theory.append(lambda_stat)
        p_values_theory.append(p_val)


    # Convert to arrays
    lambda_stats_mc = np.array(lambda_stats_mc)
    p_values_mc = np.array(p_values_mc)

    lambda_theory = np.array(lambda_theory)
    p_values_theory = np.array(p_values_theory)


    # Theoretical distributions
    
    
    # Plot Lambda distributions
    stcp.plot_lambda_estimator(
        {
            "Monte Carlo": lambda_stats_mc,
            "Theoretical": lambda_theory,
        },
        outdir / "lambda.png",
    )

    # Plot p-value distributions
    stcp.plot_p_value(
        {
            "Monte Carlo": p_values_mc,
            "Theoretical": p_values_theory,
        },
        outdir / "p_values.png",
    )

if __name__ == "__main__":
    seed = 42
    main(seed)

