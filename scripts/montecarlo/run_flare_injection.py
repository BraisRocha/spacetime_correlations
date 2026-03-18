"""
Run background simulations and compare them
with flare-injection simulations.
"""

from pathlib import Path

import astropy.units as u
import numpy as np
from astropy.time import Time

import scipy.stats as scp

import spacetimecorr as stc
import spacetimecorr.plotting as stcp

from tqdm import tqdm

def main(seed:int) -> None:
    """
    Generate simulated event samples and compute window exposures.
    """

    # Simulation parameters
    n_events = int(1e5)
    n_simulations = int(1e3)

    # Observation interval
    t0 = Time("2026-01-01T00:00:00", scale="utc")
    tf = t0 + 1 * u.week

    # Sky window parameters (RA [deg], Dec [deg], radius [deg])
    centre = np.array([30.0, 0.0])
    radius = 3.0

    # Pierre Auger Observatory coordinates
    latitude_pa = -35.15
    longitude_pa = -69.15
    altitude_pa = 1425

    # Flare parameters
    flare_duration = 1 * u.day
    flare_sigma = 1 # degree
    

    rng_manager = stc.RNGManager(seed=seed)
    rng_events = rng_manager.get("events")
    rng_exposure = rng_manager.get("exposure")
    rng_flare = rng_manager.get("flare")

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
    outdir = project_root / "output" / "scripts" / "flare_injection"
    outdir.mkdir(parents=True, exist_ok=True)

    lambda_stats_bkg = []
    p_values_bkg = []

    lambda_flare = []
    p_values_flare = []

    for _ in tqdm(range(n_simulations), desc="Simulations"):

        parent_sample = stc.EventSample(
            n_events=n_events,
            t0=t0,
            tf=tf,
            rng=rng_events,
        )

        subsample = parent_sample.select_subsample(window=window)
        subsample.add_directional_exposure(
            window=window,
            exposure_model=exposure_model,
        )

        # Apply the statistical method
        lambda_stat, p_val = stc.lambda_estimator(sample=subsample)

        lambda_stats_bkg.append(lambda_stat)
        p_values_bkg.append(p_val)

        #Inject flare into the sample
        n_flare = int(scp.poisson.rvs(0.5 * subsample.expected_counts))

        flare = stc.Flare(
            n_events=n_flare,
            duration=flare_duration,
            t0=t0,
            tf=tf,
            centre=window.centre,
            exposure=exposure_model,
            rng=rng_flare
        )

        flare.generate_in_window(
            window=window,
            sigma=flare_sigma
        )

        subsample.inject_flare(flare=flare)

        lambda_stat, p_val = stc.lambda_estimator(
            sample=subsample)
        lambda_flare.append(lambda_stat)
        p_values_flare.append(p_val)

    # Convert to arrays
    lambda_stats_bkg = np.array(lambda_stats_bkg)
    p_values_bkg = np.array(p_values_bkg)

    lambda_flare = np.array(lambda_flare)
    p_values_flare = np.array(p_values_flare)
    
    # Plot Lambda distributions
    stcp.plot_lambda_estimator(
        {
            "Isotropy": lambda_stats_bkg,
            "Flare": lambda_flare,
        },
        outdir / "lambda.png",
    )

    # Plot p-value distributions
    stcp.plot_p_value(
        {
            "Isotropy": p_values_bkg,
            "Flare": p_values_flare,
        },
        outdir / "p_values.png",
    )

if __name__ == "__main__":
    seed = 42
    main(seed)