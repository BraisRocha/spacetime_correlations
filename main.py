"""Run spacetime correlation simulations."""

from pathlib import Path

import astropy.units as u
import numpy as np
from astropy.time import Time

import spacetimecorr as stc


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

    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)

    for _ in range(n_simulations):
        if _ % 100 == 0:
            print(_+1)

        parent_sample = stc.EventSample(
            n_events=n_events,
            t0=t0,
            tf=tf,
            rng=rng_events,
        )
        parent_sample.sample_equatorial_coordinates()

        subsample = window.select(sample=parent_sample)
        subsample.add_directional_exposure_for_window(
            parent_sample=parent_sample,
            window=window,
            exposure_model=exposure_model,
        )

    # TODO: apply statistical methods to compute Lambda and p-value.


if __name__ == "__main__":
    seed = 42
    main(seed)

