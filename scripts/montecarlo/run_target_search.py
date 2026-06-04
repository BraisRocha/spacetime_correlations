"""
Targeted multi-flare search over a SkyGrid.

Defines a set of flares at random positions inside the Pierre Auger
field of view, each with its own intensity (signal-to-noise ratio) and a
common duration, and builds one circular search window per flare with a
:class:`~spacetimecorr.SkyGrid`. For every window the script:

    1. simulates in-window isotropy via ``EventSample.in_window`` and
       attaches directional exposure,
    2. injects the flare drawn for that window,
    3. computes the Lambda estimator and a Poisson counting statistic,
       together with their p-values.

The collected outputs are length-``number_of_flares`` arrays of the two
statistics and their p-values, plus the window centres and radii.
"""

import time
from pathlib import Path

import astropy.units as u
import numpy as np
import scipy.stats as scp

from astropy.time import Time
from tqdm import tqdm

import spacetimecorr as stc
from spacetimecorr.io import setup_logger, make_run_dir, write_metadata


def main(seed: int) -> None:
    """Run a multi-target flare search.

    Lays down ``number_of_flares`` flares at random points inside Auger's
    field of view — each with its own intensity (signal-to-noise ratio)
    and a common duration — and places one circular search window around
    each via a :class:`~spacetimecorr.SkyGrid`. For every window the run
    simulates in-window isotropy, injects the corresponding flare, and
    computes both a Poisson counting statistic and the Lambda estimator.
    It collects, per window, the two statistics with their p-values along
    with the window centres and radii.
    """

    start_time = time.time()

    # ------------------------------------------------------------------
    # Simulation parameters
    # ------------------------------------------------------------------
    n_total = int(5e5)
    max_attempts = 20

    # Observational interval
    T_obs = 10 * u.year
    t0 = Time("2013-01-01T00:00:00", scale="utc")
    tf = t0 + T_obs

    # Pierre Auger Observatory
    latitude_pa = -35.15
    longitude_pa = -69.15
    altitude_pa = 1425

    # ------------------------------------------------------------------
    # RNG streams (created here so the flare layout is reproducible)
    # ------------------------------------------------------------------
    rng_manager = stc.RNGManager(seed=seed)
    rng_layout = rng_manager.get("flare_layout")   # flare positions / intensities
    rng_sample = rng_manager.get("sample")         # in-window event draw
    rng_exposure = rng_manager.get("exposure")     # directional-exposure sampling
    rng_flare = rng_manager.get("flare")           # flare time/space generation

    # ------------------------------------------------------------------
    # Flare layout and search windows
    # ------------------------------------------------------------------
    number_of_flares = 50
    flare_duration = 7 * u.day
    flare_sigma = 1.0              # angular spread of each flare [deg]
    search_radius = 2.0           # search-window radius around each flare [deg]

    # Ranges the per-flare position and intensity are drawn from.
    ra_range = (0.0, 360.0)        # [deg]
    dec_range = (-90.0, 10.0)      # [deg], kept inside Auger's field of view
    intensity_range = (0.25, 3.0)  # signal-to-noise ratio (n_flare / sqrt(expected_n))

    def sample_flare_parameters(
        n: int, rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Draw ``n`` flare centres of shape ``(n, 2)`` and intensities ``(n,)``."""
        ra = rng.uniform(*ra_range, size=n)
        dec = rng.uniform(*dec_range, size=n)
        intensity = rng.uniform(*intensity_range, size=n)
        return np.column_stack((ra, dec)), intensity

    flare_centres, flare_intensities = sample_flare_parameters(
        number_of_flares, rng_layout,
    )

    # One circular search window per flare.
    grid = stc.SkyGrid(centres=flare_centres, radii=search_radius)

    # ------------------------------------------------------------------
    # Output directory
    # ------------------------------------------------------------------
    project_root = Path(__file__).resolve().parents[2]
    base_dir = project_root / "output" / "scripts"

    outdir, sim_ID = make_run_dir(
        base_dir=base_dir,
        run_code="targeted_search",
        seed=seed,
    )

    # ------------------------------------------------------------------
    # Logger
    # ------------------------------------------------------------------
    logger = setup_logger(log_path=outdir / "run.log", name="targeted_search")

    logger.info("Starting targeted search run")
    logger.info("Simulation ID: %s", sim_ID)
    logger.info("Output directory: %s", outdir)
    logger.info("Seed: %d", seed)
    logger.info(
        "Defined %d flares | radius=%.2f deg, duration=%s, sigma=%.2f deg",
        len(grid), search_radius, flare_duration, flare_sigma,
    )
    logger.info(
        "Sampling ranges | Dec=[%.1f, %.1f] deg, S/N=[%.2f, %.2f]",
        dec_range[0], dec_range[1], intensity_range[0], intensity_range[1],
    )

    # ------------------------------------------------------------------
    # Models
    # ------------------------------------------------------------------
    observatory = stc.Observatory(
        latitude=latitude_pa, longitude=longitude_pa, altitude=altitude_pa,
    )
    exposure_model = stc.ExposureModel(
        observatory=observatory, t0=t0, tf=tf, rng=rng_exposure,
    )

    # ------------------------------------------------------------------
    # Storage
    # ------------------------------------------------------------------
    # One entry per window/flare, aligned with the order of `grid`. NaN marks
    # a window that never completed (see the failure handling below).
    lambda_array = np.full(number_of_flares, np.nan)
    n_sample_array = np.full(number_of_flares, np.nan)

    pvalues_lambda = np.full(number_of_flares, np.nan)
    pvalues_poisson = np.full(number_of_flares, np.nan)

    # Exposure-weighted expected count per window, shape (N,). Fixed across
    # attempts, so it is computed once for the whole grid.
    grid_expected_n = grid.expected_n_in_window(n_total, exposure_model)

    n_zero_flare = 0   # windows whose flare multiplicity drew zero
    n_failed = 0       # windows that exhausted max_attempts without success

    pbar = tqdm(total=number_of_flares, desc="Windows analysed")

    # Each window is analysed independently. An attempt can fail
    # stochastically — most commonly when `Flare.generate_in_window` cannot
    # land enough events inside the cap before hitting its draw budget
    # (raised as RuntimeError). Such attempts are retried up to
    # `max_attempts`; configuration/contract errors (ValueError) are left to
    # propagate so they surface loudly instead of being silently retried.
    for i, window in enumerate(grid):
        success = False
        attempt = 0

        while not success and attempt < max_attempts:
            attempt += 1

            try:
                # --- Background: isotropic events drawn inside the window ---
                sample = stc.EventSample.in_window(
                    window=window,
                    n_total=n_total,
                    exposure_model=exposure_model,
                    t0=t0,
                    tf=tf,
                    rng=rng_sample,
                )

                # --- Flare multiplicity from the per-window S/N ratio ---
                # Intensity is defined as S/N, so the flare count scales with
                # the Poisson noise sqrt(expected_n) of the background.
                mu_flare = flare_intensities[i] * np.sqrt(grid_expected_n[i])
                n_flare = int(scp.poisson.rvs(mu_flare, random_state=rng_flare))

                if n_flare == 0:
                    logger.info(
                        "Window %d (centre=[%.2f, %.2f], radius=%.2f deg): "
                        "flare multiplicity drew zero (mu_flare=%.3f); "
                        "no flare injected.",
                        i, window.centre[0], window.centre[1], window.radius,
                        mu_flare,
                    )
                    n_zero_flare += 1

                else:
                    flare = stc.Flare(
                        n_flare=n_flare,
                        duration=flare_duration,
                        t0=t0, tf=tf,
                        centre=window.centre,
                        exposure_model=exposure_model,
                        rng=rng_flare,
                    )
                    flare.generate_in_window(window=window, sigma=flare_sigma)
                    sample.inject_flare(flare=flare, mode="overdensity")

                # --- Attach exposure and compute the statistic ---
                sample.assign_directional_exposure(
                    window=window, exposure_model=exposure_model,
                )

                lambda_array[i] = stc.lambda_estimator(sample=sample)
                n_sample_array[i] = sample.n_sample

                success = True

            except RuntimeError:
                logger.exception(
                    "Window %d attempt %d/%d failed (centre=[%.2f, %.2f], "
                    "radius=%.2f deg); retrying.",
                    i, attempt, max_attempts,
                    window.centre[0], window.centre[1], window.radius,
                )
                continue

        if not success:
            n_failed += 1
            logger.error(
                "Window %d (centre=[%.2f, %.2f], radius=%.2f deg) failed after "
                "%d attempts; recorded as NaN.",
                i, window.centre[0], window.centre[1], window.radius,
                max_attempts,
            )

        pbar.update(1)

    pbar.close()

    logger.info(
        "Loop finished: %d windows | %d failed | %d zero-flare draws.",
        number_of_flares, n_failed, n_zero_flare,
    )

    # ------------------------------------------------------------------
    # Final outputs and metadata
    # ------------------------------------------------------------------
    # TODO: compute p-values, save results.npz and write_metadata(...).






