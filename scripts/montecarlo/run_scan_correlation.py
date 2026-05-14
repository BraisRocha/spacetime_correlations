"""
Correlation-type scan: compare Lambda sensitivity to spatial, temporal,
and spatio-temporal signals.

Compares the response of the Lambda estimator to three distinct types of
correlation signals, each isolated using a different flare injection scheme.

Methods
-------
Spatio-temporal correlation  (default mode)
    A flare replaces n_flare events drawn from the full sky, creating an
    overdensity in the target region. Both spatial and temporal anisotropies
    are introduced, exercising Lambda as intended.

Spatial correlation  (Delta t_flare = T_obs)
    Setting the flare duration equal to the total observation time removes
    any temporal anisotropy, acting Lambda on a spatial-only anisotropy.

Temporal correlation  (isotropic-window injection)
    The flare replaces n_flare events drawn exclusively from within the
    angular window, preserving the overall sky density. No spatial overdensity
    is created, acting Lambda on a temporal-only anisotropy.

Output
------
Lambda and p-value distributions for each of the three correlation regimes.
"""

from pathlib import Path

import astropy.units as u
import numpy as np
from astropy.time import Time
from tqdm import tqdm
import scipy.stats as scp
import copy
import time

import spacetimecorr as stc
from spacetimecorr.io import setup_logger, make_run_dir, write_metadata

def main(seed: int) -> None:

    start_time = time.time()

    # ------------------------------------------------------------------
    # Simulation parameters
    # ------------------------------------------------------------------
    N = int(1e6)
    n_simulations = int(1e4)
    max_attempts = int(3 * n_simulations)

    # Observation interval
    T_obs = 10 * u.year
    t0 = Time("2013-01-01T00:00:00", scale="utc")
    tf = t0 + T_obs

    # Sky window parameters (RA [deg], Dec [deg], radius [deg])
    centre = np.array([30.0, 0.0])
    radius = 1.5

    # Pierre Auger Observatory coordinates
    latitude_pa = -35.15
    longitude_pa = -69.15
    altitude_pa = 1425

    # ------------------------------------------------------------------
    # Output directory
    # ------------------------------------------------------------------
    project_root = Path(__file__).resolve().parents[2]
    base_dir = project_root / "output" / "scripts"

    outdir, sim_ID = make_run_dir(
        base_dir=base_dir,
        run_code="scan_correlation",
        seed=seed,
    )

    # ------------------------------------------------------------------
    # Logger and metadata
    # ------------------------------------------------------------------
    logger = setup_logger(
        log_path=outdir / "run.log",
        name="scan_correlation",
    )

    logger.info("Starting correlation scan run")
    logger.info("Simulation ID: %s", sim_ID)
    logger.info("Output directory: %s", outdir)
    logger.info("Seed: %d", seed)

    # ------------------------------------------------------------------
    # RNGs and models
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # Storage
    # ------------------------------------------------------------------
    lambda_bkg = []
    lambda_ST = []
    lambda_S = []
    lambda_T = []

    n_events_bkg = []
    n_events_ST = []
    n_events_S = []
    n_events_T = []

    expected_n = window.expected_n_in_window(N)

    # Flare design (constant across realisations)
    flare_duration_ST = 30 * u.day        # spatio-temporal / temporal cases
    flare_duration_S = T_obs              # spatial case (flare spans all T_obs)
    flare_sigma = 1.0                     # deg
    mu_flare = 0.2 * expected_n           # mean flare multiplicity per trial

    n_success = 0
    n_failures = 0
    attempt = 0
    n_zero_flare = 0

    # Progress bar tracks successful simulations, not attempts
    pbar = tqdm(total=n_simulations, desc="Successful simulations")

    # ------------------------------------------------------------------
    # Main simulation loop
    # ------------------------------------------------------------------
    while n_success < n_simulations and attempt < max_attempts:
        attempt += 1

        """
        Failed simulation attempts are logged and replaced by fresh random
        attempts until n_simulations successful runs are collected, or until
        max_attempts is reached.
        """

        try:
            
            parent_sample = stc.EventSample(
                n_events=N,
                t0=t0,
                tf=tf,
                rng=rng_events,
            )

            #---------------------------------------------------------
            # Isotropy Case
            #---------------------------------------------------------
            # `select_subsample` returns a new object, so the parent is
            # not mutated and no defensive copy is needed here.
            subsample = parent_sample.select_subsample(window=window)
            subsample.assign_directional_exposure(
                window=window,
                exposure_model=exposure_model,
            )

            # lambda estimator
            lambda_stat_bkg= stc.lambda_estimator(sample=subsample)
            lambda_bkg.append(lambda_stat_bkg)
            n_events_bkg.append(subsample.n_events)

            # Draw flare multiplicity once per trial; the same n_flare is used
            # for the spatio-temporal, temporal, and spatial cases.
            n_flare = int(
                scp.poisson.rvs(
                    mu_flare,
                    random_state=rng_flare,
                )
            )

            if n_flare == 0:
                # No flare events drawn this trial: the three injection cases
                # collapse to the background-only subsample. We still record
                # the result to keep the Monte-Carlo statistics unbiased.
                logger.info(
                    "Simulation attempt %d: drawn flare multiplicity is zero "
                    "(mu=%.3f). No flare injected.",
                    attempt,
                    mu_flare,
                )
                n_zero_flare += 1

                lambda_ST.append(lambda_stat_bkg)
                lambda_T.append(lambda_stat_bkg)
                lambda_S.append(lambda_stat_bkg)

                n_events_ST.append(subsample.n_events)
                n_events_T.append(subsample.n_events)
                n_events_S.append(subsample.n_events)

                n_success += 1
                pbar.update(1)
                continue

            #---------------------------------------------------------
            # Spatio-Temporal Case
            #---------------------------------------------------------
            working_sample = copy.deepcopy(parent_sample)

            flare = stc.Flare(
                    n_events=n_flare,
                    duration=flare_duration_ST,
                    t0=t0,
                    tf=tf,
                    centre=window.centre,
                    exposure_model=exposure_model,
                    rng=rng_flare,
                )
            flare.generate_in_window(
                window=window,
                sigma=flare_sigma,
            )

            # Flare injection and window cut
            working_sample.inject_flare(flare=flare)
            subsample = working_sample.select_subsample(window=window)
            subsample.assign_directional_exposure(
                window=window,
                exposure_model=exposure_model,
            )

            # lambda estimator
            lambda_stat_ST= stc.lambda_estimator(sample=subsample)
            lambda_ST.append(lambda_stat_ST)
            n_events_ST.append(subsample.n_events)

            #---------------------------------------------------------
            # Temporal Case
            #---------------------------------------------------------
            # The flare is injected on the window-selected subsample.
            subsample = parent_sample.select_subsample(window=window)
            subsample.inject_flare(flare=flare)
            subsample.assign_directional_exposure(
                window=window,
                exposure_model=exposure_model,
            )

            # lambda estimator
            lambda_stat_T= stc.lambda_estimator(sample=subsample)
            lambda_T.append(lambda_stat_T)
            n_events_T.append(subsample.n_events)

            #---------------------------------------------------------
            # Spatial Case
            #---------------------------------------------------------
            working_sample = copy.deepcopy(parent_sample)

            flare = stc.Flare(
                    n_events=n_flare,
                    duration=flare_duration_S,
                    t0=t0,
                    tf=tf,
                    centre=window.centre,
                    exposure_model=exposure_model,
                    rng=rng_flare,
                )
            flare.generate_in_window(
                window=window,
                sigma=flare_sigma,
            )

            # Flare injection and window cut
            working_sample.inject_flare(flare=flare)
            subsample = working_sample.select_subsample(window=window)
            subsample.assign_directional_exposure(
                window=window,
                exposure_model=exposure_model,
            )

            # lambda estimator
            lambda_stat_S= stc.lambda_estimator(sample=subsample)
            lambda_S.append(lambda_stat_S)
            n_events_S.append(subsample.n_events)

            n_success += 1
            pbar.update(1)

        except Exception:
            n_failures += 1
            logger.exception(
                "Simulation attempt %d failed "
                "(successes=%d, failures=%d)",
                attempt,
                n_success,
                n_failures,
            )
            continue

    pbar.close()

    logger.info(
        "Run finished: attempts=%d, successes=%d, failures=%d",
        attempt,
        n_success,
        n_failures,
    )
    if attempt > 0:
        logger.info(
            "Zero-flare realizations: %d / %d (%.2f%%)",
            n_zero_flare,
            attempt,
            100.0 * n_zero_flare / attempt,
        )

    # ------------------------------------------------------------------
    # Final checks
    # ------------------------------------------------------------------
    if n_success == 0:
        raise RuntimeError(
            "All simulation attempts failed. "
            f"See log file: {outdir / 'run.log'}"
        )

    if n_success < n_simulations:
        warning_msg = (
            f"Requested {n_simulations} successful simulations, "
            f"but only obtained {n_success} before reaching "
            f"max_attempts={max_attempts}."
        )
        logger.warning(warning_msg)
        print(f"Warning: {warning_msg}")
        print(f"See log file: {outdir / 'run.log'}")

    # ------------------------------------------------------------------
    # Convert to arrays and compute parameters
    # ------------------------------------------------------------------
    lambda_bkg = np.array(lambda_bkg)
    lambda_ST = np.array(lambda_ST)
    lambda_T = np.array(lambda_T)
    lambda_S = np.array(lambda_S)

    n_events_bkg = np.array(n_events_bkg)
    n_events_ST = np.array(n_events_ST)
    n_events_T = np.array(n_events_T)
    n_events_S = np.array(n_events_S)

    p_values_bkg = stc.lambda_marginal_sf(lambda_bkg, expected_n)
    p_values_ST = stc.lambda_marginal_sf(lambda_ST, expected_n)
    p_values_T = stc.lambda_marginal_sf(lambda_T, expected_n)
    p_values_S = stc.lambda_marginal_sf(lambda_S, expected_n)

    # ------------------------------------------------------------------
    # Save outputs and metadata
    # ------------------------------------------------------------------
    np.savez_compressed(
        outdir / "results.npz",
        lambda_bkg=lambda_bkg,
        lambda_ST=lambda_ST,
        lambda_T=lambda_T,
        lambda_S=lambda_S,
        n_events_bkg=n_events_bkg,
        n_events_ST=n_events_ST,
        n_events_T=n_events_T,
        n_events_S=n_events_S,
        p_values_bkg=p_values_bkg,
        p_values_ST=p_values_ST,
        p_values_T=p_values_T,
        p_values_S=p_values_S,
    )

    write_metadata(
        outdir=outdir,
        metadata={
            "script": Path(__file__).name,
            "run_code": "scan_correlation",
            "seed": seed,
            "n_events": N,
            "expected_n": expected_n,
            "n_simulations_requested": n_simulations,
            "n_simulations_successful": n_success,
            "max_attempts": max_attempts,
            "t0": t0.isot,
            "tf": tf.isot,
            "T_obs_days": T_obs.to_value(u.day),
            "centre_deg": centre.tolist(),
            "radius_deg": radius,
            "latitude_pa_deg": latitude_pa,
            "longitude_pa_deg": longitude_pa,
            "altitude_pa_m": altitude_pa,
        },
    )

    logger.info("Saved results to %s", outdir / "results.npz")

    end_time = time.time()
    elapsed = end_time - start_time
    logger.info(f"Simulation finished in {elapsed:.2f} seconds")

if __name__ == "__main__":
    seed = 42
    main(seed)