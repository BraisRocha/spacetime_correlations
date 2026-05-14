"""
Compare background-only and flare-injected Lambda distributions.
"""

from pathlib import Path

import astropy.units as u
import numpy as np
import scipy.stats as scp

from astropy.time import Time
from tqdm import tqdm
import time

import spacetimecorr as stc
from spacetimecorr.io import setup_logger, make_run_dir, write_metadata


def main(seed: int) -> None:
    """
    Generate simulated event samples and compare background-only and
    flare-injected simulations.

    Failed simulation attempts are logged and replaced by fresh random
    attempts until n_simulations successful runs are collected, or until
    max_attempts is reached.
    """

    start_time = time.time()

    # ------------------------------------------------------------------
    # Simulation parameters
    # ------------------------------------------------------------------
    n_events = int(1e5)
    n_simulations = int(1e3)
    max_attempts = int(3 * n_simulations)

    # Observation interval
    T_obs = 10 * u.year
    t0 = Time("2013-01-01T00:00:00", scale="utc")
    tf = t0 + T_obs

    # Sky window parameters (RA [deg], Dec [deg], radius [deg])
    centre = np.array([30.0, 0.0])
    radius = 2

    # Pierre Auger Observatory coordinates
    latitude_pa = -35.15
    longitude_pa = -69.15
    altitude_pa = 1425

    # Flare parameters
    flare_duration = 1 * u.day
    flare_sigma = 1.0  # deg
    flare_intensity = 0.1 # events of the flare/events expected from isotropy within the window

    # ------------------------------------------------------------------
    # Output directory
    # ------------------------------------------------------------------
    project_root = Path(__file__).resolve().parents[2]
    base_dir = project_root / "output" / "scripts"

    outdir, sim_ID = make_run_dir(
        base_dir=base_dir,
        run_code="compare_bg_signal",
        seed=seed,
    )

    # ------------------------------------------------------------------
    # Logger and metadata
    # ------------------------------------------------------------------
    logger = setup_logger(
        log_path=outdir / "run.log",
        name="compare_bg_signal",
    )

    logger.info("Starting bg vs signal comparison run")
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
    pvalues_bkg = []

    lambda_flare = []
    pvalues_flare = []

    delta_exposure_bkg = []
    delta_exposure_flare = []

    expected_n = window.expected_n_in_window(n_events)
    print(window.sky_fraction)
    print(expected_n)

    # Mean number of flare events drawn per realization. This depends only
    # on parameters fixed before the loop, so compute it once.
    mu_flare = flare_intensity * expected_n

    n_events_bkg = []
    n_events_flare = []

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

        try:
            # Isotropy
            parent_sample = stc.EventSample(
                n_events=n_events,
                t0=t0,
                tf=tf,
                rng=rng_events,
            )
            subsample = parent_sample.select_subsample(window=window)
            subsample.assign_directional_exposure(
                window=window,
                exposure_model=exposure_model,
            )
            
            delta_exposure_bkg_val = np.diff(np.sort(subsample.exposure))
            lambda_stat_bkg = stc.lambda_estimator(sample=subsample)

            # In-window event count for the background-only realisation.
            n_in_window_before = subsample.n_events

            # Draw flare multiplicity
            n_flare = int(
                scp.poisson.rvs(
                    mu_flare,
                    random_state=rng_flare,
                )
            )

            if n_flare == 0:
                logger.info(
                    "Simulation %d: drawn flare multiplicity is zero (mu=%.3f). "
                    "No flare will be injected in this realization.",
                    attempt,
                    mu_flare,
                )
                n_zero_flare += 1

            else:
                # Isotropy + Flare
                flare = stc.Flare(
                    n_events=n_flare,
                    duration=flare_duration,
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

                # We inject the flare in the parent sample and redo the window cut
                parent_sample.inject_flare(flare=flare)
                subsample = parent_sample.select_subsample(window=window)
                subsample.assign_directional_exposure(
                    window=window,
                    exposure_model=exposure_model,
                )

            # When n_flare == 0 the subsample is identical to the background
            # one, but we still compute the statistic so the trial is recorded
            # as a (zero-injection) outcome of the Poisson draw.
            delta_exposure_flare_val = np.diff(np.sort(subsample.exposure))
            lambda_stat_flare = stc.lambda_estimator(sample=subsample)

            # In-window event count after (potential) flare injection.
            n_in_window_after = subsample.n_events

            # Only store results after the full background+flare chain succeeds
            lambda_bkg.append(lambda_stat_bkg)
            lambda_flare.append(lambda_stat_flare)

            delta_exposure_bkg.append(delta_exposure_bkg_val)
            delta_exposure_flare.append(delta_exposure_flare_val)

            n_events_bkg.append(n_in_window_before)
            n_events_flare.append(n_in_window_after)

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
            f"See log file: {outdir / 'failed_simulations.log'}"
        )

    if n_success < n_simulations:
        warning_msg = (
            f"Requested {n_simulations} successful simulations, "
            f"but only obtained {n_success} before reaching "
            f"max_attempts={max_attempts}."
        )
        logger.warning(warning_msg)
        print(f"Warning: {warning_msg}")
        print(f"See log file: {outdir / 'failed_simulations.log'}")

    # ------------------------------------------------------------------
    # Convert to arrays
    # ------------------------------------------------------------------
    lambda_bkg = np.array(lambda_bkg)
    lambda_flare = np.array(lambda_flare)

    # Flatten delta-exposure lists
    delta_exposure_bkg = np.concatenate(delta_exposure_bkg)
    delta_exposure_flare = np.concatenate(delta_exposure_flare)

    n_events_bkg = np.array(n_events_bkg)
    n_events_flare = np.array(n_events_flare)

    pvalues_bkg = stc.lambda_marginal_sf(lambda_bkg, expected_n)
    pvalues_flare = stc.lambda_marginal_sf(lambda_flare, expected_n)

    # ------------------------------------------------------------------
    # Save outputs and metadata
    # ------------------------------------------------------------------
    np.savez_compressed(
        outdir / "results.npz",
        lambda_bkg=lambda_bkg,
        lambda_flare=lambda_flare,
        pvalues_bkg=pvalues_bkg,
        pvalues_flare=pvalues_flare,
        delta_exposure_bkg=delta_exposure_bkg,
        delta_exposure_flare=delta_exposure_flare,
        expected_exposure_rate=subsample.expected_exposure_rate,
        n_events_bkg=n_events_bkg,
        n_events_flare=n_events_flare,
    )

    write_metadata(
        outdir=outdir,
        metadata={
            "script": Path(__file__).name,
            "run_code": "compare_bg_signal",
            "seed": seed,
            "n_events": n_events,
            "mu_window": expected_n,
            "n_simulations_requested": n_simulations,
            "max_attempts": max_attempts,
            "t0": t0.isot,
            "tf": tf.isot,
            "T_obs_days": T_obs.to_value(u.day),
            "centre_deg": centre.tolist(),
            "radius_deg": radius,
            "latitude_pa_deg": latitude_pa,
            "longitude_pa_deg": longitude_pa,
            "altitude_pa_m": altitude_pa,
            "flare_duration_days": flare_duration.to_value(u.day),
            "flare_sigma_deg": flare_sigma,
            "expected_exposure_rate": subsample.expected_exposure_rate,
            "mu_flare": mu_flare,
        },
    )

    logger.info("Saved results to %s", outdir / "results.npz")

    end_time = time.time()
    elapsed = end_time - start_time
    logger.info(f"Simulation finished in {elapsed:.2f} seconds")

if __name__ == "__main__":
    seed = 42
    main(seed)