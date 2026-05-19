"""
Compare background-only and flare-injected Lambda distributions.

For each trial:
    1. Generate one isotropic in-window sample directly via
       ``EventSample.in_window`` and attach exposure.
    2. Compute background Lambda.
    3. Draw ``n_flare ~ Poisson(flare_intensity * expected_n)``.
       If ``n_flare > 0``, generate and inject the flare in place. The
       flare *replaces* ``n_flare`` random slots in the sample, so
       ``n_sample`` is preserved (this is the new in-window pipeline's
       inject semantics).
    4. Compute the post-injection Lambda.

If ``n_flare > n_sample`` (very rare for the parameters used here)
``inject_flare`` raises ``ValueError`` and the script terminates with a
loud error so the operator can adjust ``n_total``, ``flare_intensity``,
or the window geometry.
"""

from pathlib import Path

import astropy.units as u
import numpy as np
import scipy.stats as scp
import time
import copy

from astropy.time import Time
from tqdm import tqdm

import spacetimecorr as stc
from spacetimecorr.io import setup_logger, make_run_dir, write_metadata


def main(seed: int) -> None:
    start_time = time.time()

    # ------------------------------------------------------------------
    # Simulation parameters
    # ------------------------------------------------------------------
    n_total = int(2e5)
    n_simulations = int(1e3)
    max_attempts = int(3 * n_simulations)

    # Observation interval
    T_obs = 10 * u.year
    t0 = Time("2013-01-01T00:00:00", scale="utc")
    tf = t0 + T_obs

    # Sky window
    centre = np.array([30.0, 0.0])
    radius = 2

    # Pierre Auger Observatory
    latitude_pa = -35.15
    longitude_pa = -69.15
    altitude_pa = 1425

    # Flare parameters
    flare_duration = 1 * u.day
    flare_sigma = 1.0  # deg
    flare_intensity = 0.  # n_flare / expected_n

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
    logger = setup_logger(log_path=outdir / "run.log", name="compare_bg_signal")

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
        latitude=latitude_pa, longitude=longitude_pa, altitude=altitude_pa,
    )
    exposure_model = stc.ExposureModel(
        observatory=observatory, t0=t0, tf=tf, rng=rng_exposure,
    )

    expected_n = window.expected_n_in_window(n_total, exposure_model)
    mu_flare = flare_intensity * expected_n

    logger.info("sky_fraction=%g, expected_n=%g, mu_flare=%g",
                window.sky_fraction, expected_n, mu_flare)

    # ------------------------------------------------------------------
    # Storage
    # ------------------------------------------------------------------
    lambda_bkg = []
    lambda_flare_list = []
    delta_exposure_bkg = []
    delta_exposure_flare = []
    n_sample_bkg = []
    n_sample_flare = []

    n_success = 0
    n_failures = 0
    attempt = 0
    n_zero_flare = 0

    pbar = tqdm(total=n_simulations, desc="Successful simulations")

    # ------------------------------------------------------------------
    # Main simulation loop
    # ------------------------------------------------------------------
    while n_success < n_simulations and attempt < max_attempts:
        attempt += 1

        try:
            # --- Background-only ---
            bkg_sample = stc.EventSample.in_window(
                window=window,
                n_total=n_total,
                exposure_model=exposure_model,
                t0=t0,
                tf=tf,
                rng=rng_events,
            )
            bkg_sample.assign_directional_exposure(
                window=window, exposure_model=exposure_model,
            )

            lambda_stat_bkg = stc.lambda_estimator(sample=bkg_sample)
            delta_exposure_bkg_val = np.diff(np.sort(bkg_sample.exposure))
            n_in_window_bkg = bkg_sample.n_sample

            # --- Flare injection on a deep copy ---
            n_flare = int(scp.poisson.rvs(mu_flare, random_state=rng_flare))

            if n_flare == 0:
                logger.info(
                    "Simulation %d: drawn flare multiplicity is zero (mu=%.3f). "
                    "No flare will be injected in this realization.",
                    attempt, mu_flare,
                )
                n_zero_flare += 1
                flare_sample = bkg_sample

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

                flare_sample = copy.deepcopy(bkg_sample)
                flare_sample.inject_flare(flare=flare)
                flare_sample.assign_directional_exposure(
                    window=window, exposure_model=exposure_model,
                )

            lambda_stat_flare = stc.lambda_estimator(sample=flare_sample)
            delta_exposure_flare_val = np.diff(np.sort(flare_sample.exposure))
            n_in_window_flare = flare_sample.n_sample

            # Record only after the full chain succeeds.
            lambda_bkg.append(lambda_stat_bkg)
            lambda_flare_list.append(lambda_stat_flare)
            delta_exposure_bkg.append(delta_exposure_bkg_val)
            delta_exposure_flare.append(delta_exposure_flare_val)
            n_sample_bkg.append(n_in_window_bkg)
            n_sample_flare.append(n_in_window_flare)

            n_success += 1
            pbar.update(1)

        except RuntimeError:
            n_failures += 1
            logger.exception(
                "Simulation attempt %d failed "
                "(successes=%d, failures=%d)",
                attempt, n_success, n_failures,
            )
            continue

    pbar.close()

    logger.info(
        "Run finished: attempts=%d, successes=%d, failures=%d",
        attempt, n_success, n_failures,
    )
    logger.info(
        "Zero-flare realizations: %d / %d (%.2f%%)",
        n_zero_flare, attempt, 100.0 * n_zero_flare / max(attempt, 1),
    )

    # ------------------------------------------------------------------
    # Final checks
    # ------------------------------------------------------------------
    if n_success == 0:
        raise RuntimeError(
            f"All simulation attempts failed. See log file: {outdir / 'run.log'}"
        )

    if n_success < n_simulations:
        warning_msg = (
            f"Requested {n_simulations} successful simulations, "
            f"but only obtained {n_success} before reaching "
            f"max_attempts={max_attempts}."
        )
        logger.warning(warning_msg)
        print(f"Warning: {warning_msg}")

    # ------------------------------------------------------------------
    # Convert to arrays
    # ------------------------------------------------------------------
    lambda_bkg = np.array(lambda_bkg)
    lambda_flare = np.array(lambda_flare_list)
    delta_exposure_bkg = np.concatenate(delta_exposure_bkg)
    delta_exposure_flare = np.concatenate(delta_exposure_flare)
    n_sample_bkg = np.array(n_sample_bkg)
    n_sample_flare = np.array(n_sample_flare)

    pvalues_bkg = stc.lambda_marginal_sf(lambda_bkg, expected_n)
    pvalues_flare = stc.lambda_marginal_sf(lambda_flare, expected_n)

    # ------------------------------------------------------------------
    # Save outputs and metadata
    # ------------------------------------------------------------------
    # `expected_exposure_rate` is the same for bkg / flare since both attach
    # exposure with the same (window, exposure_model) pair.
    expected_exposure_rate = bkg_sample.expected_exposure_rate

    np.savez_compressed(
        outdir / "results.npz",
        lambda_bkg=lambda_bkg,
        lambda_flare=lambda_flare,
        pvalues_bkg=pvalues_bkg,
        pvalues_flare=pvalues_flare,
        delta_exposure_bkg=delta_exposure_bkg,
        delta_exposure_flare=delta_exposure_flare,
        expected_exposure_rate=expected_exposure_rate,
        n_sample_bkg=n_sample_bkg,
        n_sample_flare=n_sample_flare,
    )

    write_metadata(
        outdir=outdir,
        metadata={
            "script": Path(__file__).name,
            "run_code": "compare_bg_signal",
            "seed": seed,
            "n_total": n_total,
            "expected_n": expected_n,
            "n_simulations_requested": n_simulations,
            "n_simulations_successful": n_success,
            "max_attempts": max_attempts,
            "n_zero_flare": n_zero_flare,
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
            "flare_intensity": flare_intensity,
            "mu_flare": mu_flare,
            "expected_exposure_rate": expected_exposure_rate,
        },
    )

    logger.info("Saved results to %s", outdir / "results.npz")
    elapsed = time.time() - start_time
    logger.info(f"Simulation finished in {elapsed:.2f} seconds")


if __name__ == "__main__":
    seed = 42
    main(seed)
