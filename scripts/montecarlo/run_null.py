"""
Run null-hypothesis (pure isotropy) simulations.

This script:
    - generates isotropic event samples directly inside a sky window via
      ``EventSample.in_window``,
    - computes the Lambda estimator,
    - computes p-values using two methods:
        - p-value(x | n),
        - p-value(x),
    - computes the spatial p-value from the realised in-window count.

Pipeline note
-------------
Each trial draws ``n_sample ~ Poisson(expected_n)`` directly inside the
window (exposure-weighted), so the spatial p-value
``Poisson.sf(n_sample - 1, expected_n)`` is uniform on ``[0, 1]`` by
construction (Poisson-SF of a Poisson-drawn count).
"""

from pathlib import Path

import astropy.units as u
import numpy as np
import scipy.stats as scp
import time

from astropy.time import Time
from tqdm import tqdm

import spacetimecorr as stc
from spacetimecorr.io import setup_logger, make_run_dir, write_metadata


def main(seed: int) -> None:
    """Generate background-only in-window samples and compute Lambda + p-values."""
    start_time = time.time()

    # ------------------------------------------------------------------
    # Simulation parameters
    # ------------------------------------------------------------------
    n_total = int(5e5)
    n_simulations = int(1e4)
    max_attempts = int(3 * n_simulations)

    # Observation interval
    T_obs = 1 * u.week
    t0 = Time("2026-01-01T00:00:00", scale="utc")
    tf = t0 + T_obs

    # Sky window parameters (RA [deg], Dec [deg], radius [deg])
    centre = np.array([30.0, 0.0])
    radius = 2.0

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
        run_code="null",
        seed=seed,
    )

    # ------------------------------------------------------------------
    # Logger and metadata
    # ------------------------------------------------------------------
    logger = setup_logger(log_path=outdir / "run.log", name="null")

    logger.info("Starting null run")
    logger.info("Simulation ID: %s", sim_ID)
    logger.info("Output directory: %s", outdir)
    logger.info("Seed: %d", seed)

    # ------------------------------------------------------------------
    # RNGs and models
    # ------------------------------------------------------------------
    rng_manager = stc.RNGManager(seed=seed)
    rng_events = rng_manager.get("events")
    rng_exposure = rng_manager.get("exposure")

    window = stc.SkyWindow(centre=centre, radius=radius)
    observatory = stc.Observatory(
        latitude=latitude_pa, longitude=longitude_pa, altitude=altitude_pa,
    )
    exposure_model = stc.ExposureModel(
        observatory=observatory, t0=t0, tf=tf, rng=rng_exposure,
    )

    # Exposure-weighted expected count in the window. Fixed across trials.
    expected_n = window.expected_n_in_window(n_total, exposure_model)

    # ------------------------------------------------------------------
    # Storage
    # ------------------------------------------------------------------
    lambda_mc = []
    n_sample_window = []

    n_success = 0
    n_failures = 0
    attempt = 0

    pbar = tqdm(total=n_simulations, desc="Successful simulations")

    # ------------------------------------------------------------------
    # Main simulation loop
    # ------------------------------------------------------------------
    # Only stochastic-rejection failures (RuntimeError) are retried;
    # configuration/contract violations (ValueError) propagate and crash
    # the run so the operator notices.
    while n_success < n_simulations and attempt < max_attempts:
        attempt += 1

        try:
            sample = stc.EventSample.in_window(
                window=window,
                n_total=n_total,
                exposure_model=exposure_model,
                t0=t0,
                tf=tf,
                rng=rng_events,
            )
            sample.assign_directional_exposure(
                window=window, exposure_model=exposure_model,
            )

            lambda_stat_mc = stc.lambda_estimator(sample=sample)

            lambda_mc.append(lambda_stat_mc)
            n_sample_window.append(sample.n_sample)

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
        print(f"See log file: {outdir / 'run.log'}")

    # ------------------------------------------------------------------
    # Convert to arrays and compute parameters
    # ------------------------------------------------------------------
    lambda_mc = np.array(lambda_mc)
    n_sample_window = np.array(n_sample_window)

    p_values_conditional = stc.lambda_conditional_sf(lambda_mc, n_sample_window)
    p_values_marginal = stc.lambda_marginal_sf(lambda_mc, expected_n)
    p_values_spatial = scp.poisson.sf(n_sample_window - 1, expected_n)

    # ------------------------------------------------------------------
    # Save outputs and metadata
    # ------------------------------------------------------------------
    np.savez_compressed(
        outdir / "results.npz",
        lambda_mc=lambda_mc,
        p_values_conditional=p_values_conditional,
        p_values_marginal=p_values_marginal,
        p_values_spatial=p_values_spatial,
        n_sample_window=n_sample_window,
    )

    write_metadata(
        outdir=outdir,
        metadata={
            "script": Path(__file__).name,
            "run_code": "null",
            "seed": seed,
            "n_total": n_total,
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
    elapsed = time.time() - start_time
    logger.info(f"Simulation finished in {elapsed:.2f} seconds")


if __name__ == "__main__":
    seed = 42
    main(seed)
