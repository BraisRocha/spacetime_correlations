"""
Run background simulations and compare them
with flare-injection simulations.
"""

from pathlib import Path

import astropy.units as u
import numpy as np
import scipy.stats as scp

from astropy.time import Time
from tqdm import tqdm

import spacetimecorr as stc
import spacetimecorr.plotting as stcp
from spacetimecorr.io import setup_logger, make_run_dir, write_metadata


def main(seed: int) -> None:
    """
    Generate simulated event samples and compare background-only and
    flare-injected simulations.

    Failed simulation attempts are logged and replaced by fresh random
    attempts until n_simulations successful runs are collected, or until
    max_attempts is reached.
    """

    # ------------------------------------------------------------------
    # Simulation parameters
    # ------------------------------------------------------------------
    n_events = int(1e5)
    n_simulations = int(1e3)
    max_attempts = int(3 * n_simulations)

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
    flare_duration = 0.1 * u.day
    flare_sigma = 1.0  # deg

    # ------------------------------------------------------------------
    # Output directory
    # ------------------------------------------------------------------
    project_root = Path(__file__).resolve().parents[2]
    base_dir = project_root / "output" / "scripts"

    outdir = make_run_dir(
        base_dir=base_dir,
        run_code="flare_injection",
        seed=seed,
    )

    # ------------------------------------------------------------------
    # Logger and metadata
    # ------------------------------------------------------------------
    logger = setup_logger(
        log_path=outdir / "run.log",
        name="flare_injection",
    )

    write_metadata(
        outdir=outdir,
        metadata={
            "script": Path(__file__).name,
            "run_code": "flare_injection",
            "seed": seed,
            "n_events": n_events,
            "n_simulations_requested": n_simulations,
            "max_attempts": max_attempts,
            "t0": t0.isot,
            "tf": tf.isot,
            "centre_deg": centre.tolist(),
            "radius_deg": radius,
            "latitude_pa_deg": latitude_pa,
            "longitude_pa_deg": longitude_pa,
            "altitude_pa_m": altitude_pa,
            "flare_duration_days": flare_duration.to_value(u.day),
            "flare_sigma_deg": flare_sigma,
        },
    )

    logger.info("Starting flare injection run")
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
    lambda_stats_bkg = []
    p_values_bkg = []

    lambda_flare = []
    p_values_flare = []

    n_success = 0
    n_failures = 0
    attempt = 0

    # Progress bar tracks successful simulations, not attempts
    pbar = tqdm(total=n_simulations, desc="Successful simulations")

    # ------------------------------------------------------------------
    # Main simulation loop
    # ------------------------------------------------------------------
    while n_success < n_simulations and attempt < max_attempts:
        attempt += 1

        try:
            # Generate parent sample
            parent_sample = stc.EventSample(
                n_events=n_events,
                t0=t0,
                tf=tf,
                rng=rng_events,
            )

            # Select events in the sky window
            subsample = parent_sample.select_subsample(window=window)

            # Add directional exposure
            subsample.add_directional_exposure(
                window=window,
                exposure_model=exposure_model,
            )

            # Background-only test
            lambda_stat_bkg, p_val_bkg = stc.lambda_estimator(sample=subsample)

            # Draw flare multiplicity
            n_flare = int(
                scp.poisson.rvs(
                    0.5 * subsample.expected_counts,
                    random_state=rng_flare,
                )
            )

            # Create and generate flare
            flare = stc.Flare(
                n_events=n_flare,
                duration=flare_duration,
                t0=t0,
                tf=tf,
                centre=window.centre,
                exposure=exposure_model,
                rng=rng_flare,
            )

            flare.generate_in_window(
                window=window,
                sigma=flare_sigma,
            )

            # Inject flare and re-run statistic
            subsample.inject_flare(flare=flare)

            lambda_stat_flare, p_val_flare = stc.lambda_estimator(sample=subsample)

            # Only store results after the full background+flare chain succeeds
            lambda_stats_bkg.append(lambda_stat_bkg)
            p_values_bkg.append(p_val_bkg)
            lambda_flare.append(lambda_stat_flare)
            p_values_flare.append(p_val_flare)

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
    lambda_stats_bkg = np.array(lambda_stats_bkg)
    p_values_bkg = np.array(p_values_bkg)

    lambda_flare = np.array(lambda_flare)
    p_values_flare = np.array(p_values_flare)

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------
    stcp.plot_lambda_estimator(
        {
            "Isotropy": lambda_stats_bkg,
            "Flare": lambda_flare,
        },
        outdir / "lambda.png",
    )

    stcp.plot_p_value(
        {
            "Isotropy": p_values_bkg,
            "Flare": p_values_flare,
        },
        outdir / "p_values.png",
    )

    logger.info("Saved plots to %s", outdir)


if __name__ == "__main__":
    seed = 42
    main(seed)