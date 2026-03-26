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
    T_obs = 1 * u.week
    t0 = Time("2026-01-01T00:00:00", scale="utc")
    tf = t0 + T_obs

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
    lambda_bkg = []
    p_values_bkg = []

    lambda_flare = []
    p_values_flare = []

    delta_exposure_bkg = []
    delta_exposure_flare = []

    spatial_p_values = []

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
            # Isotropy
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
            
            delta_exposure_bkg_val = np.diff(np.sort(subsample.dir_exposure))
            lambda_stat_bkg, p_val_bkg = stc.lambda_estimator(sample=subsample)

            # Spatial-only estimator

            spatial_p_val = stc.spatial_estimator(subsample)

            # Draw flare multiplicity
            mu_flare = 0.5 * subsample.expected_counts
            n_flare = int(
                scp.poisson.rvs(
                    mu_flare,
                    random_state=rng_flare,
                )
            )

            n_zero_flare = 0
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
                    exposure=exposure_model,
                    rng=rng_flare,
                )
                flare.generate_in_window(
                    window=window,
                    sigma=flare_sigma,
                )

                subsample.inject_flare(flare=flare)

            # Although n_flare may be 0 parameters are still obtained
            delta_exposure_flare_val = np.diff(np.sort(subsample.dir_exposure))
            lambda_stat_flare, p_val_flare = stc.lambda_estimator(sample=subsample)

            # Only store results after the full background+flare chain succeeds
            lambda_bkg.append(lambda_stat_bkg)
            p_values_bkg.append(p_val_bkg)
            lambda_flare.append(lambda_stat_flare)
            p_values_flare.append(p_val_flare)

            delta_exposure_bkg.append(delta_exposure_bkg_val)
            delta_exposure_flare.append(delta_exposure_flare_val)

            spatial_p_values.append(spatial_p_val)

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
    p_values_bkg = np.array(p_values_bkg)

    lambda_flare = np.array(lambda_flare)
    p_values_flare = np.array(p_values_flare)

    # Flatten delta-exposure lists
    delta_exposure_bkg = np.concatenate(delta_exposure_bkg)
    delta_exposure_flare = np.concatenate(delta_exposure_flare)

    spatial_p_values = np.array(spatial_p_values)

    # ------------------------------------------------------------------
    # Save outputs and metadata
    # ------------------------------------------------------------------
    np.savez_compressed(
        outdir / "results.npz",
        lambda_bkg=lambda_bkg,
        p_values_bkg=p_values_bkg,
        lambda_flare=lambda_flare,
        p_values_flare=p_values_flare,
        delta_exposure_bkg=delta_exposure_bkg,
        delta_exposure_flare=delta_exposure_flare,
        exp_rate_exposure=subsample.exp_rate_exposure,
        spatial_p_values=spatial_p_values,
    )

    write_metadata(
        outdir=outdir,
        metadata={
            "script": Path(__file__).name,
            "run_code": "flare_injection",
            "seed": seed,
            "n_events": n_events,
            "mu_window": subsample.expected_counts,
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
            "exp_rate_exposure": subsample.exp_rate_exposure,
            "mu_flare": mu_flare,
        },
    )

    logger.info("Saved results to %s", outdir / "results.npz")

if __name__ == "__main__":
    seed = 42
    main(seed)