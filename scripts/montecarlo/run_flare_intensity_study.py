"""
Flare-intensity study: scan the Lambda statistic across a range of
flare signal-to-noise ratios at fixed observation time and flare duration.

For each Monte Carlo trial, an isotropic background sample is generated
once and Lambda is computed. The same parent sample is then reused as
the substrate for several flare injections, one per intensity in
``flare_intensity``, producing a ``(n_intensities, n_simulations)``
matrix of Lambda values. Sharing the parent sample across intensities
reduces background-driven variance in the intensity-to-intensity
comparison.

Outputs are written to ``output/scripts/flare_intensity_study/<sim_ID>/``
as ``results.npz`` plus a ``metadata.json`` describing the run.
"""

from pathlib import Path

import astropy.units as u
import numpy as np
import scipy.stats as scp

from astropy.time import Time
from tqdm import tqdm
import time
import copy

import spacetimecorr as stc
from spacetimecorr.io import setup_logger, make_run_dir, write_metadata


def main(seed: int) -> None:
    """
    Run the flare-intensity scan.

    Per trial:
        1. Generate an isotropic parent sample over ``T_obs``.
        2. Window-select and compute ``lambda_bkg``.
        3. For each intensity in ``flare_intensity``: deepcopy the parent
           sample, draw a Poisson flare multiplicity with mean
           ``intensity * mu_window``, inject the flare (if non-zero),
           re-window, and compute ``lambda_flare[i, trial]``.

    Failed trials are logged and replaced by fresh attempts until either
    ``n_simulations`` successes are collected or ``max_attempts`` is
    reached. In the latter case the result arrays are trimmed to the
    number of completed trials and a warning is emitted.
    """

    start_time = time.time()

    # ------------------------------------------------------------------
    # Simulation parameters
    # ------------------------------------------------------------------
    n_events = int(1e5)
    n_simulations = int(1e4)
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
    flare_duration = 30 * u.day
    flare_sigma = 1.0  # deg
    flare_intensity = np.array([0.1, 0.2, 0.3, 0.4, 0.5]) # Signal to Noise ratio

    # ------------------------------------------------------------------
    # Output directory
    # ------------------------------------------------------------------
    project_root = Path(__file__).resolve().parents[2]
    base_dir = project_root / "output" / "scripts"

    outdir, sim_ID = make_run_dir(
        base_dir=base_dir,
        run_code="flare_intensity_study",
        seed=seed,
    )

    # ------------------------------------------------------------------
    # Logger and metadata
    # ------------------------------------------------------------------
    logger = setup_logger(
        log_path=outdir / "run.log",
        name="flare_intensity_study",
    )

    logger.info("Starting flare intensity study run")
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
    n_intensities = len(flare_intensity)

    lambda_bkg = np.zeros(n_simulations)
    lambda_flare = np.zeros((n_intensities, n_simulations))

    expected_n = window.expected_n_in_window(n_events)

    # Mean number of flare events drawn per realization. This depends only
    # on parameters fixed before the loop, so compute it once.
    mu_flare = flare_intensity * expected_n

    n_success = 0
    n_failures = 0
    attempt = 0
    n_zero_flare = np.zeros(n_intensities, dtype=int)

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

            bkg_subsample = parent_sample.select_subsample(window=window)
            bkg_subsample.assign_directional_exposure(
                window=window,
                exposure_model=exposure_model,
            )

            lambda_bkg[n_success] = stc.lambda_estimator(sample=bkg_subsample)

            for i, mu in enumerate(mu_flare):
                working_sample = copy.deepcopy(parent_sample)

                # Draw flare multiplicity
                n_flare = int(
                    scp.poisson.rvs(
                        mu,
                        random_state=rng_flare,
                    )
                )

                if n_flare == 0:
                    logger.info(
                        "Simulation %d, intensity %.2f: drawn flare multiplicity "
                        "is zero (mu_flare=%.3f). No flare will be injected.",
                        attempt,
                        flare_intensity[i],
                        mu,
                    )
                    n_zero_flare[i] += 1
                    subsample = bkg_subsample

                else:
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

                    working_sample.inject_flare(flare=flare)
                    subsample = working_sample.select_subsample(window=window)
                    subsample.assign_directional_exposure(
                        window=window,
                        exposure_model=exposure_model,
                    )

                # When n_flare == 0 the subsample equals the background one,
                # but we still record the statistic so the trial is counted
                # as a (zero-injection) outcome of the Poisson draw.
                lambda_flare[i, n_success] = stc.lambda_estimator(sample=subsample)

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
    for i, f in enumerate(flare_intensity):
        logger.info(
            "Zero-flare realizations at intensity %.2f: %d / %d (%.2f%%)",
            f,
            n_zero_flare[i],
            n_success,
            100.0 * n_zero_flare[i] / max(n_success, 1),
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
    # Trim unused trailing slots if max_attempts was reached
    # ------------------------------------------------------------------
    if n_success < n_simulations:
        unfilled = n_simulations - n_success
        warning_msg = (
            f"Result matrix not fully filled: {unfilled} of {n_simulations} "
            f"slots are empty (n_success={n_success}). Trimming arrays to "
            f"shape (n_intensities, {n_success})."
        )
        logger.warning(warning_msg)
        print(f"Warning: {warning_msg}")

        lambda_bkg = lambda_bkg[:n_success]
        lambda_flare = lambda_flare[:, :n_success]

    pvalues_bkg = stc.lambda_marginal_sf(lambda_bkg, expected_n)
    pvalues_flare = stc.lambda_marginal_sf(
        lambda_flare.ravel(), expected_n
    ).reshape(lambda_flare.shape)

    # ------------------------------------------------------------------
    # Save outputs and metadata
    # ------------------------------------------------------------------
    np.savez_compressed(
        outdir / "results.npz",
        lambda_bkg=lambda_bkg,
        lambda_flare=lambda_flare,
        pvalues_bkg=pvalues_bkg,
        pvalues_flare=pvalues_flare,
        flare_intensity=flare_intensity,
        expected_exposure_rate=bkg_subsample.expected_exposure_rate,
    )

    elapsed = time.time() - start_time

    write_metadata(
        outdir=outdir,
        metadata={
            "script": Path(__file__).name,
            "run_code": "flare_intensity_study",
            "seed": seed,
            "runtime_seconds": elapsed,
            "n_events": n_events,
            "mu_window": expected_n,
            "n_simulations_requested": n_simulations,
            "n_simulations_completed": n_success,
            "max_attempts": max_attempts,
            "n_zero_flare_per_intensity": n_zero_flare.tolist(),
            "time": {
                "t0": t0.isot,
                "tf": tf.isot,
                "T_obs_days": T_obs.to_value(u.day),
            },
            "window": {
                "centre_deg": centre.tolist(),
                "radius_deg": radius,
            },
            "observatory": {
                "latitude_deg": latitude_pa,
                "longitude_deg": longitude_pa,
                "altitude_m": altitude_pa,
            },
            "flare": {
                "duration_days": flare_duration.to_value(u.day),
                "sigma_deg": flare_sigma,
                "intensity": flare_intensity.tolist(),
            },
        },
    )

    logger.info("Saved results to %s", outdir / "results.npz")
    logger.info(f"Simulation finished in {elapsed:.2f} seconds")

if __name__ == "__main__":
    seed = 42
    main(seed)