"""
Scan the Lambda statistic across a range of flare signal-to-noise ratios
at fixed observation time and flare duration.

For each Monte Carlo trial, a background-only in-window sample is
generated once and Lambda is computed. The same parent sample is then
reused as the substrate for several flare injections, one per intensity
in ``flare_intensity``, producing a ``(n_intensities, n_simulations)``
matrix of Lambda values. Sharing the parent sample across intensities
reduces background-driven variance in the intensity-to-intensity
comparison.

Outputs are written to ``output/scripts/scan_intensity/<sim_ID>/``
as ``results.npz`` plus a ``metadata.json`` describing the run.
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
    """Run the flare-intensity scan."""
    start_time = time.time()

    # ------------------------------------------------------------------
    # Simulation parameters
    # ------------------------------------------------------------------
    n_total = int(2e5)
    n_simulations = int(1e4)
    max_attempts = int(3 * n_simulations)

    # Observation interval
    T_obs = 1 * u.year
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
    flare_intensity = np.array([0.1, 0.2, 0.3, 0.4, 0.5])  # S/N ratio

    # ------------------------------------------------------------------
    # Output directory
    # ------------------------------------------------------------------
    project_root = Path(__file__).resolve().parents[2]
    base_dir = project_root / "output" / "scripts"

    outdir, sim_ID = make_run_dir(
        base_dir=base_dir,
        run_code="scan_intensity",
        seed=seed,
    )

    # ------------------------------------------------------------------
    # Logger and metadata
    # ------------------------------------------------------------------
    logger = setup_logger(log_path=outdir / "run.log", name="scan_intensity")

    logger.info("Starting intensity scan run")
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

    logger.info("expected_n=%g, mu_flare per intensity=%s", expected_n, mu_flare.tolist())

    # ------------------------------------------------------------------
    # Storage
    # ------------------------------------------------------------------
    n_intensities = len(flare_intensity)
    lambda_bkg = np.zeros(n_simulations)
    lambda_flare = np.zeros((n_intensities, n_simulations))

    n_success = 0
    n_failures = 0
    attempt = 0
    n_zero_flare = np.zeros(n_intensities, dtype=int)

    pbar = tqdm(total=n_simulations, desc="Successful simulations")

    # ------------------------------------------------------------------
    # Main simulation loop
    # ------------------------------------------------------------------
    while n_success < n_simulations and attempt < max_attempts:
        attempt += 1

        try:
            # Background-only sample drawn directly inside the window
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

            lambda_bkg[n_success] = stc.lambda_estimator(sample=bkg_sample)

            for i, mu in enumerate(mu_flare):
                n_flare = int(scp.poisson.rvs(mu, random_state=rng_flare))

                if n_flare == 0:
                    logger.info(
                        "Simulation %d, intensity %.2f: drawn flare multiplicity "
                        "is zero (mu_flare=%.3f). No flare will be injected.",
                        attempt, flare_intensity[i], mu,
                    )
                    n_zero_flare[i] += 1
                    sample = bkg_sample
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

                    sample = copy.deepcopy(bkg_sample)
                    sample.inject_flare(flare=flare)
                    sample.assign_directional_exposure(
                        window=window, exposure_model=exposure_model,
                    )

                lambda_flare[i, n_success] = stc.lambda_estimator(sample=sample)

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
    for i, f in enumerate(flare_intensity):
        logger.info(
            "Zero-flare realizations at intensity %.2f: %d / %d (%.2f%%)",
            f, n_zero_flare[i], n_success,
            100.0 * n_zero_flare[i] / max(n_success, 1),
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

    # Trim unused trailing slots when max_attempts was reached
    if n_success < n_simulations:
        unfilled = n_simulations - n_success
        logger.warning(
            "Result matrix not fully filled: %d of %d slots are empty. "
            "Trimming arrays to (n_intensities=%d, n_success=%d).",
            unfilled, n_simulations, n_intensities, n_success,
        )
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
        expected_exposure_rate=bkg_sample.expected_exposure_rate,
    )

    elapsed = time.time() - start_time

    write_metadata(
        outdir=outdir,
        metadata={
            "script": Path(__file__).name,
            "run_code": "scan_intensity",
            "seed": seed,
            "runtime_seconds": elapsed,
            "n_total": n_total,
            "expected_n": expected_n,
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
