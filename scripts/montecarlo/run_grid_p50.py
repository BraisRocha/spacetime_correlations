"""
Single-point worker for the (duration, intensity) 2D p50 grid.

Designed to run as one HTCondor job per (flare_duration, flare_intensity)
grid cell. Each job:

    1. Generates ``n_simulations`` in-window background realizations via
       ``EventSample.in_window``.
    2. Injects a flare of the requested duration and signal-to-noise
       ratio on top of each background sample (in-place replacement).
    3. Saves only the 50th percentile of the flare-injected Lambda
       distribution.

The ``flare_duration`` (in days), ``flare_intensity`` (S/N), ``seed``,
and an optional ``job_id`` are passed in as command-line arguments by
the Condor submit file. Outputs are written to
``output/scripts/grid_p50/<run_name>/``.
"""

import argparse
import os
import sys
import time
from pathlib import Path

import astropy.units as u
import numpy as np
import scipy.stats as scp
from astropy.time import Time

import spacetimecorr as stc
from spacetimecorr.io import setup_logger, make_run_dir, write_metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run one (flare_duration, flare_intensity) point of the "
            "Lambda 50th-percentile heatmap."
        )
    )
    parser.add_argument(
        "--flare-duration-days",
        type=float,
        required=True,
        help="Flare duration in days.",
    )
    parser.add_argument(
        "--flare-intensity",
        type=float,
        required=True,
        help="Flare signal-to-noise ratio (mean flare events / expected_n).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        required=True,
        help="Random seed for this job.",
    )
    parser.add_argument(
        "--job-id",
        type=str,
        default=None,
        help=(
            "Optional job identifier (e.g. Condor $(Process)). Used to "
            "name per-job output files within a shared submission directory."
        ),
    )
    parser.add_argument(
        "--submission-id",
        type=str,
        default=None,
        help=(
            "Identifier shared by all jobs in one submission batch "
            "(e.g. a timestamp or Condor ClusterId). When provided all jobs "
            "write into the same directory and files are suffixed by job_id."
        ),
    )
    return parser.parse_args()


def main(
    seed: int,
    flare_duration_days: float,
    flare_intensity_value: float,
    job_id: str | None,
    submission_id: str | None,
) -> None:
    start_time = time.time()

    # ------------------------------------------------------------------
    # Simulation parameters
    # ------------------------------------------------------------------
    n_total = int(1e5)
    n_simulations = int(1e4)
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

    # Flare parameters (single point in the grid)
    flare_duration = flare_duration_days * u.day
    flare_sigma = 1.0  # deg
    flare_intensity = float(flare_intensity_value)

    # ------------------------------------------------------------------
    # Output directory
    # ------------------------------------------------------------------
    project_root = Path(__file__).resolve().parents[2]
    base_dir = project_root / "output" / "scripts"

    outdir, sim_ID = make_run_dir(
        base_dir=base_dir,
        run_code="grid_p50",
        seed=seed,
        job_id=job_id,
        submission_id=submission_id,
    )

    data_dir = outdir / "data"
    data_dir.mkdir(exist_ok=True)

    job_suffix = f"_job{job_id}" if job_id is not None else ""

    # ------------------------------------------------------------------
    # Logger and metadata
    # ------------------------------------------------------------------
    logger = setup_logger(
        log_path=data_dir / f"run{job_suffix}.log",
        name="grid_p50",
    )

    logger.info("Starting p50 grid run")
    logger.info("Simulation ID: %s", sim_ID)
    logger.info("Output directory: %s", outdir)
    logger.info("Seed: %d", seed)
    logger.info("Job ID: %s", job_id)
    logger.info(
        "Flare duration: %.3f day, intensity (S/N): %.4f",
        flare_duration.to_value(u.day),
        flare_intensity,
    )

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

    # ------------------------------------------------------------------
    # Storage
    # ------------------------------------------------------------------
    lambda_flare = np.zeros(n_simulations)
    n_sample_window = np.zeros(n_simulations, dtype=int)

    n_success = 0
    n_failures = 0
    attempt = 0
    n_zero_flare = 0

    # Periodic progress logging interval (no tqdm: stdout/stderr in HTCondor
    # are non-interactive log files, where tqdm output is noisy).
    progress_step = max(1, n_simulations // 20)

    # ------------------------------------------------------------------
    # Main simulation loop
    # ------------------------------------------------------------------
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

            n_flare = int(scp.poisson.rvs(mu_flare, random_state=rng_flare))

            if n_flare == 0:
                logger.info(
                    "Simulation %d: drawn flare multiplicity is zero "
                    "(mu_flare=%.3f). No flare will be injected.",
                    attempt, mu_flare,
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

            sample.assign_directional_exposure(
                window=window, exposure_model=exposure_model,
            )

            lambda_flare[n_success] = stc.lambda_estimator(sample=sample)
            n_sample_window[n_success] = sample.n_sample

            n_success += 1

            if n_success % progress_step == 0 or n_success == n_simulations:
                logger.info(
                    "Progress: %d / %d successful simulations "
                    "(attempts=%d, failures=%d)",
                    n_success, n_simulations, attempt, n_failures,
                )

        except RuntimeError:
            n_failures += 1
            logger.exception(
                "Simulation attempt %d failed "
                "(successes=%d, failures=%d)",
                attempt, n_success, n_failures,
            )
            continue

    logger.info(
        "Run finished: attempts=%d, successes=%d, failures=%d",
        attempt, n_success, n_failures,
    )
    logger.info(
        "Zero-flare realizations: %d / %d (%.2f%%)",
        n_zero_flare, max(n_success, 1),
        100.0 * n_zero_flare / max(n_success, 1),
    )

    # ------------------------------------------------------------------
    # Final checks
    # ------------------------------------------------------------------
    if n_success == 0:
        logger.error("All simulation attempts failed.")
        raise RuntimeError(
            f"All simulation attempts failed. See log file: {outdir / 'run.log'}"
        )

    if n_success < n_simulations:
        logger.warning(
            "Requested %d successful simulations, but only obtained %d "
            "before reaching max_attempts=%d. Trimming arrays.",
            n_simulations, n_success, max_attempts,
        )
        lambda_flare = lambda_flare[:n_success]
        n_sample_window = n_sample_window[:n_success]

    # ------------------------------------------------------------------
    # 50th percentiles
    # ------------------------------------------------------------------
    lambda_flare_p50 = float(np.percentile(lambda_flare, 50))
    n_sample_window_p50 = float(np.percentile(n_sample_window, 50))

    logger.info("lambda_flare p50: %.6e", lambda_flare_p50)
    logger.info("n_sample_window p50: %.1f", n_sample_window_p50)

    np.savez_compressed(
        data_dir / f"results{job_suffix}.npz",
        lambda_flare_p50=lambda_flare_p50,
        n_sample_window_p50=n_sample_window_p50,
        flare_duration_days=flare_duration.to_value(u.day),
        flare_intensity=flare_intensity,
    )

    elapsed = time.time() - start_time

    write_metadata(
        outdir=data_dir,
        filename=f"metadata{job_suffix}.json",
        metadata={
            "script": Path(__file__).name,
            "run_code": "grid_p50",
            "seed": seed,
            "job_id": job_id,
            "condor_cluster": os.environ.get("CONDOR_CLUSTER_ID"),
            "condor_process": os.environ.get("CONDOR_PROCESS_ID"),
            "runtime_seconds": elapsed,
            "n_total": n_total,
            "expected_n": expected_n,
            "n_simulations_requested": n_simulations,
            "n_simulations_completed": n_success,
            "max_attempts": max_attempts,
            "n_zero_flare": int(n_zero_flare),
            "lambda_flare_p50": lambda_flare_p50,
            "n_sample_window_p50": n_sample_window_p50,
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
                "intensity": flare_intensity,
            },
        },
    )

    logger.info("Saved results to %s", data_dir / f"results{job_suffix}.npz")
    logger.info("Simulation finished in %.2f seconds", elapsed)


if __name__ == "__main__":
    # Line-buffer stdout/stderr so HTCondor .out/.err files reflect
    # progress promptly even if the job is killed mid-run.
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)

    args = parse_args()
    main(
        seed=args.seed,
        flare_duration_days=args.flare_duration_days,
        flare_intensity_value=args.flare_intensity,
        job_id=args.job_id,
        submission_id=args.submission_id,
    )
